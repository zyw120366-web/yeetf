from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results" / "ye_strategy"
ACCOUNT_STATE = ROOT / "results" / "live" / "account_state.json"


def active_symbol(row: pd.Series) -> str | None:
    active = row[pd.to_numeric(row, errors="coerce").fillna(0.0).gt(0.0)]
    return str(active.index[0]) if len(active) else None


def load_account_state(day: pd.Timestamp) -> dict:
    if not ACCOUNT_STATE.exists():
        raise RuntimeError("confirmed live account state is missing; do not infer holdings from the backtest")
    account = json.loads(ACCOUNT_STATE.read_text(encoding="utf-8"))
    if account.get("confirmation_status") != "confirmed":
        raise RuntimeError("live account state is not confirmed; fail closed")
    account_day = pd.Timestamp(str(account.get("as_of", "")).split("_")[0])
    if account_day != day:
        raise RuntimeError(
            f"live account state is stale ({account_day.date()}); confirm cash, positions and pending orders for {day.date()}"
        )
    positions = [item for item in account.get("positions", []) if float(item.get("quantity", 0.0)) > 0]
    if len(positions) > 1:
        raise RuntimeError("ye live account may contain at most one positive ETF position")
    if account.get("pending_orders"):
        raise RuntimeError("live account has pending orders; reconcile before creating a new buy plan")
    if float(account.get("available_cash", -1.0)) < 0 or float(account.get("total_equity", -1.0)) <= 0:
        raise RuntimeError("live account cash/equity is invalid")
    return account


def live_position(account: dict) -> dict | None:
    positions = [item for item in account.get("positions", []) if float(item.get("quantity", 0.0)) > 0]
    return positions[0] if positions else None


def exit_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if not bool(row["above_ma120"]):
        reasons.append("跌破MA120（硬退出）")
    soft_confirmed = bool(row.get("soft_exit_confirmation", True))
    if soft_confirmed and float(row["roc20"]) < 0:
        reasons.append("ROC20转负")
    if soft_confirmed and bool(row["dual_rank_decline"]):
        reasons.append("5日与20日排名同时下滑")
    return reasons


def choose_live_target(current: str | None, candidates: pd.DataFrame, reasons: list[str]) -> str | None:
    if current and not reasons:
        return current
    available = candidates.loc[~candidates["symbol"].eq(current)]
    return str(available.iloc[0]["symbol"]) if not available.empty else None


def order_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen path-specific entry score used by the signal engine."""

    score = "selection_score" if "selection_score" in candidates.columns else "momentum_score"
    return candidates.sort_values([score, "rank"], ascending=[False, True])


def estimate_buy(symbol: str, available_cash: float, day: pd.Timestamp, market: dict) -> dict:
    frame = pd.read_csv(ROOT / "market_data" / "prices" / f"{symbol}.csv", parse_dates=["datetime"])
    row = frame.loc[frame["datetime"].eq(day)]
    if row.empty:
        raise RuntimeError(f"missing last close for {symbol} on {day.date()}")
    close = float(row.iloc[-1]["close"])
    premium = symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"
    cost = market["execution"]["fixed_premium_sensitive" if premium else "fixed_default"]
    commission_rate = float(cost["commission_rate"])
    slippage_rate = float(cost["slippage_rate"])
    minimum_commission = float(market["execution"]["minimum_commission"])
    lot = int(market["project"]["lot_size"])
    estimated_price = close * (1.0 + slippage_rate)
    quantity = math.floor(available_cash / estimated_price / lot) * lot
    while quantity > 0:
        gross = quantity * estimated_price
        commission = max(minimum_commission, gross * commission_rate)
        if gross + commission <= available_cash:
            break
        quantity -= lot
    return {
        "last_close": close,
        "estimated_execution_price": estimated_price,
        "estimated_quantity_at_last_close": quantity,
        "estimated_notional": quantity * estimated_price,
        "estimated_commission": max(minimum_commission, quantity * estimated_price * commission_rate) if quantity else 0.0,
        "quantity_rule": "下一交易日开盘按实际可用现金、实际开盘成交价、100份整数倍和最低佣金重新计算；估算数量不是预先成交。",
    }


def liquidity_instruction(symbol: str, side: str, day: pd.Timestamp, market: dict, reference_capital: float) -> dict:
    """Return the frozen capacity reference; this is not a broker fill."""
    frame = pd.read_csv(ROOT / "market_data" / "prices" / f"{symbol}.csv", parse_dates=["datetime"])
    history = frame.loc[frame["datetime"].le(day)].tail(int(market["execution"]["liquidity_lookback_days"]))
    median_amount = float(history["amount"].median()) if not history.empty else 0.0
    rate = float(market["execution"]["max_participation_rate"])
    daily_cap = median_amount * rate
    estimated_opens = math.ceil(reference_capital / daily_cap) if daily_cap > 0 else None
    return {
        "side": side,
        "symbol": symbol,
        "target_weight": 0.0 if side == "sell" else 1.0,
        "reference_capital": reference_capital,
        "amount_median_20d": median_amount,
        "max_participation_rate": rate,
        "max_notional_per_open": daily_cap,
        "estimated_opens_at_reference_capital": estimated_opens,
        "instruction": "完整目标超过单日上限时，按同一目标拆为连续开盘子订单；不得把拆单视为新信号。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the next-open ye order plan")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    review_path = ROOT / "market_data" / "sentiment" / "ai_review" / f"{args.date}.json"
    if not review_path.exists():
        raise RuntimeError("AI review is missing; fail closed and do not create a buy plan")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "complete" or review.get("coverage") != 1.0:
        raise RuntimeError("AI review is incomplete; fail closed and do not create a buy plan")
    weights = pd.read_csv(FORMAL / "signal_weights.csv", index_col=0)
    weights.index = pd.to_datetime(weights.index)
    day = pd.Timestamp(args.date)
    if day not in weights.index:
        raise RuntimeError(f"no strategy signal row for {args.date}")
    model_target = active_symbol(weights.loc[day])
    account = load_account_state(day)
    position = live_position(account)
    current = str(position["symbol"]) if position else None
    ranking = pd.read_csv(ROOT / "results" / "comparison" / "latest_ranking.csv")
    ranking = ranking.loc[ranking["date"].eq(args.date)].copy()
    if ranking.empty:
        raise RuntimeError(f"missing live ranking for {args.date}")
    candidates = order_candidates(
        ranking.loc[ranking["final_entry_pass"].astype(bool)]
    )
    held_exit_reasons: list[str] = []
    if current:
        held = ranking.loc[ranking["symbol"].eq(current)]
        if held.empty:
            raise RuntimeError(f"confirmed live holding {current} is outside the fixed ETF pool")
        held_exit_reasons = exit_reasons(held.iloc[-1])
    target = choose_live_target(current, candidates, held_exit_reasons)
    actions: list[dict] = []
    if current and current != target:
        actions.append({"side": "sell", "symbol": current, "target_weight": 0.0, "reasons": held_exit_reasons})
    if target and target != current:
        actions.append({"side": "buy", "symbol": target, "target_weight": 1.0})
    if not actions:
        actions.append({"side": "hold", "symbol": target, "target_weight": 1.0 if target else 0.0})
    config = yaml.safe_load((ROOT / "config" / "ye_strategy.yaml").read_text(encoding="utf-8"))
    market = yaml.safe_load((ROOT / "config" / "market.yaml").read_text(encoding="utf-8"))
    execution_orders = [
        liquidity_instruction(
            action["symbol"], action["side"], day, market,
            float(account["total_equity"]),
        )
        for action in actions
        if action["side"] in {"buy", "sell"} and action["symbol"]
    ]
    for order in execution_orders:
        if order["side"] == "buy":
            order["buy_estimate"] = estimate_buy(
                order["symbol"], float(account["available_cash"]), day, market
            )
        elif position and order["symbol"] == current:
            order["confirmed_quantity"] = float(position["quantity"])
    cash_management = config["cash_management"]
    payload = {
        "strategy": "ye 策略",
        "signal_date": args.date,
        "execute": "下一交易日开盘",
        "current_symbol": current,
        "target_symbol": target,
        "backtest_shadow_target_symbol": model_target,
        "account_state": {
            "source": account["source"],
            "as_of": account["as_of"],
            "confirmation_status": account["confirmation_status"],
            "first_live_day": bool(account.get("first_live_day", False)),
            "total_equity": float(account["total_equity"]),
            "available_cash": float(account["available_cash"]),
            "performance": account.get("performance", {}),
            "positions": account.get("positions", []),
            "pending_orders": account.get("pending_orders", []),
        },
        "decision_basis": {
            "live_position_source": "results/live/account_state.json；不得使用回测持仓代替",
            "eligible_candidates": [
                {
                    "symbol": str(row["symbol"]), "name": str(row["name"]),
                    "rank": int(row["rank"]), "momentum_score": float(row["momentum_score"]),
                    "selection_score": float(row.get("selection_score", row["momentum_score"])),
                    "entry_path": "常规动量" if bool(row["normal_entry"]) else "新趋势" if bool(row["emerging_entry"]) else "质量延伸",
                }
                for _, row in candidates.iterrows()
            ],
            "selection_rule": "空仓时按正式路径选择分排序：常规/质量延伸使用动量分，新趋势使用ROC20+0.05×R²20；已有仓位未触发卖出条件时不因其他候选分数更高而换仓。",
            "backtest_shadow_note": "回测影子持仓只用于绩效和一致性审计，不代表真实账户持仓，也不决定首次实盘买单。",
        },
        "actions": actions,
        "execution": {
            "target_policy": "策略只产生 0% 或 100% 的目标仓位，不进行主观分批建仓、加仓、减仓或止盈。",
            "order_sequence": "换仓时先卖旧仓，再买新仓；子订单只由流动性上限触发。",
            "broker_fill_confirmation_required": True,
            "confirmation_rule": "下一次收盘后运行前，必须以真实成交数量确认实际仓位和未完成订单；计划不得当作已成交。",
            "orders": execution_orders,
        },
        "cost": config["execution"]["cost_rule"],
        "cash_management": {
            "enabled": cash_management["enabled"],
            "product": cash_management["product"],
            "instruction": "收盘后仅对实际闲置现金执行；不改变上述ETF计划。",
            "subscribe_window": cash_management["subscribe_window"],
            "amount_rule": cash_management["amount_rule"],
            "fee_rate": cash_management["fee_rate"],
            "annualized_yield": cash_management["annualized_yield"],
            "maturity": cash_management["maturity"],
            "failure_action": cash_management["safeguards"][1],
        },
        "ai_review": {
            "reviewer": review.get("reviewer", "codex_chat"), "input_count": review["input_count"],
            "reviewed_count": review["reviewed_count"], "coverage": review["coverage"],
        },
    }
    output = ROOT / "results" / "live" / f"{args.date}_order_plan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    template = ROOT / "results" / "live" / f"{args.date}_actual_fills.template.json"
    fill_rows = [
        {
            "side": item["side"],
            "symbol": item["symbol"],
            "status": "pending",
            "quantity": None,
            "price": None,
            "broker_order_id": None,
        }
        for item in actions
        if item["side"] in {"buy", "sell"} and item.get("symbol")
    ]
    template_payload = {
        "signal_date": args.date,
        "confirmation_status": "pending",
        "source": "用户或券商成交回单确认",
        "fills": fill_rows,
        "note": "按本计划的下一交易日真实成交填写；计划和估算数量都不是成交。",
    }
    template.write_text(json.dumps(template_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
