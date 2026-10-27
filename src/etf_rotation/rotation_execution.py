"""Formal rotation funded by the old position only; unknown legacy cash stays unknown."""
import json
from pathlib import Path

import yaml

from .funding import affordable_quantity
from .live import atomic_json, raw_quote, validate_sell_position


def validate_proceeds_plan(account, plan, date):
    validate_sell_position(account, date)
    execution = plan.get("execution", {})
    orders = execution.get("orders", [])
    p = account["positions"][0]
    if (plan.get("signal_date") != date or execution.get("funding_scope") != "sale_proceeds_only"
            or execution.get("broker_fill_confirmation_required") is not True
            or len(orders) != 2 or [o.get("side") for o in orders] != ["sell", "buy"]
            or orders[0].get("symbol") != p["symbol"]
            or orders[0].get("confirmed_quantity") != p["quantity"]
            or orders[1].get("symbol") != plan.get("target_symbol")
            or plan.get("target_symbol") in {None, p["symbol"]}
            or orders[1].get("requires_full_sale") is not True
            or [(a.get("side"), a.get("symbol")) for a in plan.get("actions", [])]
               != [(o["side"], o["symbol"]) for o in orders]
            or orders[1].get("buy_estimate", {}).get("estimated_quantity_at_last_close", 0) <= 0):
        raise ValueError("卖出款换仓计划、数量或先卖后买约束无效")


def publish_proceeds_rotation(root: Path, date: str) -> bool:
    from scripts.validate_live_readiness import previous_execution_is_reconciled
    from .daily_report import analyze, publish
    from .run_record import write_run_record
    account = json.loads((root / "results/live/account_state.json").read_text())
    policy = yaml.safe_load((root / "config/strategy_governance.yaml").read_text())
    if not policy["live_audit"].get("sale_proceeds_rotation", {}).get("enabled"):
        return False
    validate_sell_position(account, date)
    if not previous_execution_is_reconciled(date, root):
        raise ValueError("上一订单未对账，禁止重复换仓")
    analysis = analyze(root, date, account)
    if (analysis.get("status") != "complete" or not analysis.get("holding_trusted")
            or not analysis.get("exit_reasons")):
        return False
    p = account["positions"][0]
    target = analysis.get("technical_target")
    candidates = {r["symbol"] for r in analysis["ranking"] if r.get("final_entry_pass") is True}
    if not target or target == p["symbol"] or target not in candidates:
        return False
    market = yaml.safe_load((root / "config/market.yaml").read_text())
    def costs(symbol):
        premium = symbol.startswith("513") or symbol == "159941.SZ"
        return market["execution"]["fixed_premium_sensitive" if premium else "fixed_default"]
    _, old_close = raw_quote(root, p["symbol"], date)
    _, new_close = raw_quote(root, target, date)
    minimum = float(market["execution"]["minimum_commission"])
    lot = int(market["project"]["lot_size"])
    sell_cost, buy_cost = costs(p["symbol"]), costs(target)
    sell_price = old_close * (1 - sell_cost["slippage_rate"])
    sell_gross = p["quantity"] * sell_price
    sell_fee = max(minimum, sell_gross * sell_cost["commission_rate"])
    net = sell_gross - sell_fee
    buy_price = new_close * (1 + buy_cost["slippage_rate"])
    quantity = affordable_quantity(net, buy_price, buy_cost["commission_rate"], minimum, lot)
    if quantity <= 0:
        return False
    # Same formal capacity rule, based on this sale's capital, not unknown equity.
    for symbol, notional in ((p["symbol"], sell_gross), (target, quantity * buy_price)):
        import pandas as pd
        bars = pd.read_csv(root / f"market_data/prices/{symbol}.csv")
        median = float(bars.loc[bars.datetime.le(date)].tail(market["execution"]["liquidity_lookback_days"]).amount.median())
        if notional > median * market["execution"]["max_participation_rate"]:
            return False  # Sell-only fallback; never bypass capacity to fund a buy.
    sell = {"side": "sell", "symbol": p["symbol"], "target_weight": 0.,
            "confirmed_quantity": p["quantity"], "reasons": analysis["exit_reasons"],
            "instruction": "先卖出全部已确认持仓；未全部成交或仍有未完成卖单，禁止买入新ETF。"}
    buy = {"side": "buy", "symbol": target, "target_weight": 1., "requires_full_sale": True,
           "cost_parameters": {"commission_rate": buy_cost["commission_rate"],
                               "minimum_commission": minimum, "lot_size": lot},
           "instruction": "卖单全部成交后，仅使用本次卖出净款且不超过券商实际可用资金；按实际买价重新计算100股整数倍，不能机械照抄参考数量。",
           "buy_estimate": {"last_close": new_close, "estimated_execution_price": buy_price,
                            "estimated_quantity_at_last_close": quantity, "estimated_sale_net": net,
                            "estimated_sell_fee": sell_fee,
                            "quantity_rule": "开盘数量由实际卖出成交价、费用及银行买价决定；价格变化会增减100股，旧现金和历史入金不纳入预算。"}}
    plan = {"strategy": "ye 策略", "signal_date": date, "execute": "下一交易日开盘先卖后买",
            "status": "READY", "current_symbol": p["symbol"], "target_symbol": target,
            "account_state": account, "actions": [sell, buy],
            "decision_basis": {"opportunity_switch": analysis["opportunity_switch"],
                               "live_position_source": "用户已确认成交；不使用回测影子持仓"},
            "execution": {"orders": [sell, buy], "funding_scope": "sale_proceeds_only",
                          "release_required": True, "broker_fill_confirmation_required": True,
                          "order_sequence": "旧仓全部成交→确认净款及券商可用→按实际买价核量→买入"}}
    validate_proceeds_plan(account, plan, date)
    ready = {"signal_date": date, "status": "READY", "buy_allowed": True, "sell_allowed": True,
             "funding_scope": "sale_proceeds_only", "blocking_items": [],
             "note": "仅本次卖出净款的条件换仓已放行；历史现金、本金及账户总收益仍待核，不代表资金全账已核清。"}
    atomic_json(root / f"results/live/{date}_order_plan.json", plan)
    atomic_json(root / "results/live/readiness_report.json", ready)
    atomic_json(root / f"results/live/{date}_actual_fills.template.json", {
        "signal_date": date, "execution_date": None, "broker_available_after_sale": None,
        "confirmation_status": "pending", "source": "券商实际成交",
        "fills": [{"side": o["side"], "symbol": o["symbol"], "status": "unfilled",
                   "quantity": 0, "price": 0, "fee": None} for o in [sell, buy]]})
    publish(root, date)
    write_run_record(root, date)
    return True
