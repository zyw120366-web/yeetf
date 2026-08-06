"""Advance the live account under the user's standing execution authorization."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = ROOT / "results" / "live" / "account_state.json"


def trading_dates() -> pd.DatetimeIndex:
    frame = pd.read_csv(ROOT / "market_data" / "prices" / "510300.SH.csv", parse_dates=["datetime"])
    return pd.DatetimeIndex(frame["datetime"])


def quote(symbol: str, day: pd.Timestamp) -> tuple[float, float]:
    frame = pd.read_csv(ROOT / "market_data" / "prices" / f"{symbol}.csv", parse_dates=["datetime"])
    row = frame.loc[frame["datetime"].eq(day)]
    if row.empty:
        raise RuntimeError(f"missing {symbol} quote for {day.date()}")
    return float(row.iloc[-1]["open"]), float(row.iloc[-1]["close"])


def costs(symbol: str, market: dict) -> tuple[float, float, float]:
    premium = symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"
    item = market["execution"]["fixed_premium_sensitive" if premium else "fixed_default"]
    return float(item["commission_rate"]), float(item["slippage_rate"]), float(market["execution"]["minimum_commission"])


def instrument_name(symbol: str, market: dict) -> str:
    for item in market["universe"]:
        candidate = f"{item['code']}.{item['market']}"
        if candidate == symbol:
            return str(item["name"])
    return symbol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    day = pd.Timestamp(args.date)
    governance = yaml.safe_load((ROOT / "config" / "strategy_governance.yaml").read_text(encoding="utf-8"))
    authorization = governance["live_audit"].get("standing_execution_authorization", {})
    if not authorization.get("enabled"):
        raise RuntimeError("standing execution authorization is disabled")
    account = json.loads(ACCOUNT.read_text(encoding="utf-8"))
    account_day = pd.Timestamp(str(account["as_of"]).split("_")[0])
    if account_day == day:
        print(json.dumps({"status": "already_current", "date": args.date}, ensure_ascii=False))
        return
    dates = trading_dates()
    prior_dates = dates[dates < day]
    if day not in dates or not len(prior_dates) or account_day != prior_dates[-1]:
        raise RuntimeError("account may only advance one confirmed trading session")
    prior = str(prior_dates[-1].date())
    plan_path = ROOT / "results" / "live" / f"{prior}_order_plan.json"
    if not plan_path.exists():
        raise RuntimeError(f"missing prior order plan: {plan_path.name}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if account.get("pending_orders"):
        raise RuntimeError("pending broker orders require explicit correction")
    market = yaml.safe_load((ROOT / "config" / "market.yaml").read_text(encoding="utf-8"))
    lot = int(market["project"]["lot_size"])
    positions = [p for p in account.get("positions", []) if float(p.get("quantity", 0)) > 0]
    if len(positions) > 1:
        raise RuntimeError("only one live ETF position is supported")
    position = positions[0] if positions else None
    cash = float(account["available_cash"])
    fills: list[dict] = []
    for action in plan["actions"]:
        side, symbol = action["side"], action.get("symbol")
        if side == "hold":
            continue
        if side == "sell":
            if not position or position["symbol"] != symbol:
                raise RuntimeError("planned sell does not match the live account")
            opening, _ = quote(symbol, day)
            rate, slip, minimum = costs(symbol, market)
            price = opening * (1 - slip)
            quantity = float(position["quantity"])
            gross = quantity * price
            fee = max(minimum, gross * rate)
            cash += gross - fee
            fills.append({"side": "sell", "symbol": symbol, "status": "filled", "quantity": quantity, "price": price})
            position = None
        if side == "buy":
            opening, _ = quote(symbol, day)
            rate, slip, minimum = costs(symbol, market)
            price = opening * (1 + slip)
            quantity = math.floor(cash / price / lot) * lot
            while quantity and quantity * price + max(minimum, quantity * price * rate) > cash:
                quantity -= lot
            gross = quantity * price
            fee = max(minimum, gross * rate) if quantity else 0.0
            cash -= gross + fee
            position = {"symbol": symbol, "name": instrument_name(symbol, market), "opened_on": str(day.date()), "quantity": quantity, "average_cost": price}
            fills.append({"side": "buy", "symbol": symbol, "status": "filled", "quantity": quantity, "price": price})
    output_positions: list[dict] = []
    if position and float(position["quantity"]) > 0:
        _, close = quote(str(position["symbol"]), day)
        position["market_price"] = close
        position["market_value"] = float(position["quantity"]) * close
        position["unrealized_pnl"] = float(position["quantity"]) * (close - float(position["average_cost"]))
        output_positions = [position]
    account.update({
        "as_of": f"{args.date}_close",
        "confirmation_status": "assumed_authorized",
        "source": "用户2026-08-06常设授权：未另行报告时，按上一交易日开盘计划完整执行并按当日实际开盘价记账",
        "available_cash": round(cash, 6),
        "total_equity": round(cash + sum(float(p["market_value"]) for p in output_positions), 6),
        "positions": output_positions,
        "pending_orders": [],
        "execution_assumption": {"enabled": True, "authorization_date": "2026-08-06", "prior_signal_date": prior, "assumed_fills": fills},
    })
    account["note"] = "用户常设授权下的计划执行记账；若券商实际成交、出入金或未完成订单与此不同，必须立即以真实记录更正。"
    ACCOUNT.write_text(json.dumps(account, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reconciliation = {"signal_date": prior, "status": "assumed_authorized", "source": account["source"], "actual_fills": fills}
    audit = ROOT / "results" / "audit" / f"{prior}_execution_reconciliation.json"
    audit.write_text(json.dumps(reconciliation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "advanced", "date": args.date, "fills": fills, "equity": account["total_equity"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
