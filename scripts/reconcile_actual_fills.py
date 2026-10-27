"""Reconcile a user-confirmed broker-fill file with one ye order plan."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from etf_rotation.live import atomic_json, fingerprint, validate_account, validate_fills, live_lock, raw_quote


ROOT = Path(__file__).resolve().parents[1]
RECONCILED_STATUSES = {"confirmed", "not_required", "baseline_confirmed"}


def planned_trades(plan: dict) -> list[tuple[str, str]]:
    return [
        (str(item["side"]), str(item["symbol"]))
        for item in plan["actions"]
        if item["side"] in {"buy", "sell"} and item.get("symbol")
    ]


def accept_confirmed_account_baseline(
    report: dict,
    account: dict,
    baseline_date: str,
    account_source: str,
) -> dict:
    """Accept a user-confirmed account after a wholly unfilled/cancelled plan.

    This deliberately does not turn cancelled orders into fills.  It preserves
    the execution exception and records that the confirmed live account is the
    new starting point for subsequent plans.
    """
    records = report.get("actual_fills", [])
    validate_account(account, baseline_date)
    if baseline_date <= str(report.get("signal_date", "")):
        raise ValueError("账户基线日必须晚于订单信号日")
    zero_fill_exception = (
        report.get("status") == "exception"
        and report.get("confirmation_status") == "complete"
        and report.get("matched_plan") is True
        and report.get("all_filled") is False
        and bool(records)
        and all(str(item.get("status")) in {"cancelled", "unfilled"} for item in records)
        and all(float(item.get("quantity", 0.0)) == 0.0 for item in records)
    )
    positive_positions = [
        item for item in account.get("positions", [])
        if float(item.get("quantity", 0.0)) > 0.0
    ]
    account_is_confirmed = (
        account.get("confirmation_status") == "confirmed"
        and str(account.get("as_of", "")).startswith(baseline_date)
        and not account.get("pending_orders")
        and len(positive_positions) <= 1
        and float(account.get("total_equity", 0.0)) > 0.0
        and float(account.get("available_cash", -1.0)) >= 0.0
    )
    if not zero_fill_exception:
        raise ValueError("只有全部取消或未成交、且数量均为0的已确认异常计划才能重置账户基线。")
    if not account_is_confirmed:
        raise ValueError("账户未被用户明确确认、日期不符、存在待处理订单或账户字段不完整。")

    accepted = dict(report)
    accepted["status"] = "baseline_confirmed"
    accepted["baseline_acceptance"] = {
        "date": baseline_date,
        "status": "confirmed",
        "account_source": account_source,
        "account_confirmation_source": account.get("source", "unknown"),
        "account_as_of": account.get("as_of"),
        "account_snapshot": account,
        "account_sha256": fingerprint(account),
        "account_symbol": (
            str(positive_positions[0].get("symbol")) if positive_positions else None
        ),
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "原计划未成交记录完整保留；用户确认的真实账户被接受为后续执行基线。",
    }
    return accepted


def reconcile() -> None:
    parser = argparse.ArgumentParser(description="Reconcile confirmed broker fills with a ye plan")
    parser.add_argument("--date", required=True, help="signal date of the order plan")
    parser.add_argument("--fills", type=Path, help="defaults to results/live/YYYY-MM-DD_actual_fills.json")
    parser.add_argument(
        "--accept-account-baseline-date",
        help="explicitly accept the user-confirmed account state on this date after a wholly cancelled/unfilled plan",
    )
    args = parser.parse_args()
    plan_path = ROOT / "results" / "live" / f"{args.date}_order_plan.json"
    fills_path = args.fills or ROOT / "results" / "live" / f"{args.date}_actual_fills.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = planned_trades(plan)
    if not expected:
        report = {
            "signal_date": args.date,
            "status": "not_required",
            "reason": "计划仅为持有或空仓，没有待核对的买卖订单。",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    elif not fills_path.exists():
        report = {
            "signal_date": args.date,
            "status": "pending_confirmation",
            "reason": "未找到用户或券商确认的实际成交文件。",
            "expected_orders": [{"side": side, "symbol": symbol} for side, symbol in expected],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    else:
        fills = json.loads(fills_path.read_text(encoding="utf-8"))
        validate_fills(fills, plan, args.date)
        records = fills.get("fills", [])
        actual = [(str(item.get("side")), str(item.get("symbol"))) for item in records]
        statuses = {str(item.get("status")) for item in records}
        confirmation = fills.get("confirmation_status") == "complete"
        matched = sorted(actual) == sorted(expected)
        filled = statuses == {"filled"}
        report = {
            "signal_date": args.date,
            "status": "confirmed" if confirmation and matched and filled else "exception",
            "expected_orders": [{"side": side, "symbol": symbol} for side, symbol in expected],
            "actual_fills": records,
            "broker_available_after_sale": fills.get("broker_available_after_sale"),
            "confirmation_status": fills.get("confirmation_status"),
            "matched_plan": matched,
            "all_filled": filled,
            "source": fills.get("source", "unknown"),
            "note": fills.get("note", ""),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    if plan.get("execution", {}).get("funding_scope") == "sale_proceeds_only" and report["status"] == "confirmed":
        account_path = ROOT / "results/live/account_state.json"
        account = json.loads(account_path.read_text())
        existing_path = ROOT / f"results/audit/{args.date}_execution_reconciliation.json"
        existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
        if (existing.get("actual_fills") == records and existing.get("account_after") == account):
            report = existing
        else:
            if fingerprint(account) != fingerprint(plan["account_state"]):
                raise ValueError("账户已改变，不能重复或覆盖真实换仓成交")
            execution_date = fills.get("execution_date")
            if not execution_date or execution_date <= args.date:
                raise ValueError("换仓成交日期必须晚于信号日")
            buy = records[-1]
            _, close = raw_quote(ROOT, buy["symbol"], execution_date)
            report["account_before_sha256"] = fingerprint(account)
            old = account["positions"][0]["symbol"]
            account["positions"] = [{"symbol": buy["symbol"], "opened_on": execution_date,
                "quantity": buy["quantity"], "average_cost": buy["price"], "market_price": close,
                "market_value": buy["quantity"] * close}]
            account["as_of"] = execution_date + "_close"
            account["actual_execution"] = {"signal_date": args.date, "execution_date": execution_date, "fills": records}
            account.setdefault("last_exit_dates", {})[old] = execution_date
            account["available_cash"] = account["total_equity"] = None
            account["source"] = "用户或券商真实换仓成交，历史资金仍待核"
            report["account_after"] = account
            atomic_json(existing_path, report)
            atomic_json(account_path, account)
    if args.accept_account_baseline_date:
        account_path = ROOT / "results" / "live" / "account_state.json"
        if not account_path.exists():
            raise SystemExit("无法接受账户基线：results/live/account_state.json 不存在。")
        account = json.loads(account_path.read_text(encoding="utf-8"))
        try:
            report = accept_confirmed_account_baseline(
                report,
                account,
                args.accept_account_baseline_date,
                account_path.relative_to(ROOT).as_posix(),
            )
        except ValueError as exc:
            raise SystemExit(f"无法接受账户基线：{exc}") from exc
    output_dir = ROOT / "results" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.date}_execution_reconciliation.json"
    if output.exists() and not args.accept_account_baseline_date:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "baseline_confirmed" and existing.get("actual_fills") == report.get("actual_fills"):
            report = existing
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] in RECONCILED_STATUSES else 2)


def main() -> None:
    with live_lock(ROOT):
        reconcile()


if __name__ == "__main__":
    main()
