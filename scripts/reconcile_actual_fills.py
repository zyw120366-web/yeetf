"""Reconcile a user-confirmed broker-fill file with one ye order plan."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def planned_trades(plan: dict) -> list[tuple[str, str]]:
    return [
        (str(item["side"]), str(item["symbol"]))
        for item in plan["actions"]
        if item["side"] in {"buy", "sell"} and item.get("symbol")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile confirmed broker fills with a ye plan")
    parser.add_argument("--date", required=True, help="signal date of the order plan")
    parser.add_argument("--fills", type=Path, help="defaults to results/live/YYYY-MM-DD_actual_fills.json")
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
            "confirmation_status": fills.get("confirmation_status"),
            "matched_plan": matched,
            "all_filled": filled,
            "source": fills.get("source", "unknown"),
            "note": fills.get("note", ""),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    output_dir = ROOT / "results" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.date}_execution_reconciliation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] in {"confirmed", "not_required"} else 2)


if __name__ == "__main__":
    main()
