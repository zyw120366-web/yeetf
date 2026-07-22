"""Create an inspectable ye live-run card after the daily plan is frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ye daily live-run card")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    governance = yaml.safe_load((ROOT / "config" / "strategy_governance.yaml").read_text(encoding="utf-8"))
    live = ROOT / "results" / "live"
    audit = ROOT / "results" / "audit"
    plan_path = live / f"{args.date}_order_plan.json"
    review_path = ROOT / "market_data" / "sentiment" / "ai_review" / f"{args.date}.json"
    readiness_path = live / "readiness_report.json"
    manifest_path = audit / f"{args.date}_run_manifest.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actions = [
        {
            "side": item["side"], "symbol": item.get("symbol"),
            "target_weight": item["target_weight"], "reasons": item.get("reasons", []),
        }
        for item in plan["actions"]
    ]
    template_path = live / f"{args.date}_actual_fills.template.json"
    reconciliation_path = audit / f"{args.date}_execution_reconciliation.json"
    card = {
        "card_type": "ye_live_run_card",
        "strategy": governance["formal_strategy"],
        "signal_date": args.date,
        "execute": plan["execute"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "current_symbol": plan["current_symbol"],
            "target_symbol": plan["target_symbol"],
            "backtest_shadow_target_symbol": plan.get("backtest_shadow_target_symbol"),
            "actions": actions,
            "cost": plan["cost"],
        },
        "account_state": plan.get("account_state", {}),
        "decision_basis": plan.get("decision_basis", {}),
        "cash_management": plan.get("cash_management", {"enabled": False}),
        "sentiment_review": {
            "status": review.get("status"),
            "coverage": review.get("coverage"),
            "reviewed_count": review.get("reviewed_count"),
            "input_count": review.get("input_count"),
            "prompt_version": review.get("prompt_version"),
            "review_metadata": review.get("review_metadata"),
            "review_protocol": review.get("review_protocol"),
            "protocol_status": (
                "fingerprinted" if review.get("review_protocol")
                else "legacy_pre_policy"
            ),
        },
        "release": {
            "readiness": readiness["status"],
            "blocking_items": readiness.get("blocking_items", []),
            "plan_is_not_fill": True,
        },
        "audit": {
            "run_manifest": manifest_path.relative_to(ROOT).as_posix(),
            "run_manifest_sha256": sha256(manifest_path),
            "critical_file_count": manifest["counts"]["critical_files"],
            "source_file_count": manifest["counts"]["source_files"],
            "price_file_count": manifest["counts"]["price_files"],
        },
        "execution_reconciliation": {
            "status": "not_required" if not any(item["side"] in {"buy", "sell"} for item in plan["actions"]) else "pending_next_open",
            "actual_fill_template": template_path.relative_to(ROOT).as_posix(),
            "reconciliation_output": reconciliation_path.relative_to(ROOT).as_posix(),
        },
    }
    audit.mkdir(parents=True, exist_ok=True)
    dated = audit / f"{args.date}_live_run_card.json"
    latest = audit / "latest_live_run_card.json"
    text = json.dumps(card, ensure_ascii=False, indent=2)
    dated.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    print(json.dumps({"run_card": str(dated), "readiness": readiness["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
