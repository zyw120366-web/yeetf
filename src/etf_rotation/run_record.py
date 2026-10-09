"""Small daily receipt: bind decisions, inventory inputs, use Git for code history.

The manifest is not a byte-for-byte archive of the whole repository. News
authenticity is checked at decision time; prices remain append-only. Neither
routine delivery nor next-day execution needs to rehash years of market data.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from .live import atomic_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delivery_files(date: str) -> list[str]:
    return [f"results/live/{date}_{suffix}" for suffix in
            ("order_plan.json", "daily_report.json", "daily_report.md")] + [
                "outputs/ETF轮动策略_今日日报.html", "dashboard/public/ye-daily.html"]


def write_run_record(root: Path, date: str) -> dict:
    def read(relative):
        return json.loads((root / relative).read_text(encoding="utf-8"))

    plan = read(f"results/live/{date}_order_plan.json")
    report = read(f"results/live/{date}_daily_report.json")
    ready = read("results/live/readiness_report.json")
    if any(x.get("signal_date") != date for x in (plan, report, ready)):
        raise ValueError("运行记录输入日期不一致")
    if report["status"] != ready["status"] or ready["status"] not in {"READY", "SELL_ONLY", "BLOCKED"}:
        raise ValueError("运行记录必须对应同一终态")
    analysis = report["analysis"]
    dependencies = list(dict.fromkeys(report["valuation"].get("source_files", [])
                                     + analysis.get("news_dependencies", {}).get("files", [])
                                     + [f"market_data/sentiment/{prefix}{d}.json"
                                        for d in plan.get("review_dates_required", []) for prefix in ("", "ai_review/")]))
    for relative in dependencies:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"日报所用证据不存在：{relative}")
    prices = [str(p.relative_to(root)) for p in sorted((root / "market_data/prices").glob("*.csv"))]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        changed = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "scripts", "config", "dashboard/scripts", "run_strategies.py"],
            cwd=root, stderr=subprocess.DEVNULL, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision, changed = None, None
    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 2, "signal_date": date, "status": ready["status"], "created_at_utc": created,
        "code_revision": revision, "uncommitted_code": changed,
        "scope": "只绑定当日计划与日报；行情/资讯记录路径，不重复全库哈希。历史代码及输入版本追溯Git。",
        "critical_files": [{"path": p, "sha256": sha256(root / p)} for p in delivery_files(date)],
        "evidence_files": [{"path": p} for p in dependencies],
        "price_files": [{"path": p} for p in prices],
        "counts": {"critical_files": len(delivery_files(date)), "source_files": 0, "price_files": len(prices)},
    }
    manifest_path = root / f"results/audit/{date}_run_manifest.json"
    atomic_json(manifest_path, manifest)
    governance_path = root / "config/strategy_governance.yaml"
    governance = yaml.safe_load(governance_path.read_text()) if governance_path.is_file() else {}
    requires_fill = any(a["side"] in {"buy", "sell"} for a in plan["actions"])
    card = {
        "card_type": "ye_live_run_card", "schema_version": 2, "signal_date": date,
        "strategy": governance.get("formal_strategy"), "created_at_utc": created,
        "execute": plan.get("execute"), "account_state": plan.get("account_state", {}),
        "decision": {k: plan.get(k) for k in ("current_symbol", "target_symbol", "backtest_shadow_target_symbol", "actions", "cost")},
        "decision_basis": plan.get("decision_basis", {}),
        "sentiment_review": report["review"],
        "observation": {k: analysis.get(k) for k in ("status", "prices_status")},
        "daily_report": f"results/live/{date}_daily_report.json",
        "release": {"readiness": ready["status"], "blocking_items": ready.get("blocking_items", []), "plan_is_not_fill": True},
        "audit": {"run_manifest": str(manifest_path.relative_to(root)), "run_manifest_sha256": sha256(manifest_path),
                  "critical_file_count": len(delivery_files(date)), "source_file_count": 0, "price_file_count": len(prices)},
        "execution_reconciliation": {
            "status": "pending_next_open" if requires_fill else "not_required",
            "actual_fill_template": f"results/live/{date}_actual_fills.template.json" if requires_fill else None,
            "reconciliation_output": f"results/audit/{date}_execution_reconciliation.json"},
    }
    atomic_json(root / f"results/audit/{date}_live_run_card.json", card)
    return card
