from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the immutable ye run manifest")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    critical = [
        ROOT / "config" / "market.yaml",
        ROOT / "config" / "ye_strategy.yaml",
        ROOT / "config" / "etfwin_official.yaml",
        ROOT / "config" / "sentiment.yaml",
        ROOT / "config" / "strategy_governance.yaml",
        ROOT / "config" / "research_hypotheses.yaml",
        ROOT / "market_data" / "sentiment" / "ai_review" / f"{args.date}.json",
        ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv",
        ROOT / "results" / "ye_strategy" / "summary.json",
        ROOT / "results" / "ye_strategy" / "signal_weights.csv",
        ROOT / "results" / "ye_strategy" / "trade_audit.json",
        ROOT / "results" / "etfwin_reference" / "reference_summary.json",
        ROOT / "results" / "comparison" / "latest_signals.json",
        ROOT / "results" / "comparison" / "metrics.csv",
        ROOT / "results" / "live" / "account_state.json",
        ROOT / "results" / "live" / f"{args.date}_order_plan.json",
        ROOT / "results" / "live" / "readiness_report.json",
        ROOT / "dashboard" / "public" / "ye-strategy.html",
        ROOT / "dashboard" / "public" / "ye-daily.html",
        ROOT / "dashboard" / "public" / "ye-backtest.html",
    ]
    missing = [path for path in critical if not path.exists()]
    if missing:
        raise FileNotFoundError("missing manifest inputs: " + ", ".join(map(str, missing)))
    price_files = sorted((ROOT / "market_data" / "prices").glob("*.csv"))
    source_files = sorted((ROOT / "src" / "etf_rotation").glob("*.py")) + [
        ROOT / "run_strategies.py",
        ROOT / "scripts" / "build_sentiment_features.py",
        ROOT / "scripts" / "build_trade_audit.py",
        ROOT / "scripts" / "build_live_order_plan.py",
        ROOT / "scripts" / "validate_live_readiness.py",
        ROOT / "scripts" / "build_daily_reference_report.py",
        ROOT / "scripts" / "build_live_run_card.py",
        ROOT / "scripts" / "reconcile_actual_fills.py",
        ROOT / "scripts" / "run_after_close.py",
        ROOT / "dashboard" / "scripts" / "build_ye_strategy_html.py",
    ]
    payload = {
        "strategy": "ye 策略",
        "signal_date": args.date,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "critical_files": [file_record(path) for path in critical],
        "source_files": [file_record(path) for path in source_files],
        "price_files": [file_record(path) for path in price_files],
        "counts": {
            "critical_files": len(critical),
            "source_files": len(source_files),
            "price_files": len(price_files),
        },
    }
    target_dir = ROOT / "results" / "audit"
    target_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    dated = target_dir / f"{args.date}_run_manifest.json"
    latest = target_dir / "latest_run_manifest.json"
    dated.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    print(json.dumps({"manifest": str(dated), **payload["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
