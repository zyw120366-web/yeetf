from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from etf_rotation.live import atomic_json, publish_blocked, live_lock
from etf_rotation.sentiment_ai import validate_live_review


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
sys.path.insert(0, str(ROOT))


def run(*args: str) -> None:
    subprocess.run([PYTHON, *args], cwd=ROOT, check=True)


def execute(args) -> None:
    atomic_json(ROOT / "results/live/readiness_report.json", {
        "signal_date": args.date, "status": "RUNNING", "blocking_items": ["本次运行尚未完成"]})
    run_pipeline(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen ye strategy after market close")
    parser.add_argument("--date", required=True, help="signal date, YYYY-MM-DD")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--fetch-prices", action="store_true", help="refresh daily bars before calculating")
    args = parser.parse_args()
    with live_lock(ROOT):
        try:
            execute(args)
        except Exception as exc:
            publish_blocked(ROOT, args.date, f"{type(exc).__name__}: {exc}")
            from etf_rotation.risk_release import publish_sell_only
            try:
                if publish_sell_only(ROOT, args.date, str(exc)):
                    print("SELL_ONLY：仅风险卖出放行，禁止新买入")
                    return
            except Exception as risk_exc:
                publish_blocked(ROOT, args.date, f"完整运行失败：{exc}；独立卖出发布失败：{risk_exc}")
            raise SystemExit(2) from exc


def run_pipeline(args) -> None:
    if args.fetch_prices:
        run("scripts/fetch_prices.py", "--force")
    market_path = ROOT / "config" / "market.yaml"
    text = market_path.read_text(encoding="utf-8")
    calendar_path = next(
        (
            path for path in (
                ROOT / "market_data" / "prices" / "000001.SH.csv",
                ROOT / "market_data" / "prices" / "510300.SH.csv",
            )
            if path.exists()
        ),
        None,
    )
    if calendar_path is None:
        raise RuntimeError("no benchmark trading-calendar price file is available")
    benchmark = pd.read_csv(calendar_path, parse_dates=["datetime"])
    if pd.Timestamp(args.date) not in set(benchmark["datetime"]):
        raise RuntimeError(f"{args.date} is not present in the benchmark trading calendar")
    if not (ROOT / "market_data/live_quotes" / f"{args.date}.json").exists():
        run("scripts/fetch_live_quotes.py", "--date", args.date)
    from scripts.advance_authorized_live_account import advance
    advance(args.date)
    updated, count = re.subn(r"(?m)^(\s*data_end:\s*)['\"]?\d{4}-\d{2}-\d{2}['\"]?\s*$", rf"\g<1>'{args.date}'", text, count=1)
    if count != 1:
        raise RuntimeError("could not update config/market.yaml project.data_end")
    market_path.write_text(updated, encoding="utf-8")
    if not args.skip_collect:
        run("scripts/collect_daily_sentiment.py", "--date", args.date)
    review_path = ROOT / "market_data" / "sentiment" / "ai_review" / f"{args.date}.json"
    if not review_path.exists():
        raise RuntimeError(
            "Codex chat review is missing. Export the review queue, review every row in this conversation, "
            "then run commit_manual_sentiment_review.py before finalizing the order plan."
        )
    validate_live_review(ROOT, args.date)
    run("scripts/build_sentiment_features.py")
    run("run_strategies.py")
    run("scripts/build_trade_audit.py")
    run("scripts/build_live_order_plan.py", "--date", args.date)
    run("scripts/validate_live_readiness.py", "--date", args.date)
    run("scripts/build_daily_reference_report.py", "--date", args.date)
    run("dashboard/scripts/build_ye_strategy_html.py")
    run("scripts/build_run_manifest.py", "--date", args.date)
    run("scripts/build_live_run_card.py", "--date", args.date)


if __name__ == "__main__":
    main()
