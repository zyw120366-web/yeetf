"""Freeze research-only observations of delayed upstream field availability."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from scripts.collect_daily_sentiment import ths_hot_reason

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research/structural_audit_20260911/news_contract_probe.json"


def main():
    if OUT.exists():
        print(OUT.read_text())
        return
    observations = []
    for date in ("2026-09-10", "2026-09-11"):
        old = json.loads((ROOT / f"market_data/sentiment/{date}.json").read_text())
        frozen = old["sources"]["ths_hot_reason"]["rows"]
        fresh = ths_hot_reason(date)
        observations.append({
            "date": date,
            "endpoint": f"https://zx.10jqka.com.cn/event/api/getharden/date/{date}/orderby/date/orderway/desc/charset/GBK/",
            "frozen_rows": len(frozen), "frozen_keys": sorted(set().union(*(r.keys() for r in frozen))),
            "fresh_count": len(fresh), "fresh_keys": sorted(set().union(*(r.keys() for r in fresh))),
            "fresh_rows": fresh,
        })
    payload = {
        "observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "purpose": "research_only; never replace historical frozen reviews",
        "interpretation": "next-day field enrichment observed; exact intraday completion time is not established",
        "observations": observations,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "observations"}, ensure_ascii=False))
    for row in observations:
        print(json.dumps({k: v for k, v in row.items() if k != "fresh_rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
