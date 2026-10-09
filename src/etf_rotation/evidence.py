"""Explicit dependencies for live news decisions; missing is never negative."""
from pathlib import Path

import pandas as pd

from .sentiment_ai import validate_live_review


def review_context(root: Path, date: str, calendar, config: dict, *, validator=None) -> dict:
    validator = validator or validate_live_review
    live = config["enhanced_selection"]["sentiment_available"]
    days = pd.DatetimeIndex(calendar)
    days = days[days <= pd.Timestamp(date)]
    windows = {"current": 1,
               "emerging": int(live["emerging_trend"]["memory_days"]),
               "soft_exit": int(live["hot_exit_protection"]["memory_days"])}
    required = {key: [str(d.date()) for d in days[-size:]] for key, size in windows.items()}
    checks = {}
    for day in sorted({d for dates in required.values() for d in dates}):
        try:
            review = validator(root, day)
            checks[day] = {"status": "complete", "review": {k: review.get(k) for k in
                ("status", "input_count", "reviewed_count", "coverage", "snapshot_hash")}}
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
            missing = not (root / f"market_data/sentiment/{day}.json").is_file()
            checks[day] = {"status": "missing" if missing else "invalid",
                           "message": f"{day}资讯快照缺失" if missing else f"{day}资讯审核未通过",
                           "diagnostic": f"{type(exc).__name__}: {exc}"}
    complete = {key: len(dates) == windows[key] and all(checks[d]["status"] == "complete" for d in dates)
                for key, dates in required.items()}
    issues = [v["message"] for v in checks.values() if v["status"] != "complete"]
    if any(len(required[k]) != n for k, n in windows.items()):
        issues.append("资讯核验所需交易日历不足")
    return {"status": "complete" if all(complete.values()) else "partial",
            "required_dates": required, "checks": checks, "complete": complete,
            "issues": issues,
            "files": [f"market_data/sentiment/{prefix}{d}.json" for d, check in checks.items()
                      if check["status"] == "complete" for prefix in ("", "ai_review/")]}
