import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from etf_rotation.live import atomic_json
from scripts.build_live_order_plan import opportunity_switch_status

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fault,expected", [(None, 4), ("candidate", 1), ("gap", 1),
                                          ("partial", 1), ("position", 1), ("false", 1)])
def test_replay_blocked_reports_ignores_broken_cached_streak(tmp_path, fault, expected):
    dates = pd.to_datetime(["2026-10-01", "2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27"])
    prices = tmp_path / "market_data/prices"
    prices.mkdir(parents=True)
    pd.DataFrame({"datetime": dates}).to_csv(prices / "510300.SH.csv", index=False)
    account = {"positions": [{"symbol": "old", "opened_on": "2026-10-01", "quantity": 100}]}
    for day in dates[1:-1]:
        report = {"signal_date": str(day.date()), "status": "BLOCKED", "account": account,
                  "analysis": {"status": "complete", "opportunity_switch": {
                      "candidate_symbol": "new", "qualifies_today": True, "confirmation_streak": 1}}}
        report = json.loads(json.dumps(report))
        if day == dates[-2]:
            if fault == "gap":
                continue
            if fault == "candidate":
                report["analysis"]["opportunity_switch"]["candidate_symbol"] = "other"
            if fault == "partial":
                report["analysis"]["status"] = "partial"
            if fault == "position":
                report["account"]["positions"][0]["quantity"] = 200
            if fault == "false":
                report["analysis"]["opportunity_switch"]["qualifies_today"] = False
        atomic_json(tmp_path / f"results/live/{day.date()}_daily_report.json", report)
    config = yaml.safe_load((ROOT / "config/ye_strategy.yaml").read_text())
    config["enhanced_selection"]["opportunity_switch"]["minimum_hold_days"] = 1
    result = opportunity_switch_status(dates[-1], "old", pd.Series({"rank": 8, "momentum_score": 0}),
        pd.DataFrame([{"symbol": "new", "rank": 1, "momentum_score": .1, "pool_role": "core"}]),
        account, config, root=tmp_path)
    assert result["confirmation_streak"] == expected
    assert result["triggered"] == (expected >= 2)
