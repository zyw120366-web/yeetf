from __future__ import annotations

import pandas as pd
import hashlib
import pytest

from etf_rotation.data import audited_price_cutoff, merge_frozen_history
from etf_rotation.live import atomic_json


def test_merge_frozen_history_preserves_old_bars_and_scales_new_session() -> None:
    cached = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-09-07", "2026-09-08"]),
            "open": [9.0, 10.0],
            "high": [9.5, 10.5],
            "low": [8.5, 9.5],
            "close": [9.0, 10.0],
            "vol": [100.0, 110.0],
            "amount": [900.0, 1100.0],
        }
    )
    downloaded = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-09-07", "2026-09-08", "2026-09-09"]),
            "open": [4.5, 5.0, 5.5],
            "high": [4.75, 5.25, 5.75],
            "low": [4.25, 4.75, 5.25],
            "close": [4.5, 5.0, 5.5],
            "vol": [100.0, 110.0, 120.0],
            "amount": [900.0, 1100.0, 1320.0],
        }
    )

    merged = merge_frozen_history(cached, downloaded)

    assert merged.iloc[:2].reset_index(drop=True).equals(cached)
    assert merged.iloc[-1]["close"] == 11.0
    assert merged.iloc[-1]["vol"] == 120.0


def freeze_fixture(root, date, status="READY", complete=True):
    manifest = root / f"results/audit/{date}_run_manifest.json"
    atomic_json(manifest, {"signal_date": date, "critical_files": [
        {"path": "market_data/prices/x.csv", "sha256": "frozen-evidence"}]})
    card = {"signal_date": date, "release": {"readiness": status},
            "observation": {"status": "complete" if complete else "unavailable"},
            "audit": {"run_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    atomic_json(root / f"results/audit/{date}_live_run_card.json", card)


@pytest.mark.parametrize("status", ["READY", "SELL_ONLY", "BLOCKED"])
def test_complete_report_freezes_prices_regardless_of_order_release(tmp_path, status):
    freeze_fixture(tmp_path, "2026-09-14")
    freeze_fixture(tmp_path, "2026-09-15", status)
    cutoff = audited_price_cutoff(tmp_path, "2026-09-16")
    assert cutoff == "2026-09-15"
    old = pd.DataFrame({"datetime": pd.to_datetime(["2026-09-14", "2026-09-15"]),
                        "close": [2.291, 2.307], "outstanding_share": [10, 11]})
    new = pd.DataFrame({"datetime": pd.to_datetime(["2026-09-14", "2026-09-15", "2026-09-16"]),
                        "close": [2.291, 2.307, 2.32], "outstanding_share": [30, 31, 32]})
    result = merge_frozen_history(old, new, cutoff)
    pd.testing.assert_frame_equal(result.iloc[:2].reset_index(drop=True), old)
    assert len(result) == 3 and result.iloc[-1].outstanding_share == 32


def test_unfinished_partial_and_future_runs_do_not_freeze_new_day(tmp_path):
    freeze_fixture(tmp_path, "2026-09-10")
    freeze_fixture(tmp_path, "2026-09-11", "BLOCKED", complete=False)
    freeze_fixture(tmp_path, "2026-09-14", "RUNNING")
    freeze_fixture(tmp_path, "2026-09-16")
    assert audited_price_cutoff(tmp_path, "2026-09-15") == "2026-09-10"


def test_bad_manifest_binding_stops_refresh_instead_of_rewriting_history(tmp_path):
    freeze_fixture(tmp_path, "2026-09-15", "BLOCKED")
    atomic_json(tmp_path / "results/audit/2026-09-15_run_manifest.json", {"signal_date": "2026-09-15"})
    with pytest.raises(ValueError, match="绑定失效"):
        audited_price_cutoff(tmp_path, "2026-09-16")


def test_no_audit_does_not_freeze_unreviewed_cache(tmp_path):
    assert audited_price_cutoff(tmp_path, "2026-09-16") == "1900-01-01"


def test_actual_latest_observation_sets_freeze_boundary():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert audited_price_cutoff(root, "2026-09-15") == "2026-09-15"
