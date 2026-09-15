from __future__ import annotations

import pandas as pd

from etf_rotation.data import merge_frozen_history


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
