from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.backtest import BacktestResult
from etf_rotation.evaluation import realized_round_trips, round_trip_timing, timing_summary


def result_with_exit(dates: pd.DatetimeIndex, exit_index: int) -> BacktestResult:
    trades = pd.DataFrame(
        [
            {"date": dates[2], "symbol": "A", "side": "BUY", "qty": 100, "price": 10.0},
            {
                "date": dates[exit_index],
                "symbol": "A",
                "side": "SELL",
                "qty": 100,
                "price": 10.1,
            },
        ]
    )
    equity = pd.Series(100_000.0, index=dates)
    return BacktestResult("test", equity, equity.pct_change(), trades, pd.DataFrame(), {})


def test_fixed_horizon_labels_do_not_expand_with_holding_period() -> None:
    dates = pd.bdate_range("2025-01-02", periods=40)
    prices = [10.0] * 40
    prices[12] = 10.7
    prices[13] = 10.8
    panel = {"close": pd.DataFrame({"A": prices}, index=dates)}

    early = round_trip_timing(result_with_exit(dates, 8), panel)
    late = round_trip_timing(result_with_exit(dates, 11), panel)

    assert early.iloc[0]["entry_forward_max_return_20d"] == late.iloc[0][
        "entry_forward_max_return_20d"
    ]
    assert not bool(early.iloc[0]["fixed_horizon_false_buy"])
    assert bool(early.iloc[0]["material_premature_exit"])
    summary = timing_summary(early)
    assert summary["fixed_horizon_false_buy_rate"] == 0.0
    assert summary["material_premature_exit_rate"] == 1.0
    assert summary["failed_operation_rate"] == 1.0


def test_failed_operation_rate_uses_one_label_per_round_trip() -> None:
    timing = pd.DataFrame(
        {
            "entry_delay_days": [0, 0, 0, 0],
            "exit_delay_days": [0, 0, 0, 0],
            "peak_giveback": [0.0, 0.0, 0.0, 0.0],
            "move_capture_ratio": [1.0, 1.0, 1.0, 1.0],
            "false_start": [False, False, False, False],
            "fixed_horizon_false_buy": [False, True, False, True],
            "material_premature_exit": [False, False, True, True],
            "failed_operation": [False, True, True, True],
        }
    )

    summary = timing_summary(timing)

    assert summary["failed_operation_rate"] == 0.75


def test_realized_round_trips_aggregate_partial_fills_and_costs() -> None:
    dates = pd.bdate_range("2025-01-02", periods=8)
    trades = pd.DataFrame(
        [
            {"date": dates[0], "symbol": "A", "side": "BUY", "qty": 60, "price": 10.0, "fee": 2.0},
            {"date": dates[1], "symbol": "A", "side": "BUY", "qty": 40, "price": 11.0, "fee": 2.0},
            {"date": dates[4], "symbol": "A", "side": "SELL", "qty": 30, "price": 12.0, "fee": 1.0},
            {"date": dates[5], "symbol": "A", "side": "SELL", "qty": 70, "price": 13.0, "fee": 2.0},
            {"date": dates[6], "symbol": "B", "side": "BUY", "qty": 10, "price": 5.0, "fee": 1.0},
        ]
    )

    rounds = realized_round_trips(trades, dates)

    assert len(rounds) == 1
    row = rounds.iloc[0]
    assert row["symbol"] == "A"
    assert row["entry_date"] == dates[0]
    assert row["entry_completed_date"] == dates[1]
    assert row["exit_date"] == dates[4]
    assert row["exit_completed_date"] == dates[5]
    assert row["holding_days"] == 5
    assert row["buy_fill_count"] == 2
    assert row["sell_fill_count"] == 2
    assert row["buy_cost"] == 1044.0
    assert row["sell_proceeds"] == 1267.0
    assert row["net_pnl"] == 223.0
    assert row["net_return"] == 1267.0 / 1044.0 - 1.0
    assert row["outcome"] == "胜"
