from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.backtest import run_backtest


def panel(close: pd.DataFrame, open_price: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    return {
        "open": open_price if open_price is not None else close.copy(),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close.copy(),
        "vol": close * 0 + 1_000_000,
        "amount": close * 1_000_000,
    }


def project() -> dict[str, float | int]:
    return {
        "initial_capital": 100_000,
        "lot_size": 100,
        "commission_rate": 0.0,
        "minimum_commission": 0.0,
        "slippage_rate": 0.0,
    }


def test_close_signal_executes_on_next_open() -> None:
    dates = pd.bdate_range("2025-01-02", periods=4)
    symbol = "510300.SH"
    close = pd.DataFrame({symbol: [10.0, 11.0, 12.0, 13.0]}, index=dates)
    signals = pd.DataFrame({symbol: [1.0, 0.0, 0.0, 0.0]}, index=dates)
    result = run_backtest(
        "lag", panel(close), signals, str(dates[1].date()), str(dates[-1].date()), project()
    )
    assert pd.Timestamp(result.trades.iloc[0]["date"]) == dates[1]
    assert result.trades.iloc[0]["side"] == "BUY"


def test_untradeable_order_retries_next_open() -> None:
    dates = pd.bdate_range("2025-01-02", periods=5)
    symbol = "510300.SH"
    close = pd.DataFrame({symbol: [10.0] * 5}, index=dates)
    open_price = close.copy()
    open_price.loc[dates[1], symbol] = np.nan
    signals = pd.DataFrame({symbol: [1.0] * 5}, index=dates)
    result = run_backtest(
        "retry",
        panel(close, open_price),
        signals,
        str(dates[1].date()),
        str(dates[-1].date()),
        project(),
    )
    assert pd.Timestamp(result.trades.iloc[0]["date"]) == dates[2]


def test_idle_cash_accrues_actual_365_interest_before_next_open() -> None:
    dates = pd.to_datetime(["2025-01-03", "2025-01-06"])
    symbol = "510300.SH"
    close = pd.DataFrame({symbol: [10.0, 10.0]}, index=dates)
    signals = pd.DataFrame({symbol: [0.0, 0.0]}, index=dates)
    result = run_backtest(
        "cash",
        panel(close),
        signals,
        str(dates[0].date()),
        str(dates[-1].date()),
        project(),
        cash_annual_rate=0.015,
    )
    expected_interest = 100_000 * 0.015 * 3 / 365
    assert result.equity.iloc[-1] == pytest.approx(100_000 + expected_interest)
    assert result.metrics["cash_interest_income"] == pytest.approx(expected_interest)
