import pandas as pd
import numpy as np
from experiments import price_basis_audit_20260911 as audit
from etf_rotation.backtest import run_backtest


def test_cash_dividend_does_not_become_price_loss_or_past_return():
    dates = pd.date_range("2020-01-01", periods=3)
    raw = pd.DataFrame({"close": [10., 9., 9.9]}, index=dates)
    events = {dates[1]: {"ratio": 1., "dividend": 1.}}
    result = audit.economic_close(raw, events)
    assert np.allclose(result, [10., 10., 11.])
    # Unknown future events do not change prices before they occur.
    pd.testing.assert_series_equal(
        audit.economic_close(raw.iloc[:1], {}), result.iloc[:1])


def test_share_split_is_not_a_crash():
    dates = pd.date_range("2020-01-01", periods=3)
    raw = pd.DataFrame({"close": [10., 5., 5.5]}, index=dates)
    result = audit.economic_close(raw, {dates[1]: {"ratio": 2., "dividend": 0.}})
    assert np.allclose(result, [10., 10., 11.])


def test_adapted_engine_matches_without_actions_and_accounts_dividend(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "OUT", tmp_path)
    adapted = audit.raw_execution_function()
    dates = pd.date_range("2020-01-01", periods=3)
    panel = {k: pd.DataFrame({"a": [10., 9., 9.]}, index=dates) for k in ("open", "close")}
    weights = pd.DataFrame({"a": 1.}, index=dates)
    project = {"initial_capital": 100000, "lot_size": 100, "commission_rate": 0.,
               "minimum_commission": 0., "slippage_rate": 0.}
    a = run_backtest("test", panel, weights, str(dates[0]), str(dates[-1]), project)
    b = adapted("test", panel, weights, str(dates[0]), str(dates[-1]), project)
    pd.testing.assert_series_equal(a.equity, b.equity)
    project["corporate_actions"] = {dates[1]: {"a": {"ratio": 1., "dividend": 1.}}}
    c = adapted("test", panel, weights, str(dates[0]), str(dates[-1]), project)
    assert np.allclose(c.equity, 100000.)
