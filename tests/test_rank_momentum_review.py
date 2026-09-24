import numpy as np
import pandas as pd
import yaml

from experiments.rank_momentum_review import ROOT, own_momentum_protection, episode_intervals


def rules():
    return yaml.safe_load((ROOT / "config/ye_strategy.yaml").read_text())["rules"]


def test_own_momentum_uses_both_existing_windows_and_no_rank_threshold():
    dates = pd.bdate_range("2020-01-01", periods=130)
    close = pd.DataFrame({"growing": np.exp(np.arange(130)**2 / 20000)}, index=dates)
    r = rules()
    score = close.pct_change(20, fill_method=None) + 1.5 * close.pct_change(60, fill_method=None)
    expected = score.ge(score.shift(5)) & score.ge(score.shift(20))
    actual = own_momentum_protection(close, r)
    pd.testing.assert_frame_equal(actual, expected)
    assert not actual.iloc[:80].any().any()
    assert actual.iloc[80:].all().all()


def test_future_prices_do_not_change_past_protection():
    dates = pd.bdate_range("2020-01-01", periods=140)
    close = pd.DataFrame({"x": np.exp(np.arange(140)**2 / 30000)}, index=dates)
    original = own_momentum_protection(close, rules())
    close.iloc[120:] *= .25
    changed = own_momentum_protection(close, rules())
    pd.testing.assert_frame_equal(original.iloc[:120], changed.iloc[:120])


def test_nonfinite_history_never_grants_protection():
    close = pd.DataFrame({"zero": [0.] * 60 + [1.] * 70, "missing": [np.nan] * 130})
    actual = own_momentum_protection(close, rules())
    assert not actual["missing"].any()
    assert not actual["zero"].iloc[:125].any()


def test_signal_episodes_execute_next_session_and_censor_unfinished():
    dates = pd.bdate_range("2026-01-01", periods=7)
    base = pd.DataFrame({"x": [0] * 7}, index=dates)
    candidate = pd.DataFrame({"x": [0, 1, 1, 0, 0, 1, 1]}, index=dates)
    first, last = episode_intervals(base, candidate)
    assert first == {"signal_start": dates[1], "execute_start": dates[2], "execute_end": dates[4], "completed": True}
    assert last == {"signal_start": dates[5], "execute_start": dates[6], "execute_end": dates[6], "completed": False}
