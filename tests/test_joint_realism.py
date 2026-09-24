"""Validate the fixed research design without modifying formal artifacts."""
import numpy as np
import pandas as pd

from experiments.joint_realism_20260914 import COMPONENTS, VARIANTS, spec
from experiments.price_basis_audit_20260911 import economic_close, raw_execution_function
from experiments import price_basis_audit_20260911 as price
from experiments.structural_audit_20260911 import remove_dde
from etf_rotation.sentiment_ai import deduplicate_reviewed_rows
from experiments.joint_realism_20260914 import checks_for, STARTS


def test_fixed_design_is_ten_unique_variants_seventy_runs():
    assert len(VARIANTS) == len(set(VARIANTS)) == 10
    assert len(VARIANTS) * (5 + 2) == 70
    assert spec("rich") == (False, False, False, None)
    assert spec("joint") == (True, True, False, None)
    for component in COMPONENTS:
        assert spec("without_" + component) == (True, True, True, {component: False})
    assert spec("without_all")[3] == dict.fromkeys(COMPONENTS, False)


def test_missing_dde_stress_is_idempotent():
    original = pd.DataFrame({"matched_count": [3, 0, 4], "positive_dde_share": [.8, np.nan, .1], "hot_score": [.7, .4, .3]})
    once = remove_dde(original)
    pd.testing.assert_frame_equal(once, remove_dde(once))
    assert once.loc[0, "positive_dde_share"] == 0
    assert original.loc[0, "positive_dde_share"] == .8


def test_combined_split_dividend_is_causal():
    days = pd.date_range("2020-01-01", periods=5)
    raw = pd.DataFrame({"close": [10., 11., 5., 5.5, 5.]}, index=days)
    events = {days[2]: {"ratio": 2., "dividend": 1.}, days[4]: {"ratio": 1., "dividend": .5}}
    full = economic_close(raw, events)
    assert np.allclose(full, [10, 11, 11, 12.1, 12.1])
    for count in range(1, 6):
        subset = {d: e for d, e in events.items() if d <= days[count - 1]}
        pd.testing.assert_series_equal(economic_close(raw.iloc[:count], subset), full.iloc[:count])


def test_raw_ledger_retains_split_shares_and_dividend_on_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(price, "OUT", tmp_path)
    run = raw_execution_function()
    days = pd.date_range("2020-01-01", periods=4)
    panel = {k: pd.DataFrame({"etf": [10., 4.5, 5., 5.]}, index=days) for k in ("open", "close")}
    weights = pd.DataFrame({"etf": [1., 1., 0., 0.]}, index=days)
    project = {"initial_capital": 100000., "lot_size": 100, "commission_rate": 0.,
               "minimum_commission": 0., "slippage_rate": 0.,
               "corporate_actions": {days[1]: {"etf": {"ratio": 2., "dividend": 1.}}}}
    result = run("test", panel, weights, str(days[0]), str(days[-1]), project)
    assert np.allclose(result.equity, [100000, 100000, 110000, 110000])
    sells = result.trades[result.trades.side.eq("SELL")]
    assert sells.qty.sum() == 20000
    assert result.actual_weights.iloc[-1].sum() == 0


def test_reproduce_existing_equal_score_dedup_field_loss():
    # Documents a production defect; it does NOT endorse source-order dependence.
    common = {"name": "同一公司", "published_at": "2026-09-10", "ai": {"direction": 2, "confidence": .9}}
    empty = {**common, "source_hash": "a", "turnover": None}
    full = {**common, "source_hash": "b", "turnover": 1000000.}
    assert deduplicate_reviewed_rows([empty, full])[0]["turnover"] is None
    assert deduplicate_reviewed_rows([full, empty])[0]["turnover"] == 1000000.


def test_acceptance_json_serializable_and_ties_are_not_improvement():
    import json
    records = []
    for variant in ("joint_valid", "without_all"):
        for start in STARTS:
            for cost, capital in [(1, 100000), (2, 100000), (1, 15275)]:
                records.append(dict(variant=variant, start=start, cost=cost, capital=capital,
                                    total_return=1., sharpe=1., max_drawdown=-.2))
    phases = pd.DataFrame([{"variant": v, "period": p, "total_return": .2}
                           for v in ("joint_valid", "without_all") for p in range(4)])
    result = checks_for(pd.DataFrame(records), phases, "without_all")
    assert not result["passed"]
    assert result["starts_won"] == 0
    assert json.loads(json.dumps(result)) == result
