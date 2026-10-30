"""Small invariant checks for isolated research, not new live rules."""
import numpy as np
import pandas as pd
from experiments.structural_audit_20260911 import observed_breadth, remove_dde


def test_unlisted_peer_is_unknown_not_negative():
    roc = pd.DataFrame({"old": [.1, -.1], "future": [np.nan, .1], "satellite": [.2, .2]})
    categories = {s: "commodity" for s in roc}
    got = observed_breadth(roc, categories, roc.notna(), ["old", "future"], ["satellite"])
    assert got.old.tolist() == [1., .5]
    assert got.satellite.tolist() == [1., .5]


def test_satellite_cannot_confirm_itself():
    roc = pd.DataFrame({"old": [.1], "satellite": [.2]})
    got = observed_breadth(roc, {"old": "a", "satellite": "b"}, roc.notna(), ["old"], ["satellite"])
    assert got.satellite.isna().all()


def test_missing_dde_replication_preserves_count_and_other_score_parts():
    f = pd.DataFrame({"matched_count": [4, 0], "positive_dde_share": [.75, np.nan],
                      "hot_score": [.8, .4]})
    got = remove_dde(f)
    assert np.isclose(got.hot_score.iloc[0], .725)
    assert np.isclose(got.hot_score.iloc[1], .4)
    assert got.positive_dde_share.iloc[0] == 0.
    assert np.isnan(got.positive_dde_share.iloc[1])
    pd.testing.assert_series_equal(got.matched_count, f.matched_count)
    assert f.positive_dde_share.iloc[0] == .75


def test_breadth_append_future_rows_does_not_rewrite_past():
    f = pd.DataFrame({"old": [.1, -.1], "future": [np.nan, .2]})
    args = ({"old": "a", "future": "a"}, f.notna(), ["old", "future"], [])
    left = observed_breadth(f.iloc[:1], *args)
    right = observed_breadth(f, *args).iloc[:1]
    pd.testing.assert_frame_equal(left, right)
