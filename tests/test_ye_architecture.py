from __future__ import annotations

import pandas as pd

from etf_rotation.ye import anchor_category_breadth, anchor_competitive_rank


def test_ineligible_challenger_cannot_displace_core_rank() -> None:
    dates = pd.bdate_range("2026-01-01", periods=2)
    scores = pd.DataFrame(
        {"CORE_A": [0.20, 0.20], "CORE_B": [0.10, 0.10], "NEW": [0.50, 0.50]},
        index=dates,
    )
    eligibility = pd.DataFrame(True, index=dates, columns=scores.columns)
    eligibility["NEW"] = [False, True]

    rank = anchor_competitive_rank(
        scores, eligibility, ["CORE_A", "CORE_B"], ["NEW"]
    )

    assert rank.loc[dates[0], "CORE_A"] == 1.0
    assert rank.loc[dates[0], "CORE_B"] == 2.0
    assert pd.isna(rank.loc[dates[0], "NEW"])
    assert rank.loc[dates[1], "CORE_A"] == 1.0
    assert rank.loc[dates[1], "CORE_B"] == 2.0
    assert rank.loc[dates[1], "NEW"] == 1.0


def test_challenger_never_dilutes_core_category_breadth() -> None:
    dates = pd.bdate_range("2026-01-01", periods=2)
    roc20 = pd.DataFrame(
        {"CORE_A": [0.10, 0.10], "CORE_B": [-0.02, -0.02], "NEW": [-0.05, 0.05]},
        index=dates,
    )
    eligibility = pd.DataFrame(True, index=dates, columns=roc20.columns)
    categories = {symbol: "主题" for symbol in roc20.columns}

    breadth = anchor_category_breadth(
        roc20,
        categories,
        eligibility,
        ["CORE_A", "CORE_B"],
        ["NEW"],
    )

    assert breadth.loc[dates[0], "CORE_A"] == 0.5
    assert breadth.loc[dates[1], "CORE_A"] == 0.5
    assert breadth.loc[dates[0], "NEW"] == 0.5
    assert breadth.loc[dates[1], "NEW"] == 0.5


def test_challenger_cannot_self_confirm_a_new_category() -> None:
    dates = pd.bdate_range("2026-01-01", periods=1)
    roc20 = pd.DataFrame({"CORE": [0.10], "NEW": [0.20]}, index=dates)
    eligibility = pd.DataFrame(True, index=dates, columns=roc20.columns)

    breadth = anchor_category_breadth(
        roc20,
        {"CORE": "核心主题", "NEW": "全新主题"},
        eligibility,
        ["CORE"],
        ["NEW"],
    )

    assert pd.isna(breadth.loc[dates[0], "NEW"])
