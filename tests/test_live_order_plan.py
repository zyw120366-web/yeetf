from __future__ import annotations

import pandas as pd

from scripts.build_live_order_plan import choose_live_target, exit_reasons, order_candidates


def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "512010.SH", "momentum_score": 0.17},
            {"symbol": "159647.SZ", "momentum_score": 0.13},
            {"symbol": "159985.SZ", "momentum_score": 0.12},
        ]
    )


def test_first_live_day_uses_confirmed_empty_account_not_backtest_holding() -> None:
    assert choose_live_target(None, candidates(), []) == "512010.SH"


def test_confirmed_holding_is_kept_until_an_exit_rule_fires() -> None:
    assert choose_live_target("159985.SZ", candidates(), []) == "159985.SZ"
    assert choose_live_target("159985.SZ", candidates(), ["ROC20转负"]) == "512010.SH"


def test_exit_reasons_respect_hot_soft_exit_protection_but_not_ma_break() -> None:
    protected = pd.Series(
        {
            "above_ma120": True,
            "roc20": -0.01,
            "dual_rank_decline": True,
            "soft_exit_confirmation": False,
        }
    )
    assert exit_reasons(protected) == []
    protected["above_ma120"] = False
    assert exit_reasons(protected) == ["跌破MA120（硬退出）"]


def test_candidates_use_path_specific_selection_score() -> None:
    mixed = pd.DataFrame(
        [
            {"symbol": "normal", "rank": 1, "momentum_score": 0.20, "selection_score": 0.20},
            {"symbol": "emerging", "rank": 8, "momentum_score": 0.30, "selection_score": 0.18},
        ]
    )
    ordered = order_candidates(mixed)
    assert list(ordered["symbol"]) == ["normal", "emerging"]
