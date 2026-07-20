import numpy as np
import pandas as pd

from etf_rotation.etfwin import (
    EtfwinProfitProtection,
    EtfwinRules,
    etfwin_features,
    etfwin_signals,
)


def price_frame(rows: int = 180) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-01", periods=rows)
    return pd.DataFrame(
        {
            "steady": np.linspace(1.0, 1.8, rows),
            "slower": np.linspace(1.0, 1.4, rows),
            "falling": np.linspace(1.8, 1.0, rows),
        },
        index=index,
    )


def test_public_score_is_roc20_plus_one_point_five_roc60() -> None:
    close = price_frame()
    rules = EtfwinRules()
    features = etfwin_features(close, rules)
    expected = features.roc_short + 1.5 * features.roc_medium
    pd.testing.assert_frame_equal(features.raw_score, expected)


def test_entry_requires_positive_rocs_above_ma_and_bias_cap() -> None:
    close = price_frame()
    rules = EtfwinRules(max_entry_ma_bias=0.50)
    bundle, features = etfwin_signals(close, list(close.columns), rules)
    final = close.index[-1]
    assert features.roc_short.loc[final, "steady"] > 0
    assert features.roc_medium.loc[final, "steady"] > 0
    assert bundle.weights.loc[final, "steady"] == 1.0
    assert bundle.weights.loc[final, "falling"] == 0.0


def test_ma120_break_is_not_delayed_by_soft_confirmation() -> None:
    close = price_frame()
    rules = EtfwinRules(max_entry_ma_bias=0.50)
    confirmation = pd.DataFrame(False, index=close.index, columns=close.columns)
    bundle, _ = etfwin_signals(
        close,
        list(close.columns),
        rules,
        soft_exit_confirmation=confirmation,
    )
    held_before = bundle.weights["steady"].gt(0).any()
    assert held_before

    shocked = close.copy()
    shocked.loc[shocked.index[-1], "steady"] = 0.5
    bundle, _ = etfwin_signals(
        shocked,
        list(shocked.columns),
        rules,
        soft_exit_confirmation=confirmation,
    )
    assert bundle.weights.loc[shocked.index[-1], "steady"] == 0.0
    assert "跌破MA120" in bundle.diagnostics.loc[shocked.index[-1], "exit_reasons"]


def test_ma120_break_can_require_two_consecutive_closes() -> None:
    close = price_frame()
    shocked = close.copy()
    shocked.loc[shocked.index[-2]:, "steady"] = 0.5
    rules = EtfwinRules(
        max_entry_ma_bias=0.50,
        ma_exit_confirmation_days=2,
        exit_on_short_roc_negative=False,
        exit_on_dual_rank_decline=False,
    )
    bundle, _ = etfwin_signals(shocked, list(shocked.columns), rules)
    assert bundle.weights.loc[shocked.index[-2], "steady"] == 1.0
    assert bundle.weights.loc[shocked.index[-1], "steady"] == 0.0
    assert "跌破MA120" in bundle.diagnostics.loc[shocked.index[-1], "exit_reasons"]


def test_roc20_exit_can_require_two_consecutive_negative_closes() -> None:
    close = price_frame()[["steady"]].copy()
    close.loc[close.index[-2]:, "steady"] = 0.5
    rules = EtfwinRules(
        roc_short_days=5,
        roc_medium_days=10,
        ma_days=20,
        max_entry_ma_bias=1.0,
        rank_change_short_days=3,
        rank_change_long_days=6,
        roc_exit_confirmation_days=2,
        exit_on_ma_break=False,
        exit_on_dual_rank_decline=False,
    )
    bundle, _ = etfwin_signals(close, ["steady"], rules)
    assert bundle.weights.loc[close.index[-2], "steady"] == 1.0
    assert bundle.weights.loc[close.index[-1], "steady"] == 0.0
    assert "ROC20转负" in bundle.diagnostics.loc[close.index[-1], "exit_reasons"]


def test_score_override_changes_only_the_cross_sectional_ranking() -> None:
    close = price_frame()
    override = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    override.loc[close.index[-1]] = {"steady": 1.0, "slower": 3.0, "falling": 2.0}
    features = etfwin_features(
        close, EtfwinRules(), raw_score_override=override
    )
    assert features.raw_score.loc[close.index[-1], "slower"] == 3.0
    assert features.rank.loc[close.index[-1], "slower"] == 1.0


def test_score_override_keeps_every_below_ma_asset_behind_above_ma_assets() -> None:
    close = price_frame()
    override = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    override.loc[close.index[-1]] = {
        "steady": 1.0,
        "slower": 2.0,
        "falling": 1000.0,
    }
    features = etfwin_features(
        close, EtfwinRules(), raw_score_override=override
    )
    final = close.index[-1]
    assert features.above_ma.loc[final, "steady"]
    assert not features.above_ma.loc[final, "falling"]
    assert features.rank.loc[final, "falling"] > features.rank.loc[final, "steady"]


def test_eligible_only_ranking_excludes_unbuyable_asset_from_entry_slots() -> None:
    close = price_frame()
    rules = EtfwinRules(
        roc_short_days=2,
        roc_medium_days=4,
        ma_days=5,
        entry_rank_limit=1,
        max_entry_ma_bias=1.0,
        rank_change_short_days=2,
        rank_change_long_days=4,
        exit_on_ma_break=False,
        exit_on_short_roc_negative=False,
        exit_on_dual_rank_decline=False,
    )
    eligibility = pd.DataFrame(True, index=close.index, columns=close.columns)
    eligibility["steady"] = False

    fixed_pool = etfwin_signals(
        close,
        list(close.columns),
        rules,
        entry_eligibility=eligibility,
    )[0]
    eligible_pool = etfwin_signals(
        close,
        list(close.columns),
        rules,
        entry_eligibility=eligibility,
        rank_only_entry_eligible=True,
    )[0]

    assert fixed_pool.weights.iloc[-1].sum() == 0.0
    assert eligible_pool.weights.iloc[-1]["slower"] == 1.0


def test_entry_ranking_override_changes_entry_without_replacing_exit_features() -> None:
    close = price_frame()
    rules = EtfwinRules(
        roc_short_days=2,
        roc_medium_days=4,
        ma_days=5,
        entry_rank_limit=1,
        max_entry_ma_bias=1.0,
        rank_change_short_days=2,
        rank_change_long_days=4,
        exit_on_ma_break=False,
        exit_on_short_roc_negative=False,
        exit_on_dual_rank_decline=False,
    )
    baseline, baseline_features = etfwin_signals(
        close, list(close.columns), rules
    )
    entry_score = baseline_features.ranking_score.copy()
    entry_score["slower"] += 100.0
    overridden, overridden_features = etfwin_signals(
        close,
        list(close.columns),
        rules,
        entry_ranking_score_override=entry_score,
    )

    assert baseline.weights.iloc[-1]["steady"] == 1.0
    assert overridden.weights.iloc[-1]["slower"] == 1.0
    pd.testing.assert_frame_equal(
        baseline_features.rank, overridden_features.rank
    )


def test_entry_ranking_override_orders_multiple_eligible_candidates() -> None:
    close = price_frame()
    rules = EtfwinRules(
        roc_short_days=2,
        roc_medium_days=4,
        ma_days=5,
        entry_rank_limit=2,
        max_entry_ma_bias=1.0,
        rank_change_short_days=2,
        rank_change_long_days=4,
        exit_on_ma_break=False,
        exit_on_short_roc_negative=False,
        exit_on_dual_rank_decline=False,
    )
    _, features = etfwin_signals(close, list(close.columns), rules)
    entry_score = features.ranking_score.copy()
    entry_score["slower"] += 100.0
    overridden, _ = etfwin_signals(
        close,
        list(close.columns),
        rules,
        entry_ranking_score_override=entry_score,
    )

    assert overridden.weights.iloc[-1]["slower"] == 1.0


def test_atr_profit_protection_exits_after_activation_and_giveback() -> None:
    close = price_frame()[["steady"]].copy()
    close.loc[close.index[-1], "steady"] -= 0.12
    atr = pd.DataFrame(0.02, index=close.index, columns=close.columns)
    rules = EtfwinRules(
        roc_short_days=5,
        roc_medium_days=10,
        ma_days=20,
        max_entry_ma_bias=1.0,
        rank_change_short_days=3,
        rank_change_long_days=6,
        exit_on_ma_break=False,
        exit_on_short_roc_negative=False,
        exit_on_dual_rank_decline=False,
    )
    bundle, _ = etfwin_signals(
        close,
        ["steady"],
        rules,
        atr=atr,
        profit_protection=EtfwinProfitProtection(
            activation_atr=1.0, trail_atr=2.0
        ),
    )
    final = close.index[-1]
    assert bundle.weights.loc[final, "steady"] == 0.0
    assert "ATR利润保护" in bundle.diagnostics.loc[final, "exit_reasons"]


def test_reentry_cooldown_blocks_only_recently_exited_symbol() -> None:
    close = price_frame()[["steady"]]
    rules = EtfwinRules(
        roc_short_days=2,
        roc_medium_days=4,
        ma_days=5,
        max_entry_ma_bias=1.0,
        rank_change_short_days=2,
        rank_change_long_days=4,
        exit_on_ma_break=False,
        exit_on_short_roc_negative=False,
        exit_on_dual_rank_decline=False,
    )
    emergency = pd.DataFrame(False, index=close.index, columns=close.columns)
    exit_location = len(close) - 6
    emergency.iloc[exit_location, 0] = True
    bundle, _ = etfwin_signals(
        close,
        ["steady"],
        rules,
        emergency_exit=emergency,
        reentry_cooldown_days=3,
    )

    assert bundle.weights.iloc[exit_location : exit_location + 4, 0].eq(0.0).all()
    assert bundle.weights.iloc[exit_location + 4, 0] == 1.0
