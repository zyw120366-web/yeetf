from __future__ import annotations

import numpy as np
import pandas as pd

from .etfwin import EtfwinRules, etfwin_features, etfwin_signals
from .execution import entry_eligibility
from .sentiment import broadcast


def rolling_r2(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    xss = float(centered_x @ centered_x)

    def one(values: np.ndarray) -> float:
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            return np.nan
        y = np.log(values)
        centered_y = y - y.mean()
        yss = float(centered_y @ centered_y)
        return float(((centered_x @ centered_y) ** 2) / (xss * yss)) if yss > 1e-16 else 0.0

    return frame.rolling(window, min_periods=window).apply(one, raw=True)


def category_breadth(
    roc20: pd.DataFrame, categories: dict[str, str]
) -> pd.DataFrame:
    output = pd.DataFrame(index=roc20.index, columns=roc20.columns, dtype=float)
    groups: dict[str, list[str]] = {}
    for symbol, category in categories.items():
        groups.setdefault(category, []).append(symbol)
    for members in groups.values():
        share = roc20[members].gt(0.0).mean(axis=1)
        output.loc[:, members] = np.repeat(
            share.to_numpy()[:, None], len(members), axis=1
        )
    return output


def anchor_category_breadth(
    roc20: pd.DataFrame,
    categories: dict[str, str],
    eligibility: pd.DataFrame,
    core_symbols: list[str],
    challenger_symbols: list[str],
) -> pd.DataFrame:
    """Keep core breadth frozen and judge challengers only by core peers."""

    output = pd.DataFrame(index=roc20.index, columns=roc20.columns, dtype=float)
    groups: dict[str, list[str]] = {}
    for symbol in core_symbols:
        groups.setdefault(categories[symbol], []).append(symbol)
    for members in groups.values():
        share = roc20[members].gt(0.0).mean(axis=1)
        output.loc[:, members] = np.repeat(
            share.to_numpy()[:, None], len(members), axis=1
        )
    for symbol in challenger_symbols:
        members = groups.get(categories[symbol], [])
        if members:
            output[symbol] = roc20[members].gt(0.0).mean(axis=1)
        else:
            # A challenger in a newly covered category cannot use itself as
            # 100% breadth confirmation during the historical no-news regime.
            output[symbol] = np.nan
    return output


def anchor_competitive_rank(
    ranking_score: pd.DataFrame,
    eligibility: pd.DataFrame,
    core_symbols: list[str],
    challenger_symbols: list[str],
) -> pd.DataFrame:
    """Rank core ETFs only against the core; challengers compete against that ruler."""

    rank = pd.DataFrame(index=ranking_score.index, columns=ranking_score.columns, dtype=float)
    core_scores = ranking_score[core_symbols]
    rank.loc[:, core_symbols] = core_scores.rank(
        axis=1, ascending=False, method="min"
    )
    for symbol in challenger_symbols:
        score = ranking_score[symbol]
        rank[symbol] = (
            core_scores.gt(score, axis=0).sum(axis=1).astype(float) + 1.0
        ).where(score.notna() & eligibility[symbol])
    return rank


def _rules(values: dict) -> EtfwinRules:
    """Use a wide technical shell; the frozen ye gates below decide entries."""

    return EtfwinRules(
        roc_short_days=int(values["roc_short_days"]),
        roc_medium_days=int(values["roc_medium_days"]),
        roc_short_weight=float(values["roc_short_weight"]),
        roc_medium_weight=float(values["roc_medium_weight"]),
        entry_rank_limit=1000,
        ma_days=int(values["ma_days"]),
        max_entry_ma_bias=0.25,
        rank_change_short_days=int(values["rank_change_short_days"]),
        rank_change_long_days=int(values["rank_change_long_days"]),
        exit_confirmation_days=int(values.get("exit_confirmation_days", 1)),
        ma_exit_confirmation_days=int(values.get("ma_exit_confirmation_days", 1)),
        roc_exit_confirmation_days=int(values.get("roc_exit_confirmation_days", 1)),
        holdings_num=int(values["holdings_num"]),
        require_short_roc_positive=False,
        require_medium_roc_positive=False,
        require_above_ma=False,
        exit_on_ma_break=bool(values["exit_on_ma_break"]),
        exit_on_short_roc_negative=bool(values["exit_on_short_roc_negative"]),
        exit_on_dual_rank_decline=bool(values["exit_on_dual_rank_decline"]),
        penalize_below_ma_for_ranking=True,
    )


def build_ye_signals(
    panel: dict[str, pd.DataFrame],
    symbols: list[str],
    categories: dict[str, str],
    config: dict,
    sentiment: dict[str, pd.DataFrame],
    sentiment_available: pd.Series,
    *,
    raw_score_override: pd.DataFrame | None = None,
    components: dict[str, bool] | None = None,
):
    """Build the one frozen ye strategy from production configuration only."""

    values = config["rules"]
    enhanced = config["enhanced_selection"]
    live = enhanced["sentiment_available"]
    fallback_cfg = enhanced["price_only_for_historical_dates_without_sentiment"]
    close = panel["close"][symbols]
    calendar = close.index
    eligibility, listed_sessions, trailing_amount = entry_eligibility(
        panel, symbols, values
    )
    rules = _rules(values)
    features = etfwin_features(
        close, rules, raw_score_override=raw_score_override
    )
    returns = close.pct_change(fill_method=None)
    r2 = rolling_r2(close, 20)
    efficiency = close.pct_change(20, fill_method=None).abs() / returns.abs().rolling(20).sum()
    roc5 = close.pct_change(5, fill_method=None)
    architecture = enhanced.get("universe_architecture", {})
    architecture_mode = str(architecture.get("mode", "fixed_pool"))
    configured_challengers = [str(symbol) for symbol in architecture.get("challenger_symbols", [])]
    missing_challengers = set(configured_challengers) - set(symbols)
    if missing_challengers:
        raise KeyError(
            "configured challenger symbols are missing from the universe: "
            f"{sorted(missing_challengers)}"
        )
    challenger_symbols = [symbol for symbol in symbols if symbol in configured_challengers]
    core_symbols = [symbol for symbol in symbols if symbol not in set(challenger_symbols)]
    if architecture_mode == "core_anchor_challenger":
        expected_core = int(architecture["core_pool_size"])
        if len(core_symbols) != expected_core:
            raise ValueError(
                f"core pool size mismatch: expected {expected_core}, got {len(core_symbols)}"
            )
        breadth = anchor_category_breadth(
            features.roc_short,
            categories,
            eligibility,
            core_symbols,
            challenger_symbols,
        )
        decision_rank = anchor_competitive_rank(
            features.ranking_score,
            eligibility,
            core_symbols,
            challenger_symbols,
        )
    elif architecture_mode == "fixed_pool":
        breadth = category_breadth(features.roc_short, categories)
        decision_rank = features.rank
    else:
        raise ValueError(f"unsupported universe architecture: {architecture_mode}")
    dual_rank_decline = (
        decision_rank.gt(decision_rank.shift(int(values["rank_change_short_days"])))
        & decision_rank.gt(decision_rank.shift(int(values["rank_change_long_days"])))
    ).fillna(False)
    available = broadcast(sentiment_available, symbols)
    component_flags = {
        "weak_edge_filter": True,
        "emerging_trend": True,
        "quality_extension": True,
        "hot_exit_protection": True,
        **(components or {}),
    }

    normal = (
        features.roc_short.gt(0.0)
        & features.roc_medium.gt(0.0)
        & features.above_ma
        & features.ma_bias.le(float(values["max_entry_ma_bias"]))
        & decision_rank.le(int(values["entry_rank_limit"]))
    )
    fallback = (
        normal
        & decision_rank.le(int(fallback_cfg["entry_rank_limit"]))
        & breadth.ge(float(fallback_cfg["category_roc20_positive_breadth_min"]))
    )

    weak_edge = decision_rank.ge(4) & features.roc_short.lt(0.02)
    edge_following = (
        sentiment["matched_count"].ge(3)
        & sentiment["count_acceleration"].ge(0.0)
        & sentiment["positive_dde_share"].ge(0.50)
    )
    current_normal = (
        normal & (~weak_edge | edge_following)
        if component_flags["weak_edge_filter"]
        else normal.copy()
    )

    emerging_cfg = live["emerging_trend"]
    emerging_trigger = (
        features.roc_short.ge(float(emerging_cfg["roc20_min"]))
        & features.roc_medium.ge(float(emerging_cfg["roc60_range"][0]))
        & features.roc_medium.le(float(emerging_cfg["roc60_range"][1]))
        & features.ma_bias.ge(float(emerging_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(emerging_cfg["ma120_bias_range"][1]))
        & decision_rank.le(int(emerging_cfg["maximum_base_rank"]))
        & r2.ge(float(emerging_cfg["r2_20_min"]))
        & efficiency.ge(float(emerging_cfg["efficiency20_min"]))
        & sentiment["matched_count"].ge(float(emerging_cfg["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(emerging_cfg["hot_score_min"]))
        & sentiment["count_acceleration"].ge(float(emerging_cfg["count_acceleration_min"]))
        & sentiment["positive_dde_share"].ge(float(emerging_cfg["positive_dde_share_min"]))
    )
    emerging = emerging_trigger.rolling(
        int(emerging_cfg["memory_days"]), min_periods=1
    ).max().fillna(False).astype(bool)
    emerging &= (
        features.roc_short.gt(0.0)
        & features.roc_medium.ge(float(emerging_cfg["roc60_range"][0]))
        & features.ma_bias.ge(float(emerging_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(live["quality_extension"]["ma120_bias_range"][1]))
    )
    if not component_flags["emerging_trend"]:
        emerging = pd.DataFrame(False, index=calendar, columns=symbols)

    extension_cfg = live["quality_extension"]
    quality_extension = (
        features.roc_short.gt(0.0)
        & features.roc_medium.gt(0.0)
        & features.above_ma
        & decision_rank.le(int(extension_cfg["base_rank_limit"]))
        & features.ma_bias.gt(float(extension_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(extension_cfg["ma120_bias_range"][1]))
        & r2.ge(float(extension_cfg["r2_20_min"]))
        & efficiency.ge(float(extension_cfg["efficiency20_min"]))
        & roc5.ge(float(extension_cfg["roc5_min"]))
        & sentiment["matched_count"].ge(float(extension_cfg["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(extension_cfg["hot_score_min"]))
        & sentiment["count_acceleration"].ge(float(extension_cfg["count_acceleration_min"]))
    )
    if not component_flags["quality_extension"]:
        quality_extension = pd.DataFrame(False, index=calendar, columns=symbols)
    entry_gate = (~available & fallback) | (
        available & (current_normal | emerging | quality_extension)
    )

    entry_score = features.ranking_score.copy()
    emerging_score = features.roc_short + 0.05 * r2.fillna(0.0)
    entry_score = entry_score.where(~emerging, emerging_score)

    soft_exit = pd.DataFrame(True, index=calendar, columns=symbols)
    missing_exit = fallback_cfg["soft_exit_protection"]
    missing_strong = (
        ~available
        & decision_rank.le(int(missing_exit["rank_limit"]))
        & features.roc_medium.ge(float(missing_exit["roc60_min"]))
        & features.above_ma
    )
    soft_exit &= ~missing_strong
    hot = live["hot_exit_protection"]
    hot_trigger = (
        available
        & sentiment["matched_count"].ge(float(hot["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(hot["hot_score_min"]))
        & sentiment["positive_dde_share"].ge(float(hot["positive_dde_share_min"]))
    )
    hot_memory = hot_trigger.rolling(
        int(hot["memory_days"]), min_periods=1
    ).max().fillna(False).astype(bool)
    if not component_flags["hot_exit_protection"]:
        hot_memory = pd.DataFrame(False, index=calendar, columns=symbols)
    soft_exit &= ~hot_memory

    bundle, _ = etfwin_signals(
        close,
        symbols,
        rules,
        raw_score_override=raw_score_override,
        entry_eligibility=eligibility,
        entry_gate=entry_gate,
        entry_ranking_score_override=entry_score,
        soft_exit_confirmation=soft_exit,
        dual_rank_decline_override=dual_rank_decline,
        reentry_cooldown_days=int(enhanced["reentry_cooldown_days"]),
    )
    decision = {
        "entry_gate": entry_gate,
        "normal": normal,
        "current_normal": current_normal,
        "fallback": fallback,
        "weak_edge_confirmed": edge_following,
        "emerging": emerging,
        "quality_extension": quality_extension,
        "soft_exit_confirmation": soft_exit,
        "hot_exit_protection": hot_memory,
        "missing_data_soft_exit_protection": missing_strong,
        "entry_score": entry_score,
        "entry_rank": decision_rank,
        "dual_rank_decline": dual_rank_decline,
        "core_symbols": core_symbols,
        "challenger_symbols": challenger_symbols,
        "universe_architecture": architecture_mode,
        "r2_20": r2,
        "efficiency20": efficiency,
        "category_breadth": breadth,
        "components": component_flags,
    }
    return bundle, features, eligibility, listed_sessions, trailing_amount, decision
