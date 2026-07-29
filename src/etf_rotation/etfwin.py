from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import SignalBundle


@dataclass(frozen=True)
class EtfwinRules:
    """Public ETFWin guide rules expressed as a causal daily strategy.

    The guide does not publish executable source code.  This implementation
    therefore keeps every public rule explicit and configurable, so a ye
    overlay can be compared with the guide proxy without replacing its core.
    """

    roc_short_days: int = 20
    roc_medium_days: int = 60
    roc_short_weight: float = 1.0
    roc_medium_weight: float = 1.5
    entry_rank_limit: int = 5
    ma_days: int = 120
    max_entry_ma_bias: float = 0.15
    rank_change_short_days: int = 5
    rank_change_long_days: int = 20
    exit_confirmation_days: int = 1
    ma_exit_confirmation_days: int = 1
    roc_exit_confirmation_days: int = 1
    holdings_num: int = 1
    require_short_roc_positive: bool = True
    require_medium_roc_positive: bool = True
    require_above_ma: bool = True
    exit_on_ma_break: bool = True
    exit_on_short_roc_negative: bool = True
    exit_on_dual_rank_decline: bool = True
    penalize_below_ma_for_ranking: bool = True

    def __post_init__(self) -> None:
        if self.roc_short_days < 1 or self.roc_medium_days <= self.roc_short_days:
            raise ValueError("ROC windows must be positive and ordered")
        if self.entry_rank_limit < 1 or self.ma_days < 2:
            raise ValueError("rank limit and moving-average window must be positive")
        if self.rank_change_short_days < 1 or self.rank_change_long_days <= self.rank_change_short_days:
            raise ValueError("rank-change windows must be positive and ordered")
        if self.max_entry_ma_bias < 0:
            raise ValueError("entry MA bias cannot be negative")
        if self.exit_confirmation_days < 1:
            raise ValueError("exit confirmation must be positive")
        if self.ma_exit_confirmation_days < 1:
            raise ValueError("MA exit confirmation must be positive")
        if self.roc_exit_confirmation_days < 1:
            raise ValueError("ROC exit confirmation must be positive")
        if self.holdings_num not in (1, 2):
            raise ValueError("only one or two holdings are supported")


@dataclass(frozen=True)
class EtfwinProfitProtection:
    """Volatility-scaled profit protection measured in ATR units."""

    activation_atr: float
    trail_atr: float

    def __post_init__(self) -> None:
        if self.activation_atr <= 0 or self.trail_atr <= 0:
            raise ValueError("profit-protection ATR multiples must be positive")


@dataclass(frozen=True)
class EtfwinOpportunitySwitch:
    """Allow one fully qualified holding to replace a weaker existing one."""

    held_rank_must_exceed: int
    minimum_score_advantage: float
    confirmation_days: int
    minimum_hold_days: int

    def __post_init__(self) -> None:
        if self.held_rank_must_exceed < 1:
            raise ValueError("opportunity-switch rank threshold must be positive")
        if self.minimum_score_advantage < 0:
            raise ValueError("opportunity-switch score advantage cannot be negative")
        if self.confirmation_days < 1:
            raise ValueError("opportunity-switch confirmation must be positive")
        if self.minimum_hold_days < 0:
            raise ValueError("opportunity-switch minimum hold cannot be negative")


@dataclass(frozen=True)
class EtfwinFeatures:
    roc_short: pd.DataFrame
    roc_medium: pd.DataFrame
    raw_score: pd.DataFrame
    ranking_score: pd.DataFrame
    rank: pd.DataFrame
    moving_average: pd.DataFrame
    ma_bias: pd.DataFrame
    above_ma: pd.DataFrame
    dual_rank_decline: pd.DataFrame


def etfwin_features(
    close: pd.DataFrame,
    rules: EtfwinRules,
    *,
    raw_score_override: pd.DataFrame | None = None,
) -> EtfwinFeatures:
    """Calculate the public guide's causal ROC, MA and rank fields."""

    roc_short = close.pct_change(rules.roc_short_days, fill_method=None)
    roc_medium = close.pct_change(rules.roc_medium_days, fill_method=None)
    if raw_score_override is None:
        raw_score = (
            roc_short * rules.roc_short_weight
            + roc_medium * rules.roc_medium_weight
        )
    else:
        missing = set(close.columns) - set(raw_score_override.columns)
        if missing:
            raise KeyError(f"score override is missing symbols: {sorted(missing)}")
        raw_score = raw_score_override.reindex(
            index=close.index, columns=close.columns
        ).astype(float)
    moving_average = close.rolling(rules.ma_days, min_periods=rules.ma_days).mean()
    ma_bias = close / moving_average - 1.0
    above_ma = close >= moving_average

    # ETFWin's public snapshot places every below-MA product behind products
    # above MA120.  Ten in decimal-return space is the documented -1000
    # percentage-point penalty for the public ROC score.  Research score
    # overrides may have a completely different scale (z-score, percentile or
    # annualised slope), so use a row-wise penalty large enough to preserve the
    # same group ordering without changing order inside either group.
    if rules.penalize_below_ma_for_ranking:
        if raw_score_override is None:
            penalty = pd.Series(10.0, index=raw_score.index)
        else:
            score_range = raw_score.max(axis=1) - raw_score.min(axis=1)
            penalty = score_range.fillna(0.0).abs() + 1.0
        ranking_score = raw_score - (~above_ma).mul(penalty, axis=0)
    else:
        ranking_score = raw_score.copy()
    ranking_score = ranking_score.where(raw_score.notna() & moving_average.notna())
    rank = ranking_score.rank(axis=1, ascending=False, method="min")
    dual_rank_decline = (rank > rank.shift(rules.rank_change_short_days)) & (
        rank > rank.shift(rules.rank_change_long_days)
    )
    return EtfwinFeatures(
        roc_short=roc_short,
        roc_medium=roc_medium,
        raw_score=raw_score,
        ranking_score=ranking_score,
        rank=rank,
        moving_average=moving_average,
        ma_bias=ma_bias,
        above_ma=above_ma,
        dual_rank_decline=dual_rank_decline.fillna(False),
    )


def etfwin_signals(
    close: pd.DataFrame,
    symbols: list[str],
    rules: EtfwinRules,
    *,
    entry_eligibility: pd.DataFrame | None = None,
    entry_gate: pd.DataFrame | None = None,
    raw_score_override: pd.DataFrame | None = None,
    entry_ranking_score_override: pd.DataFrame | None = None,
    soft_exit_confirmation: pd.DataFrame | None = None,
    dual_rank_decline_override: pd.DataFrame | None = None,
    emergency_exit: pd.DataFrame | None = None,
    atr: pd.DataFrame | None = None,
    profit_protection: EtfwinProfitProtection | None = None,
    profit_confirmation: pd.DataFrame | None = None,
    rank_only_entry_eligible: bool = False,
    reentry_cooldown_days: int = 0,
    priority_symbols: list[str] | None = None,
    preempt_for_priority_entry: bool = False,
    opportunity_switch: EtfwinOpportunitySwitch | None = None,
    opportunity_switch_rank: pd.DataFrame | None = None,
    opportunity_switch_score: pd.DataFrame | None = None,
    opportunity_switch_symbols: list[str] | None = None,
) -> tuple[SignalBundle, EtfwinFeatures]:
    """Run an ETFWin-core rotation strategy without look-ahead.

    MA120 breaks remain immediate because the guide calls them the strongest
    exit.  ``soft_exit_confirmation`` may only confirm the ROC20/rank-decline
    warnings; it cannot override the guide's entry rules or delay an MA break.

    ``rank_only_entry_eligible`` is an explicit research switch for a point-in-
    time tradable ranking universe. Entry ranks then exclude products that fail
    ``entry_eligibility``. Exit ranks retain a held product even if it later
    becomes ineligible, and compare it only with products that were eligible on
    each historical ranking date. The default preserves fixed-pool semantics.
    """

    if reentry_cooldown_days < 0:
        raise ValueError("reentry cooldown cannot be negative")
    missing = set(symbols) - set(close.columns)
    if missing:
        raise KeyError(f"missing close series: {sorted(missing)}")
    priority = set(priority_symbols or [])
    missing_priority = priority - set(symbols)
    if missing_priority:
        raise KeyError(f"priority symbols are missing: {sorted(missing_priority)}")
    if preempt_for_priority_entry and not priority:
        raise ValueError("priority symbols are required when preemption is enabled")
    switch_symbols = set(opportunity_switch_symbols or [])
    if opportunity_switch is not None:
        if rules.holdings_num != 1:
            raise ValueError("opportunity switching requires exactly one holding")
        if opportunity_switch_rank is None or opportunity_switch_score is None:
            raise ValueError("opportunity switching requires rank and score matrices")
        missing_switch_symbols = switch_symbols - set(symbols)
        if missing_switch_symbols:
            raise KeyError(
                f"opportunity-switch symbols are missing: {sorted(missing_switch_symbols)}"
            )
        if not switch_symbols:
            raise ValueError("opportunity switching requires an allowed symbol set")
    prices = close[symbols]
    features = etfwin_features(
        prices, rules, raw_score_override=raw_score_override
    )
    if entry_eligibility is None:
        entry_eligibility = pd.DataFrame(True, index=prices.index, columns=symbols)
    entry_eligibility = (
        entry_eligibility.reindex(index=prices.index, columns=symbols)
        .fillna(False)
        .astype(bool)
    )
    entry_ranking_score = features.ranking_score
    if entry_ranking_score_override is not None:
        missing_entry_scores = set(symbols) - set(
            entry_ranking_score_override.columns
        )
        if missing_entry_scores:
            raise KeyError(
                "entry ranking score override is missing symbols: "
                f"{sorted(missing_entry_scores)}"
            )
        entry_ranking_score = entry_ranking_score_override.reindex(
            index=prices.index, columns=symbols
        ).astype(float)
    entry_rank = entry_ranking_score.rank(
        axis=1, ascending=False, method="min"
    )
    dual_rank_decline = features.dual_rank_decline
    if rank_only_entry_eligible:
        entry_rank = entry_ranking_score.where(entry_eligibility).rank(
            axis=1, ascending=False, method="min"
        )
        # For every possible holding, count eligible products with a strictly
        # better score. This reproduces ``method='min'`` while retaining the
        # held product itself even when it no longer passes the entry gate.
        exit_rank = pd.DataFrame(
            np.nan, index=prices.index, columns=symbols, dtype=float
        )
        eligible_scores = features.ranking_score.where(entry_eligibility)
        for symbol in symbols:
            target = features.ranking_score[symbol]
            better = eligible_scores.gt(target, axis=0).sum(axis=1)
            exit_rank[symbol] = (1.0 + better).where(target.notna())
        dual_rank_decline = (
            exit_rank.gt(exit_rank.shift(rules.rank_change_short_days))
            & exit_rank.gt(exit_rank.shift(rules.rank_change_long_days))
        ).fillna(False)
    if dual_rank_decline_override is not None:
        missing_decline = set(symbols) - set(dual_rank_decline_override.columns)
        if missing_decline:
            raise KeyError(
                "dual-rank-decline override is missing symbols: "
                f"{sorted(missing_decline)}"
            )
        dual_rank_decline = (
            dual_rank_decline_override.reindex(index=prices.index, columns=symbols)
            .fillna(False)
            .astype(bool)
        )
    if entry_gate is None:
        entry_gate = pd.DataFrame(True, index=prices.index, columns=symbols)
    else:
        entry_gate = (
            entry_gate.reindex(index=prices.index, columns=symbols)
            .fillna(False)
            .astype(bool)
        )
    if opportunity_switch_rank is not None:
        opportunity_switch_rank = opportunity_switch_rank.reindex(
            index=prices.index, columns=symbols
        ).astype(float)
    if opportunity_switch_score is not None:
        opportunity_switch_score = opportunity_switch_score.reindex(
            index=prices.index, columns=symbols
        ).astype(float)
    if soft_exit_confirmation is None:
        soft_exit_confirmation = pd.DataFrame(
            True, index=prices.index, columns=symbols
        )
    else:
        soft_exit_confirmation = (
            soft_exit_confirmation.reindex(index=prices.index, columns=symbols)
            .fillna(False)
            .astype(bool)
        )
    if emergency_exit is None:
        emergency_exit = pd.DataFrame(False, index=prices.index, columns=symbols)
    else:
        emergency_exit = (
            emergency_exit.reindex(index=prices.index, columns=symbols)
            .fillna(False)
            .astype(bool)
        )
    if profit_protection is not None:
        if atr is None:
            raise ValueError("ATR is required when profit protection is enabled")
        atr = atr.reindex(index=prices.index, columns=symbols).astype(float)
        if profit_confirmation is None:
            profit_confirmation = pd.DataFrame(
                True, index=prices.index, columns=symbols
            )
        else:
            profit_confirmation = (
                profit_confirmation.reindex(index=prices.index, columns=symbols)
                .fillna(False)
                .astype(bool)
            )

    entry = (
        entry_eligibility
        & entry_gate
        & entry_rank.le(rules.entry_rank_limit)
    )
    if rules.require_short_roc_positive:
        entry &= features.roc_short.gt(0.0)
    if rules.require_medium_roc_positive:
        entry &= features.roc_medium.gt(0.0)
    if rules.require_above_ma:
        entry &= features.above_ma
    entry &= features.ma_bias.le(rules.max_entry_ma_bias)
    entry &= features.raw_score.notna()

    weights = pd.DataFrame(0.0, index=prices.index, columns=symbols)
    selected: list[str] = []
    roc_weak_streak = {symbol: 0 for symbol in symbols}
    rank_weak_streak = {symbol: 0 for symbol in symbols}
    ma_break_streak = {symbol: 0 for symbol in symbols}
    entry_price = {symbol: np.nan for symbol in symbols}
    entry_atr = {symbol: np.nan for symbol in symbols}
    highest_close = {symbol: np.nan for symbol in symbols}
    last_exit_location = {symbol: -10_000_000 for symbol in symbols}
    entered_location = {symbol: -10_000_000 for symbol in symbols}
    switch_streak_symbol: str | None = None
    switch_streak_count = 0
    rows: list[dict[str, object]] = []

    for location, date in enumerate(prices.index):
        exited: list[str] = []
        reasons: dict[str, str] = {}
        switch_candidate: str | None = None
        switch_score_gap = np.nan
        switch_qualified = False
        switch_triggered = False
        priority_available = any(
            bool(entry.loc[date, symbol])
            and location - last_exit_location[symbol] > reentry_cooldown_days
            for symbol in priority
        )
        for symbol in list(selected):
            price = float(prices.loc[date, symbol])
            if np.isfinite(price):
                highest_close[symbol] = (
                    price
                    if not np.isfinite(highest_close[symbol])
                    else max(highest_close[symbol], price)
                )
            ma_warning = bool(
                rules.exit_on_ma_break
                and not bool(features.above_ma.loc[date, symbol])
            )
            ma_break_streak[symbol] = (
                ma_break_streak[symbol] + 1 if ma_warning else 0
            )
            ma_break = bool(
                ma_break_streak[symbol] >= rules.ma_exit_confirmation_days
            )
            roc_warning = bool(
                rules.exit_on_short_roc_negative
                and features.roc_short.loc[date, symbol] < 0.0
            )
            rank_warning = bool(
                rules.exit_on_dual_rank_decline
                and dual_rank_decline.loc[date, symbol]
            )
            confirmed_by_indicator = bool(
                soft_exit_confirmation.loc[date, symbol]
            )
            roc_soft_warning = bool(roc_warning and confirmed_by_indicator)
            rank_soft_warning = bool(rank_warning and confirmed_by_indicator)
            roc_weak_streak[symbol] = (
                roc_weak_streak[symbol] + 1 if roc_soft_warning else 0
            )
            rank_weak_streak[symbol] = (
                rank_weak_streak[symbol] + 1 if rank_soft_warning else 0
            )
            roc_exit = bool(
                roc_weak_streak[symbol] >= rules.roc_exit_confirmation_days
            )
            rank_exit = bool(
                rank_weak_streak[symbol] >= rules.exit_confirmation_days
            )
            protection_hit = False
            if profit_protection is not None and np.isfinite(price):
                peak = highest_close[symbol]
                start = entry_price[symbol]
                initial_atr = entry_atr[symbol]
                current_atr = float(atr.loc[date, symbol])  # type: ignore[union-attr]
                protection_hit = bool(
                    np.isfinite(peak)
                    and np.isfinite(start)
                    and np.isfinite(initial_atr)
                    and initial_atr > 0
                    and np.isfinite(current_atr)
                    and current_atr > 0
                    and (peak - start) / initial_atr
                    >= profit_protection.activation_atr
                    and price <= peak - profit_protection.trail_atr * current_atr
                    and bool(profit_confirmation.loc[date, symbol])  # type: ignore[union-attr]
                )
            reason = None
            if preempt_for_priority_entry and symbol not in priority and priority_available:
                reason = "核心池出现合格候选"
            elif emergency_exit.loc[date, symbol]:
                reason = "附加紧急风险退出"
            elif ma_break:
                reason = "跌破MA120"
            elif protection_hit:
                reason = "ATR利润保护"
            elif roc_exit or rank_exit:
                active = []
                if roc_exit:
                    active.append("ROC20转负")
                if rank_exit:
                    active.append("5日与20日排名同时下滑")
                reason = "、".join(active)
            if reason is not None:
                selected.remove(symbol)
                last_exit_location[symbol] = location
                roc_weak_streak[symbol] = 0
                rank_weak_streak[symbol] = 0
                ma_break_streak[symbol] = 0
                entry_price[symbol] = np.nan
                entry_atr[symbol] = np.nan
                highest_close[symbol] = np.nan
                exited.append(symbol)
                reasons[symbol] = reason

        if exited:
            switch_streak_symbol = None
            switch_streak_count = 0

        if opportunity_switch is not None and selected:
            held = selected[0]
            if held in switch_symbols:
                candidates = (
                    entry_ranking_score.loc[date]
                    .where(entry.loc[date])
                    .dropna()
                    .loc[lambda row: row.index.isin(switch_symbols)]
                    .drop(index=held, errors="ignore")
                    .sort_values(ascending=False)
                )
                candidates = candidates.loc[
                    [
                        symbol
                        for symbol in candidates.index
                        if location - last_exit_location[str(symbol)]
                        > reentry_cooldown_days
                    ]
                ]
                if not candidates.empty:
                    switch_candidate = str(candidates.index[0])
                    switch_score_gap = float(
                        opportunity_switch_score.loc[date, switch_candidate]
                        - opportunity_switch_score.loc[date, held]
                    )
                    switch_qualified = bool(
                        opportunity_switch_rank.loc[date, held]
                        > opportunity_switch.held_rank_must_exceed
                        and switch_score_gap
                        >= opportunity_switch.minimum_score_advantage
                        and location - entered_location[held]
                        >= opportunity_switch.minimum_hold_days
                    )
            if switch_qualified and switch_candidate is not None:
                if switch_streak_symbol == switch_candidate:
                    switch_streak_count += 1
                else:
                    switch_streak_symbol = switch_candidate
                    switch_streak_count = 1
            else:
                switch_streak_symbol = None
                switch_streak_count = 0

            if (
                switch_qualified
                and switch_candidate is not None
                and switch_streak_count >= opportunity_switch.confirmation_days
            ):
                old = held
                selected.remove(old)
                last_exit_location[old] = location
                entered_location[old] = -10_000_000
                roc_weak_streak[old] = 0
                rank_weak_streak[old] = 0
                ma_break_streak[old] = 0
                entry_price[old] = np.nan
                entry_atr[old] = np.nan
                highest_close[old] = np.nan
                exited.append(old)
                reasons[old] = "机会成本换仓"
                switch_triggered = True
                switch_streak_symbol = None
                switch_streak_count = 0

        entered: list[str] = []
        if len(selected) < rules.holdings_num:
            candidates = (
                entry_ranking_score.loc[date]
                .where(entry.loc[date])
                .dropna()
                .sort_values(ascending=False)
            )
            if priority_available:
                candidates = candidates.loc[candidates.index.isin(priority)]
            for symbol in candidates.index:
                symbol = str(symbol)
                if symbol in selected or symbol in exited:
                    continue
                if location - last_exit_location[symbol] <= reentry_cooldown_days:
                    continue
                selected.append(symbol)
                entered_location[symbol] = location
                roc_weak_streak[symbol] = 0
                rank_weak_streak[symbol] = 0
                ma_break_streak[symbol] = 0
                entry_price[symbol] = float(prices.loc[date, symbol])
                highest_close[symbol] = entry_price[symbol]
                entry_atr[symbol] = (
                    float(atr.loc[date, symbol])
                    if atr is not None
                    else np.nan
                )
                entered.append(symbol)
                if len(selected) >= rules.holdings_num:
                    break

        if selected:
            weights.loc[date, selected] = 1.0 / len(selected)
            mode = "offensive"
        else:
            mode = "cash"
        rows.append(
            {
                "date": date,
                "mode": mode,
                "selected": "|".join(selected),
                "entered": "|".join(entered),
                "exited": "|".join(exited),
                "exit_reasons": "|".join(
                    f"{symbol}:{reason}" for symbol, reason in reasons.items()
                ),
                "entry_eligible_count": int(entry.loc[date].sum()),
                "priority_entry_available": bool(priority_available),
                "holding_count": len(selected),
                "opportunity_switch_candidate": switch_candidate or "",
                "opportunity_switch_score_gap": switch_score_gap,
                "opportunity_switch_qualified": bool(switch_qualified),
                "opportunity_switch_streak": int(switch_streak_count),
                "opportunity_switch_triggered": bool(switch_triggered),
            }
        )

    diagnostics = pd.DataFrame(rows).set_index("date")
    exposure = weights.sum(axis=1)
    diagnostics["exposure"] = exposure
    return SignalBundle(weights, exposure, diagnostics), features
