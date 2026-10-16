"""V3: pool-size-invariant absolute-conviction universe research.

This is an isolated research script.  It does not modify the formal strategy or
produce live orders.  The threshold is calibrated only to match the formal
core-45 candidate-day frequency during 2018-07-02..2020-12-31; returns never
enter calibration.  All dates from 2021-01-01 onward are out-of-sample.

Run:
    PYTHONPATH=src python3 experiments/universe_architecture_v3_conviction.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.etfwin import etfwin_signals
from etf_rotation.execution import entry_eligibility, execution_project, period_metrics
from etf_rotation.sentiment import broadcast, load_sentiment_matrices
from etf_rotation.ye import _rules, rolling_r2

from universe_architecture_v2_study import (
    FEATURES,
    cash_management,
    load_yaml,
    premium_sensitive,
    run_variant,
)


OUTPUT = ROOT / "results" / "research" / "universe_architecture_v3"
CALIBRATION_START = "2018-07-02"
CALIBRATION_END = "2020-12-31"
OOS_START = "2021-01-01"
PERIODS = {
    "calibration_2018_2020": (CALIBRATION_START, CALIBRATION_END),
    "oos_2021_2022": ("2021-01-01", "2022-12-31"),
    "oos_2023_2024": ("2023-01-01", "2024-12-31"),
    "oos_2025_2026": ("2025-01-01", "2026-07-27"),
    "oos_2021_2026": (OOS_START, "2026-07-27"),
}

# Declared before observing V3 results.  The primary is the Bernstein-inspired
# parsimonious version; other variants are ablations, not a return search.
SCORE_VARIANTS = {
    "momentum": {"momentum": 1.00, "r2": 0.00, "efficiency": 0.00, "low_vol": 0.00, "crowding": 0.00},
    "momentum_quality": {"momentum": 0.50, "r2": 0.25, "efficiency": 0.25, "low_vol": 0.00, "crowding": 0.00},
    "momentum_quality_crowding": {"momentum": 0.50, "r2": 0.25, "efficiency": 0.25, "low_vol": 0.00, "crowding": 0.10},
    "balanced_risk": {"momentum": 0.45, "r2": 0.20, "efficiency": 0.20, "low_vol": 0.15, "crowding": 0.10},
}
PRIMARY = "momentum_quality_crowding_absolute_exit"


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def trailing_percentile(frame: pd.DataFrame, window: int = 252, minimum: int = 120) -> pd.DataFrame:
    """Percentile of today's value within each ETF's own trailing history."""

    return frame.rolling(window, min_periods=minimum).rank(method="average", pct=True)


def build_components(panel: dict[str, pd.DataFrame], symbols: list[str], formal: dict) -> dict[str, pd.DataFrame]:
    close = panel["close"][symbols]
    rules = _rules(formal["rules"])
    from etf_rotation.etfwin import etfwin_features

    features = etfwin_features(close, rules)
    returns = close.pct_change(fill_method=None)
    r2 = rolling_r2(close, 20)
    efficiency = close.pct_change(20, fill_method=None).abs() / returns.abs().rolling(20).sum()
    vol20 = returns.rolling(20, min_periods=20).std()
    roc5 = close.pct_change(5, fill_method=None)

    momentum_p = trailing_percentile(features.ranking_score)
    r2_p = trailing_percentile(r2)
    efficiency_p = trailing_percentile(efficiency)
    low_vol_p = 1.0 - trailing_percentile(vol20)
    bias_p = trailing_percentile(features.ma_bias)
    roc5_p = trailing_percentile(roc5)

    # Penalise only the most crowded quartile / decile.  This is deliberately
    # piecewise and small; the existing 9%/12% hard caps remain the main guard.
    bias_crowding = ((bias_p - 0.75) / 0.25).clip(lower=0.0, upper=1.0)
    acceleration_crowding = ((roc5_p - 0.90) / 0.10).clip(lower=0.0, upper=1.0)
    crowding = 0.60 * bias_crowding + 0.40 * acceleration_crowding

    return {
        "features": features,
        "r2": r2,
        "efficiency": efficiency,
        "roc5": roc5,
        "momentum": momentum_p,
        "r2_percentile": r2_p,
        "efficiency_percentile": efficiency_p,
        "low_vol_percentile": low_vol_p,
        "crowding": crowding,
    }


def conviction_score(components: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    score = (
        weights["momentum"] * components["momentum"]
        + weights["r2"] * components["r2_percentile"]
        + weights["efficiency"] * components["efficiency_percentile"]
        + weights["low_vol"] * components["low_vol_percentile"]
        - weights["crowding"] * components["crowding"]
    )
    return score.clip(lower=-0.25, upper=1.0)


def rank_free_paths(
    components: dict[str, pd.DataFrame],
    eligibility: pd.DataFrame,
    formal: dict,
    sentiment: dict[str, pd.DataFrame],
    sentiment_available: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Rebuild the existing price/AI entry paths without cross-sectional ranks."""

    features = components["features"]
    r2 = components["r2"]
    efficiency = components["efficiency"]
    roc5 = components["roc5"]
    symbols = list(eligibility.columns)
    live = formal["enhanced_selection"]["sentiment_available"]
    values = formal["rules"]
    available = broadcast(sentiment_available, symbols)

    normal = (
        features.roc_short.gt(0.0)
        & features.roc_medium.gt(0.0)
        & features.above_ma
        & features.ma_bias.le(float(values["max_entry_ma_bias"]))
    )
    # In V3, a weak ROC20 is weak regardless of how many other ETFs happen to
    # be in the pool.  News confirmation therefore no longer depends on rank 4-5.
    weak_edge = features.roc_short.lt(0.02)
    edge_following = (
        sentiment["matched_count"].ge(3)
        & sentiment["count_acceleration"].ge(0.0)
        & sentiment["positive_dde_share"].ge(0.50)
    )
    current_normal = normal & (~weak_edge | edge_following)

    emerging_cfg = live["emerging_trend"]
    emerging_trigger = (
        features.roc_short.ge(float(emerging_cfg["roc20_min"]))
        & features.roc_medium.ge(float(emerging_cfg["roc60_range"][0]))
        & features.roc_medium.le(float(emerging_cfg["roc60_range"][1]))
        & features.ma_bias.ge(float(emerging_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(emerging_cfg["ma120_bias_range"][1]))
        & r2.ge(float(emerging_cfg["r2_20_min"]))
        & efficiency.ge(float(emerging_cfg["efficiency20_min"]))
        & sentiment["matched_count"].ge(float(emerging_cfg["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(emerging_cfg["hot_score_min"]))
        & sentiment["count_acceleration"].ge(float(emerging_cfg["count_acceleration_min"]))
        & sentiment["positive_dde_share"].ge(float(emerging_cfg["positive_dde_share_min"]))
    )
    emerging = emerging_trigger.rolling(int(emerging_cfg["memory_days"]), min_periods=1).max().fillna(False).astype(bool)
    emerging &= (
        features.roc_short.gt(0.0)
        & features.roc_medium.ge(float(emerging_cfg["roc60_range"][0]))
        & features.ma_bias.ge(float(emerging_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(live["quality_extension"]["ma120_bias_range"][1]))
    )

    extension_cfg = live["quality_extension"]
    extension = (
        features.roc_short.gt(0.0)
        & features.roc_medium.gt(0.0)
        & features.above_ma
        & features.ma_bias.gt(float(extension_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(extension_cfg["ma120_bias_range"][1]))
        & r2.ge(float(extension_cfg["r2_20_min"]))
        & efficiency.ge(float(extension_cfg["efficiency20_min"]))
        & roc5.ge(float(extension_cfg["roc5_min"]))
        & sentiment["matched_count"].ge(float(extension_cfg["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(extension_cfg["hot_score_min"]))
        & sentiment["count_acceleration"].ge(float(extension_cfg["count_acceleration_min"]))
    )

    # Before sentiment starts, absolute conviction replaces both rank<=3 and
    # same-category breadth.  A new one-member theme cannot self-confirm.
    pre_gate = eligibility & ((~available & normal) | (available & (current_normal | emerging | extension)))
    return {
        "pre_gate": pre_gate.astype(bool),
        "normal": normal.astype(bool),
        "current_normal": current_normal.astype(bool),
        "emerging": emerging.astype(bool),
        "quality_extension": extension.astype(bool),
        "available": available.astype(bool),
    }


def calibrate_threshold(
    score: pd.DataFrame,
    pre_gate: pd.DataFrame,
    target_candidate_rate: float,
    start: str = CALIBRATION_START,
    end: str = CALIBRATION_END,
) -> tuple[float, float]:
    """Match candidate-day frequency; never inspect strategy returns."""

    maximum = score.where(pre_gate).max(axis=1).loc[start:end]
    valid = np.sort(maximum.dropna().unique())
    if len(valid) == 0:
        raise ValueError("no valid calibration observations")
    # Every distinct observed maximum is a possible boundary.  This works for
    # both 0..1 percentiles and raw momentum returns without a hand-picked grid.
    grid = valid
    rates = np.array([(maximum.ge(level)).mean() for level in grid])
    distance = np.abs(rates - target_candidate_rate)
    best_distance = distance.min()
    # Conservative tie-break: choose the highest threshold.
    threshold = float(grid[np.where(np.isclose(distance, best_distance))[0][-1]])
    rate = float((maximum >= threshold).mean())
    return threshold, rate


def theme_champions(gate: pd.DataFrame, score: pd.DataFrame, categories: dict[str, str]) -> pd.DataFrame:
    """Keep one qualified representative per frozen theme after the hurdle."""

    output = pd.DataFrame(False, index=gate.index, columns=gate.columns)
    groups: dict[str, list[str]] = {}
    for symbol in gate.columns:
        groups.setdefault(categories[symbol], []).append(symbol)
    for members in groups.values():
        values = score[members].where(gate[members])
        best = values.fillna(-np.inf).idxmax(axis=1)
        valid = values.notna().any(axis=1)
        for symbol in members:
            output[symbol] = gate[symbol] & valid & best.eq(symbol)
    return output.astype(bool)


def build_soft_exit(
    components: dict[str, pd.DataFrame],
    score: pd.DataFrame,
    threshold: float,
    formal: dict,
    sentiment: dict[str, pd.DataFrame],
    sentiment_available: pd.Series,
) -> pd.DataFrame:
    features = components["features"]
    symbols = list(score.columns)
    available = broadcast(sentiment_available, symbols)
    fallback = formal["enhanced_selection"]["price_only_for_historical_dates_without_sentiment"]["soft_exit_protection"]
    strong = (
        ~available
        & score.ge(threshold + 0.05)
        & features.roc_medium.ge(float(fallback["roc60_min"]))
        & features.above_ma
    )
    hot = formal["enhanced_selection"]["sentiment_available"]["hot_exit_protection"]
    hot_trigger = (
        available
        & sentiment["matched_count"].ge(float(hot["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(hot["hot_score_min"]))
        & sentiment["positive_dde_share"].ge(float(hot["positive_dde_share_min"]))
    )
    hot_memory = hot_trigger.rolling(int(hot["memory_days"]), min_periods=1).max().fillna(False).astype(bool)
    return (~strong & ~hot_memory).astype(bool)


def run_absolute_variant(
    name: str,
    market: dict,
    formal: dict,
    panel: dict[str, pd.DataFrame],
    symbols: list[str],
    categories: dict[str, str],
    sentiment: dict[str, pd.DataFrame],
    sentiment_available: pd.Series,
    score: pd.DataFrame,
    pre_gate: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    threshold: float,
    *,
    absolute_exit: bool,
) -> dict:
    eligibility, _, _ = entry_eligibility(panel, symbols, formal["rules"])
    entry_gate = theme_champions(pre_gate & score.ge(threshold), score, categories)
    if absolute_exit:
        # Pool-independent analogue of "rank worse than both 5 and 20 days":
        # conviction must be below the entry hurdle and below both past values.
        decline = (
            score.lt(threshold)
            & score.lt(score.shift(int(formal["rules"]["rank_change_short_days"])))
            & score.lt(score.shift(int(formal["rules"]["rank_change_long_days"])))
        ).fillna(False)
    else:
        # Ablation only: retain cross-sectional exit ranking to isolate entry effects.
        rank = components["features"].ranking_score.rank(axis=1, ascending=False, method="min")
        decline = (
            rank.gt(rank.shift(int(formal["rules"]["rank_change_short_days"])))
            & rank.gt(rank.shift(int(formal["rules"]["rank_change_long_days"])))
        ).fillna(False)
    soft_exit = build_soft_exit(components, score, threshold, formal, sentiment, sentiment_available)
    bundle, _ = etfwin_signals(
        panel["close"][symbols],
        symbols,
        _rules(formal["rules"]),
        entry_eligibility=eligibility,
        entry_gate=entry_gate,
        entry_ranking_score_override=score,
        soft_exit_confirmation=soft_exit,
        dual_rank_decline_override=decline,
        reentry_cooldown_days=int(formal["enhanced_selection"]["reentry_cooldown_days"]),
    )
    project = execution_project(
        market,
        premium_sensitive(symbols),
        eligibility.shift(1, fill_value=False).astype(bool),
    )
    result = run_backtest(
        name,
        panel,
        bundle.weights,
        str(market["project"]["backtest_start"]),
        str(market["project"]["data_end"]),
        project,
        cash_management=cash_management(formal),
    )
    capital = float(market["project"]["initial_capital"])
    periods = {key: period_metrics(result.equity, start, end, capital) for key, (start, end) in PERIODS.items()}
    return {
        "name": name,
        "threshold": threshold,
        "result": result,
        "periods": periods,
        "weights": bundle.weights,
        "diagnostics": bundle.diagnostics,
        "entry_gate": entry_gate,
        "pre_gate": pre_gate,
        "score": score,
        "absolute_exit": absolute_exit,
    }


def metric_row(variant: dict) -> dict:
    row = {"variant": variant["name"], "threshold": variant["threshold"], **variant["result"].metrics}
    for key, metrics in variant["periods"].items():
        row[f"return_{key}"] = metrics["total_return"]
        row[f"mdd_{key}"] = metrics["max_drawdown"]
    row["candidate_day_rate_calibration"] = float(variant["entry_gate"].loc[CALIBRATION_START:CALIBRATION_END].any(axis=1).mean())
    row["candidate_day_rate_oos"] = float(variant["entry_gate"].loc[OOS_START:].any(axis=1).mean())
    return row


def active_symbol(weights: pd.DataFrame) -> pd.Series:
    return weights.fillna(0.0).idxmax(axis=1).where(weights.fillna(0.0).max(axis=1).gt(0.0), "CASH")


def compare_paths(left: dict, right_weights: pd.DataFrame) -> dict:
    symbols = sorted(set(left["weights"].columns) | set(right_weights.columns))
    l = left["weights"].reindex(columns=symbols, fill_value=0.0).loc[CALIBRATION_START:]
    r = right_weights.reindex(index=l.index, columns=symbols, fill_value=0.0)
    difference = ~np.isclose(l.to_numpy(), r.to_numpy(), atol=1e-12).all(axis=1)
    return {"different_signal_days": int(difference.sum()), "same_signal_path": bool(not difference.any())}


def bootstrap_return_difference(primary, formal, start=OOS_START, seed=20260728, samples=4000, block=20):
    """Moving-block bootstrap of OOS daily mean-return difference (diagnostic only)."""

    p = primary["result"].daily_returns.loc[start:]
    f = formal["result"].daily_returns.loc[start:]
    diff = (p - f).dropna().to_numpy()
    n = len(diff)
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, n - block + 1))
    draws = []
    blocks_needed = int(np.ceil(n / block))
    for _ in range(samples):
        picked = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([diff[i:i + block] for i in picked])[:n]
        draws.append(sample.mean() * 252.0)
    draws = np.asarray(draws)
    return {
        "annualized_mean_daily_return_difference": float(diff.mean() * 252.0),
        "ci_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float((draws > 0).mean()),
        "samples": samples,
        "block_days": block,
        "interpretation_limit": "diagnostic only; path-dependent strategy selection and multiple variants are not removed",
    }


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    formal = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    all_symbols = universe_keys(market)
    core_size = int(formal["enhanced_selection"]["universe_architecture"]["core_pool_size"])
    core_symbols = all_symbols[:core_size]
    challengers = [str(s) for s in formal["enhanced_selection"]["universe_architecture"]["challenger_symbols"]]
    categories_all = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment_all, available = load_sentiment_matrices(FEATURES, panel["close"].index, all_symbols)

    def formal_variant(key, symbols, mode, challenger_symbols):
        value = run_variant(
            key, market, formal, panel, all_symbols, core_symbols, categories_all,
            sentiment_all, available, symbols=symbols, mode=mode,
            challengers=challenger_symbols,
        )
        return {
            "name": key,
            "threshold": np.nan,
            "result": type("ResultView", (), {
                "metrics": value["metrics"], "equity": value["equity"],
                "daily_returns": value["equity"].pct_change().fillna(0.0),
                "trades": value["trades"],
            })(),
            "periods": {
                name: period_metrics(value["equity"], start, end, float(market["project"]["initial_capital"]))
                for name, (start, end) in PERIODS.items()
            },
            "weights": value["weights"],
            "entry_gate": value["decision"]["entry_gate"],
        }

    core45 = formal_variant("baseline_core_45", core_symbols, "core_anchor_challenger", [])
    formal_f = formal_variant("formal_champion_cash_gap_51", all_symbols, "core_champion_cash_gap", challengers)
    target_rate = float(core45["entry_gate"].loc[CALIBRATION_START:CALIBRATION_END].any(axis=1).mean())

    components = build_components(panel, all_symbols, formal)
    eligibility, _, _ = entry_eligibility(panel, all_symbols, formal["rules"])
    paths = rank_free_paths(components, eligibility, formal, sentiment_all, available)
    variants: dict[str, dict] = {}
    calibration: dict[str, dict] = {}
    scores: dict[str, pd.DataFrame] = {}

    for score_name, weights in SCORE_VARIANTS.items():
        score = conviction_score(components, weights)
        scores[score_name] = score
        threshold, achieved_rate = calibrate_threshold(score, paths["pre_gate"], target_rate)
        calibration[score_name] = {
            "threshold": threshold,
            "target_candidate_day_rate": target_rate,
            "achieved_candidate_day_rate_before_theme_dedup": achieved_rate,
            "weights": weights,
        }
        for absolute_exit in (True, False):
            suffix = "absolute_exit" if absolute_exit else "relative_exit_ablation"
            key = f"{score_name}_{suffix}"
            variants[key] = run_absolute_variant(
                key, market, formal, panel, all_symbols, categories_all,
                sentiment_all, available, score, paths["pre_gate"], components,
                threshold, absolute_exit=absolute_exit,
            )

    # V3b was formulated only after V3a exposed that an ETF can be "strong
    # versus itself" while still having weak absolute momentum.  It is reported
    # as post-diagnostic evidence, never as a preregistered winner.  Raw momentum
    # remains cross-ETF comparable; quality and crowding are separate risk gates.
    raw_score = components["features"].ranking_score.copy()
    quality = 0.5 * components["r2_percentile"] + 0.5 * components["efficiency_percentile"]
    raw_specs = {
        "raw_momentum": paths["pre_gate"],
        "raw_momentum_quality_floor": paths["pre_gate"] & quality.ge(0.50),
        "raw_momentum_quality_crowding_gate": (
            paths["pre_gate"] & quality.ge(0.50) & components["crowding"].le(0.75)
        ),
    }
    raw_thresholds: dict[str, float] = {}
    for score_name, spec_gate in raw_specs.items():
        threshold, achieved_rate = calibrate_threshold(raw_score, spec_gate, target_rate)
        raw_thresholds[score_name] = threshold
        calibration[score_name] = {
            "threshold": threshold,
            "target_candidate_day_rate": target_rate,
            "achieved_candidate_day_rate_before_theme_dedup": achieved_rate,
            "status": "post_diagnostic_v3b",
            "selection_score": "raw ROC20 + 1.5 * ROC60",
            "quality_floor": 0.50 if "quality" in score_name else None,
            "crowding_ceiling": 0.75 if "crowding" in score_name else None,
        }
        for absolute_exit in (True, False):
            suffix = "absolute_exit" if absolute_exit else "relative_exit_ablation"
            key = f"{score_name}_{suffix}"
            variants[key] = run_absolute_variant(
                key, market, formal, panel, all_symbols, categories_all,
                sentiment_all, available, raw_score, spec_gate, components,
                threshold, absolute_exit=absolute_exit,
            )

    primary = variants[PRIMARY]
    metrics = pd.DataFrame([metric_row(v) for v in variants.values()])

    # Threshold sensitivity is centered on the predeclared primary and never
    # used to replace it with the highest-return threshold.
    primary_base = calibration["momentum_quality_crowding"]["threshold"]
    sensitivity_rows = []
    sensitivity_variants = []
    for delta in (-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10):
        threshold = float(np.clip(primary_base + delta, 0.20, 0.95))
        v = run_absolute_variant(
            f"primary_threshold_{threshold:.3f}", market, formal, panel,
            all_symbols, categories_all, sentiment_all, available,
            scores["momentum_quality_crowding"], paths["pre_gate"], components,
            threshold, absolute_exit=True,
        )
        sensitivity_variants.append(v)
        sensitivity_rows.append(metric_row(v))

    raw_sensitivity_rows = []
    raw_primary_name = "raw_momentum_quality_crowding_gate"
    raw_primary_gate = raw_specs[raw_primary_name]
    raw_base = raw_thresholds[raw_primary_name]
    for multiple in (0.75, 0.875, 0.95, 1.0, 1.05, 1.125, 1.25):
        threshold = float(raw_base * multiple)
        v = run_absolute_variant(
            f"raw_primary_threshold_x{multiple:.3f}", market, formal, panel,
            all_symbols, categories_all, sentiment_all, available,
            raw_score, raw_primary_gate, components, threshold, absolute_exit=True,
        )
        row = metric_row(v)
        row["threshold_multiple"] = multiple
        raw_sensitivity_rows.append(row)

    # Structural invariants: score independence, order invariance, ineligible
    # decoy invariance, and add-one challenger behaviour.
    core_components = build_components(panel, core_symbols, formal)
    core_score = conviction_score(core_components, SCORE_VARIANTS["momentum_quality_crowding"])
    common_diff = (core_score - scores["momentum_quality_crowding"][core_symbols]).abs().max().max()

    reversed_symbols = list(reversed(all_symbols))
    reordered = run_absolute_variant(
        "primary_reordered", market, formal, panel, reversed_symbols,
        {s: categories_all[s] for s in reversed_symbols},
        {k: v[reversed_symbols] for k, v in sentiment_all.items()}, available,
        scores["momentum_quality_crowding"][reversed_symbols],
        paths["pre_gate"][reversed_symbols],
        {k: (v[reversed_symbols] if isinstance(v, pd.DataFrame) else v) for k, v in components.items()},
        primary_base, absolute_exit=True,
    )

    add_one_rows = []
    for challenger in challengers:
        subset = core_symbols + [challenger]
        subset_variant = run_absolute_variant(
            f"primary_add_{challenger}", market, formal, panel, subset,
            {s: categories_all[s] for s in subset},
            {k: v[subset] for k, v in sentiment_all.items()}, available,
            scores["momentum_quality_crowding"][subset], paths["pre_gate"][subset],
            {k: (v[subset] if isinstance(v, pd.DataFrame) else v) for k, v in components.items()},
            primary_base, absolute_exit=True,
        )
        add_one_rows.append({
            "symbol": challenger,
            "total_return": subset_variant["result"].metrics["total_return"],
            "oos_return_2021_2026": subset_variant["periods"]["oos_2021_2026"]["total_return"],
            "signal_diff_days_vs_full51": compare_paths(subset_variant, primary["weights"])["different_signal_days"],
            "holding_days": int(active_symbol(subset_variant["weights"]).loc[CALIBRATION_START:].eq(challenger).sum()),
        })

    active = active_symbol(primary["weights"]).loc[CALIBRATION_START:]
    active_formal = active_symbol(formal_f["weights"]).loc[CALIBRATION_START:]
    path_daily = pd.DataFrame({
        "primary": active,
        "formal_f": active_formal.reindex(active.index),
    })
    path_daily["different"] = path_daily["primary"].ne(path_daily["formal_f"])
    challenger_holding_days = {symbol: int(active.eq(symbol).sum()) for symbol in challengers}

    # Synthetic never-eligible decoy: it may have any numerical score, but a
    # false pre-gate must leave every existing post-hurdle gate unchanged.
    decoy = "DECOY.NA"
    decoy_score = primary["score"].copy()
    decoy_score[decoy] = primary["score"][all_symbols[0]]
    decoy_gate = paths["pre_gate"].copy()
    decoy_gate[decoy] = False
    decoy_categories = {**categories_all, decoy: "synthetic_decoy"}
    decoy_effective = theme_champions(
        decoy_gate & decoy_score.ge(primary_base), decoy_score, decoy_categories
    )
    existing_gate_unchanged = bool(
        decoy_effective[all_symbols].equals(primary["entry_gate"][all_symbols])
    )

    bootstrap = bootstrap_return_difference(primary, formal_f)
    invariants = {
        "overlapping_core_score_max_abs_difference": float(common_diff),
        "overlapping_core_scores_identical": bool(common_diff < 1e-12),
        "column_order": compare_paths(reordered, primary["weights"]),
        "never_eligible_decoy_entry_count": int(decoy_effective[decoy].sum()),
        "never_eligible_decoy_leaves_existing_entry_gate_unchanged": existing_gate_unchanged,
        "challenger_holding_days": challenger_holding_days,
        "all_score_formulas_are_per_symbol_only": True,
        "theme_dedup_runs_after_absolute_hurdle": True,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    benchmark_rows = []
    for benchmark in (core45, formal_f):
        row = {"variant": benchmark["name"], **benchmark["result"].metrics}
        for key, values in benchmark["periods"].items():
            row[f"return_{key}"] = values["total_return"]
            row[f"mdd_{key}"] = values["max_drawdown"]
        benchmark_rows.append(row)
    pd.DataFrame(benchmark_rows).to_csv(OUTPUT / "benchmarks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity_rows).to_csv(OUTPUT / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(raw_sensitivity_rows).to_csv(OUTPUT / "raw_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(add_one_rows).to_csv(OUTPUT / "add_one_challenger.csv", index=False, encoding="utf-8-sig")
    path_daily.loc[path_daily["different"]].to_csv(OUTPUT / "path_differences_vs_formal.csv", encoding="utf-8-sig")
    primary["score"].to_csv(
        OUTPUT / "primary_conviction_scores.csv.gz", compression="gzip", encoding="utf-8-sig"
    )
    primary["entry_gate"].astype(int).to_csv(
        OUTPUT / "primary_entry_gate.csv.gz", compression="gzip", encoding="utf-8-sig"
    )

    payload = {
        "status": "research_evidence_v3",
        "generated_through": str(market["project"]["data_end"]),
        "formal_strategy_unchanged": True,
        "primary_predeclared": PRIMARY,
        "calibration": clean(calibration),
        "metrics": clean(metrics.to_dict(orient="records")),
        "threshold_sensitivity": clean(sensitivity_rows),
        "raw_threshold_sensitivity": clean(raw_sensitivity_rows),
        "benchmarks": clean(benchmark_rows),
        "invariants": clean(invariants),
        "bootstrap_vs_formal_oos": clean(bootstrap),
        "path_difference_days_vs_formal": int(path_daily["different"].sum()),
        "best_ex_post_threshold_sensitivity": clean(
            max(sensitivity_rows, key=lambda row: row["total_return"])
        ),
        "limitations": [
            "current-survivor ETF universe; historical delisted products are unavailable",
            "true valuation and constituent earnings-revision data are unavailable; price/volume proxies are not a Bernstein replication",
            "sentiment features begin in 2024, so the historical data regime changes",
            "all dates through 2020 calibrate frequency; all dates from 2021 are OOS for the threshold but not untouched by prior project research",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"target candidate rate={target_rate:.4f}; primary={PRIMARY}; threshold={primary_base:.3f}")
    print(metrics[["variant", "threshold", "total_return", "cagr", "max_drawdown", "sharpe", "return_oos_2021_2026"]].to_string(index=False))
    print("invariants", json.dumps(clean(invariants), ensure_ascii=False))
    print("bootstrap", json.dumps(clean(bootstrap), ensure_ascii=False))


if __name__ == "__main__":
    main()
