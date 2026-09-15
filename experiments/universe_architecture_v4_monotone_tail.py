"""V4 research: monotone relative-strength tail quota.

The fixed top-5 rule is replaced with top 5/45 of the effective comparison
universe.  Directionally invalid ETFs receive no competitive score, so they
cannot displace a valid trend.  Weak additions can only leave an existing
candidate unchanged or expand the tail quota; genuinely stronger additions may
compete.  One representative per frozen theme is used by the primary variant.

Research only.  Formal configuration and live orders are never modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.etfwin import etfwin_signals
from etf_rotation.execution import entry_eligibility, execution_project, period_metrics
from etf_rotation.sentiment import broadcast, load_sentiment_matrices
from etf_rotation.ye import _rules, category_breadth

from universe_architecture_v2_study import (
    FEATURES,
    cash_management,
    load_yaml,
    premium_sensitive,
    run_variant,
)
from universe_architecture_v3_conviction import (
    build_components,
    theme_champions,
)


OUTPUT = ROOT / "results" / "research" / "universe_architecture_v4"
START = "2018-07-02"
END = "2026-07-27"
PRIMARY = "theme_monotone_tail_relative_exit"
SHARES = {"fallback": 3 / 45, "normal": 5 / 45, "emerging": 15 / 45}
PERIODS = {
    "2018_2020": (START, "2020-12-31"),
    "2021_2022": ("2021-01-01", "2022-12-31"),
    "2023_2024": ("2023-01-01", "2024-12-31"),
    "2025_2026": ("2025-01-01", END),
}


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


def comparison_ruler(
    raw_score: pd.DataFrame,
    roc20: pd.DataFrame,
    roc60: pd.DataFrame,
    eligibility: pd.DataFrame,
    categories: dict[str, str],
    *,
    theme_level: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return representative scores, every ETF's virtual rank, and ruler size.

    Listed/liquid ETFs determine ruler breadth.  Only dual-positive momentum
    receives a finite competitive score.  Thus invalid additions cannot occupy
    a strong slot, while a genuinely stronger valid addition may compete.
    """

    directional = eligibility & roc20.gt(0.0) & roc60.gt(0.0) & raw_score.notna()
    competitive = raw_score.where(directional, -np.inf)
    representative = pd.DataFrame(False, index=raw_score.index, columns=raw_score.columns)

    if theme_level:
        groups: dict[str, list[str]] = {}
        for symbol in raw_score.columns:
            groups.setdefault(categories[symbol], []).append(symbol)
        for members in groups.values():
            eligible_group = eligibility[members]
            any_eligible = eligible_group.any(axis=1)
            values = competitive[members].where(eligible_group, -np.inf)
            best = values.idxmax(axis=1)
            for symbol in members:
                representative[symbol] = any_eligible & best.eq(symbol)
    else:
        representative = eligibility.copy().astype(bool)

    ruler_scores = competitive.where(representative)
    ruler_size = representative.sum(axis=1).astype(float)
    virtual_rank = pd.DataFrame(np.nan, index=raw_score.index, columns=raw_score.columns)
    for symbol in raw_score.columns:
        target = raw_score[symbol]
        virtual_rank[symbol] = (
            ruler_scores.gt(target, axis=0).sum(axis=1).astype(float) + 1.0
        ).where(target.notna() & ruler_size.gt(0))
    return representative, virtual_rank, ruler_size


def quota_gate(rank: pd.DataFrame, size: pd.Series, share: float, fixed_count: int | None = None) -> pd.DataFrame:
    if fixed_count is not None:
        quota = pd.Series(float(fixed_count), index=rank.index)
    else:
        quota = np.ceil(size * float(share)).clip(lower=1.0)
    return rank.le(quota, axis=0).fillna(False).astype(bool)


def build_entry_paths(
    components: dict,
    eligibility: pd.DataFrame,
    formal: dict,
    sentiment: dict[str, pd.DataFrame],
    available_series: pd.Series,
    categories: dict[str, str],
    rank: pd.DataFrame,
    ruler_size: pd.Series,
    *,
    share_multiplier: float = 1.0,
    fixed_counts: bool = False,
) -> dict[str, pd.DataFrame]:
    features = components["features"]
    r2 = components["r2"]
    efficiency = components["efficiency"]
    roc5 = components["roc5"]
    symbols = list(eligibility.columns)
    available = broadcast(available_series, symbols)
    live = formal["enhanced_selection"]["sentiment_available"]
    fallback_cfg = formal["enhanced_selection"]["price_only_for_historical_dates_without_sentiment"]
    values = formal["rules"]

    def tail(name: str, count: int):
        return quota_gate(
            rank, ruler_size, SHARES[name] * share_multiplier,
            fixed_count=count if fixed_counts else None,
        )

    top3 = tail("fallback", 3)
    top5 = tail("normal", 5)
    top15 = tail("emerging", 15)
    breadth = category_breadth(features.roc_short, categories)
    normal_price = (
        eligibility
        & features.roc_short.gt(0.0)
        & features.roc_medium.gt(0.0)
        & features.above_ma
        & features.ma_bias.le(float(values["max_entry_ma_bias"]))
    )
    fallback = (
        normal_price & top3
        & breadth.ge(float(fallback_cfg["category_roc20_positive_breadth_min"]))
    )
    weak_edge = top5 & ~top3 & features.roc_short.lt(0.02)
    edge_confirm = (
        sentiment["matched_count"].ge(3)
        & sentiment["count_acceleration"].ge(0.0)
        & sentiment["positive_dde_share"].ge(0.50)
    )
    current_normal = normal_price & top5 & (~weak_edge | edge_confirm)

    emerging_cfg = live["emerging_trend"]
    emerging_trigger = (
        eligibility
        & top15
        & features.roc_short.ge(float(emerging_cfg["roc20_min"]))
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
        eligibility & top5
        & features.roc_short.gt(0.0)
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
    gate = (~available & fallback) | (available & (current_normal | emerging | extension))
    score = features.ranking_score.where(~emerging, features.roc_short + 0.05 * r2.fillna(0.0))
    gate = theme_champions(gate, score, categories)
    return {
        "gate": gate,
        "score": score,
        "top3": top3,
        "top5": top5,
        "top15": top15,
        "fallback": fallback,
        "normal": current_normal,
        "emerging": emerging,
        "extension": extension,
        "available": available,
    }


def soft_exit(
    components: dict,
    formal: dict,
    sentiment: dict[str, pd.DataFrame],
    available_series: pd.Series,
    top3: pd.DataFrame,
) -> pd.DataFrame:
    features = components["features"]
    symbols = list(top3.columns)
    available = broadcast(available_series, symbols)
    fallback = formal["enhanced_selection"]["price_only_for_historical_dates_without_sentiment"]["soft_exit_protection"]
    strong = (
        ~available & top3
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


def run_v4(
    name: str,
    market: dict,
    formal: dict,
    panel: dict,
    symbols: list[str],
    categories: dict[str, str],
    sentiment: dict[str, pd.DataFrame],
    available: pd.Series,
    *,
    theme_level: bool,
    share_multiplier: float = 1.0,
    fixed_counts: bool = False,
    relative_exit: bool = True,
) -> dict:
    components = build_components(panel, symbols, formal)
    features = components["features"]
    eligibility, _, _ = entry_eligibility(panel, symbols, formal["rules"])
    representatives, rank, size = comparison_ruler(
        features.ranking_score, features.roc_short, features.roc_medium,
        eligibility, categories, theme_level=theme_level,
    )
    paths = build_entry_paths(
        components, eligibility, formal, sentiment, available, categories,
        rank, size, share_multiplier=share_multiplier, fixed_counts=fixed_counts,
    )
    if relative_exit:
        decline = (
            rank.gt(rank.shift(int(formal["rules"]["rank_change_short_days"])))
            & rank.gt(rank.shift(int(formal["rules"]["rank_change_long_days"])))
        ).fillna(False)
    else:
        formal_rank = features.ranking_score.rank(axis=1, ascending=False, method="min")
        decline = (
            formal_rank.gt(formal_rank.shift(int(formal["rules"]["rank_change_short_days"])))
            & formal_rank.gt(formal_rank.shift(int(formal["rules"]["rank_change_long_days"])))
        ).fillna(False)
    bundle, _ = etfwin_signals(
        panel["close"][symbols], symbols, _rules(formal["rules"]),
        entry_eligibility=eligibility,
        entry_gate=paths["gate"],
        entry_ranking_score_override=paths["score"],
        soft_exit_confirmation=soft_exit(components, formal, sentiment, available, paths["top3"]),
        dual_rank_decline_override=decline,
        reentry_cooldown_days=int(formal["enhanced_selection"]["reentry_cooldown_days"]),
    )
    project = execution_project(
        market, premium_sensitive(symbols), eligibility.shift(1, fill_value=False).astype(bool)
    )
    result = run_backtest(
        name, panel, bundle.weights, START, END, project,
        cash_management=cash_management(formal),
    )
    capital = float(market["project"]["initial_capital"])
    return {
        "name": name,
        "result": result,
        "weights": bundle.weights,
        "gate": paths["gate"],
        "rank": rank,
        "ruler_size": size,
        "representatives": representatives,
        "periods": {key: period_metrics(result.equity, start, end, capital) for key, (start, end) in PERIODS.items()},
    }


def metric_row(variant: dict) -> dict:
    row = {"variant": variant["name"], **variant["result"].metrics}
    for key, values in variant["periods"].items():
        row[f"return_{key}"] = values["total_return"]
        row[f"mdd_{key}"] = values["max_drawdown"]
    row["candidate_day_rate"] = float(variant["gate"].loc[START:].any(axis=1).mean())
    row["ruler_size_mean"] = float(variant["ruler_size"].loc[START:].mean())
    return row


def active(weights: pd.DataFrame) -> pd.Series:
    frame = weights.loc[START:].fillna(0.0)
    return frame.idxmax(axis=1).where(frame.max(axis=1).gt(0), "CASH")


def path_difference(a: pd.DataFrame, b: pd.DataFrame) -> int:
    cols = sorted(set(a.columns) | set(b.columns))
    left = a.loc[START:].reindex(columns=cols, fill_value=0.0)
    right = b.loc[START:].reindex(index=left.index, columns=cols, fill_value=0.0)
    return int((~np.isclose(left.to_numpy(), right.to_numpy(), atol=1e-12).all(axis=1)).sum())


def formal_benchmark(key, market, formal, panel, all_symbols, core_symbols, categories, sentiment, available, symbols, mode, challengers):
    value = run_variant(
        key, market, formal, panel, all_symbols, core_symbols, categories,
        sentiment, available, symbols=symbols, mode=mode, challengers=challengers,
    )
    capital = float(market["project"]["initial_capital"])
    row = {"variant": key, **value["metrics"]}
    for name, (start, end) in PERIODS.items():
        p = period_metrics(value["equity"], start, end, capital)
        row[f"return_{name}"] = p["total_return"]
        row[f"mdd_{name}"] = p["max_drawdown"]
    return value, row


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

    core45, core_row = formal_benchmark(
        "baseline_core_45", market, formal, panel, all_symbols, core_symbols,
        categories_all, sentiment_all, available, core_symbols, "core_anchor_challenger", [],
    )
    formal_f, formal_row = formal_benchmark(
        "formal_champion_cash_gap_51", market, formal, panel, all_symbols, core_symbols,
        categories_all, sentiment_all, available, all_symbols, "core_champion_cash_gap", challengers,
    )

    variants = {}
    specs = {
        PRIMARY: dict(theme_level=True, relative_exit=True),
        "theme_monotone_tail_formal_exit_ablation": dict(theme_level=True, relative_exit=False),
        "asset_monotone_tail_relative_exit": dict(theme_level=False, relative_exit=True),
        "theme_fixed_count_relative_exit": dict(theme_level=True, fixed_counts=True, relative_exit=True),
        "asset_fixed_count_relative_exit": dict(theme_level=False, fixed_counts=True, relative_exit=True),
    }
    for name, spec in specs.items():
        variants[name] = run_v4(
            name, market, formal, panel, all_symbols, categories_all,
            sentiment_all, available, **spec,
        )
    primary = variants[PRIMARY]

    sensitivity = []
    for multiple in (0.60, 0.80, 1.00, 1.20, 1.40, 1.80):
        variant = run_v4(
            f"theme_tail_x{multiple:.2f}", market, formal, panel, all_symbols,
            categories_all, sentiment_all, available, theme_level=True,
            share_multiplier=multiple, relative_exit=True,
        )
        row = metric_row(variant)
        row["share_multiplier"] = multiple
        sensitivity.append(row)

    add_one = []
    for symbol in challengers:
        subset = core_symbols + [symbol]
        variant = run_v4(
            f"add_{symbol}", market, formal, panel, subset,
            {s: categories_all[s] for s in subset},
            {key: frame[subset] for key, frame in sentiment_all.items()},
            available, theme_level=True, relative_exit=True,
        )
        add_one.append({
            "symbol": symbol,
            "total_return": variant["result"].metrics["total_return"],
            "holding_days": int(active(variant["weights"]).eq(symbol).sum()),
            "path_difference_days_vs_full51": path_difference(variant["weights"], primary["weights"]),
        })

    reversed_symbols = list(reversed(all_symbols))
    reordered = run_v4(
        "primary_reordered", market, formal, panel, reversed_symbols,
        {s: categories_all[s] for s in reversed_symbols},
        {key: frame[reversed_symbols] for key, frame in sentiment_all.items()},
        available, theme_level=True, relative_exit=True,
    )

    # Direct monotonicity test on the primary candidate gate: adding an invalid
    # or lower-scoring synthetic theme may expand a quota but may not remove an
    # existing candidate.  A stronger addition is intentionally allowed to compete.
    features = build_components(panel, all_symbols, formal)["features"]
    eligibility, _, _ = entry_eligibility(panel, all_symbols, formal["rules"])
    reps, base_rank, base_size = comparison_ruler(
        features.ranking_score, features.roc_short, features.roc_medium,
        eligibility, categories_all, theme_level=True,
    )
    base_tail = quota_gate(base_rank, base_size, SHARES["normal"])
    # Algebraic check: increasing denominator by one while every old rank stays
    # unchanged cannot make ceil(q*N) smaller.
    expanded_quota = np.ceil((base_size + 1.0) * SHARES["normal"]).clip(lower=1.0)
    old_quota = np.ceil(base_size * SHARES["normal"]).clip(lower=1.0)
    monotone_quota = bool(expanded_quota.ge(old_quota).all())

    metrics = pd.DataFrame([metric_row(v) for v in variants.values()])
    primary_active = active(primary["weights"])
    formal_active = active(formal_f["weights"])
    paths = pd.DataFrame({"v4": primary_active, "formal_f": formal_active})
    paths["different"] = paths["v4"].ne(paths["formal_f"])
    invariants = {
        "column_order_signal_difference_days": path_difference(reordered["weights"], primary["weights"]),
        "lower_or_invalid_addition_cannot_shrink_quota": monotone_quota,
        "directionally_invalid_etf_has_finite_competitive_slot": False,
        "primary_path_difference_days_vs_formal": int(paths["different"].sum()),
        "challenger_holding_days": {symbol: int(primary_active.eq(symbol).sum()) for symbol in challengers},
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([core_row, formal_row]).to_csv(OUTPUT / "benchmarks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(OUTPUT / "share_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(add_one).to_csv(OUTPUT / "add_one_challenger.csv", index=False, encoding="utf-8-sig")
    paths.loc[paths["different"]].to_csv(OUTPUT / "path_differences_vs_formal.csv", encoding="utf-8-sig")
    payload = {
        "status": "research_evidence_v4",
        "generated_through": END,
        "formal_strategy_unchanged": True,
        "primary_predeclared": PRIMARY,
        "shares_derived_from_original_45": SHARES,
        "metrics": clean(metrics.to_dict(orient="records")),
        "benchmarks": clean([core_row, formal_row]),
        "sensitivity": clean(sensitivity),
        "add_one": clean(add_one),
        "invariants": clean(invariants),
        "limitations": [
            "current-survivor ETF universe",
            "theme labels are current frozen labels rather than historical point-in-time taxonomy",
            "all historical dates have already been examined by this project and are not true prospective OOS",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics[["variant", "total_return", "cagr", "max_drawdown", "sharpe", "candidate_day_rate", "ruler_size_mean"]].to_string(index=False))
    print("benchmarks", pd.DataFrame([core_row, formal_row])[["variant", "total_return", "cagr", "max_drawdown", "sharpe"]].to_string(index=False))
    print("invariants", json.dumps(invariants, ensure_ascii=False))


if __name__ == "__main__":
    main()
