"""V5 research: directionally investable ranking ruler.

Preserve the formal top-3/top-5 counts, raw momentum score, AI exceptions and
formal exit ranking.  Only the normal-entry ranking ruler changes: an ETF may
occupy a normal ranking slot only when it is listed/liquid and both ROC20 and
ROC60 are positive.  Emerging-trend rank remains eligibility-only because that
exception intentionally permits ROC60 below zero.

Research only; never mutates the formal strategy or live artifacts.
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
from universe_architecture_v3_conviction import build_components, theme_champions


OUTPUT = ROOT / "results" / "research" / "universe_architecture_v5"
START = "2018-07-02"
END = "2026-07-27"
PRIMARY = "directional_eligible_ruler_51"
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


def build_paths(
    components: dict,
    eligibility: pd.DataFrame,
    formal: dict,
    sentiment: dict[str, pd.DataFrame],
    available_series: pd.Series,
    categories: dict[str, str],
    *,
    mask_direction: bool,
    mask_eligibility: bool,
    theme_dedup: bool,
) -> dict[str, pd.DataFrame]:
    features = components["features"]
    r2 = components["r2"]
    efficiency = components["efficiency"]
    roc5 = components["roc5"]
    symbols = list(eligibility.columns)
    available = broadcast(available_series, symbols)
    values = formal["rules"]
    enhanced = formal["enhanced_selection"]
    live = enhanced["sentiment_available"]
    fallback_cfg = enhanced["price_only_for_historical_dates_without_sentiment"]
    raw_score = features.ranking_score

    ruler = pd.DataFrame(True, index=raw_score.index, columns=raw_score.columns)
    if mask_eligibility:
        ruler &= eligibility
    if mask_direction:
        ruler &= features.roc_short.gt(0.0) & features.roc_medium.gt(0.0)
    normal_rank = raw_score.where(ruler).rank(axis=1, ascending=False, method="min")

    # Emerging trend deliberately allows ROC60<0, but still cannot be displaced
    # by a product that is unlisted or fails the liquidity gate.
    emerging_rank = raw_score.where(eligibility).rank(axis=1, ascending=False, method="min")
    formal_rank = raw_score.rank(axis=1, ascending=False, method="min")
    breadth = category_breadth(features.roc_short, categories)

    normal_price = (
        eligibility
        & features.roc_short.gt(0.0)
        & features.roc_medium.gt(0.0)
        & features.above_ma
        & features.ma_bias.le(float(values["max_entry_ma_bias"]))
        & normal_rank.le(int(values["entry_rank_limit"]))
    )
    fallback = (
        normal_price
        & normal_rank.le(int(fallback_cfg["entry_rank_limit"]))
        & breadth.ge(float(fallback_cfg["category_roc20_positive_breadth_min"]))
    )
    weak_edge = normal_rank.ge(4) & features.roc_short.lt(0.02)
    edge_confirm = (
        sentiment["matched_count"].ge(3)
        & sentiment["count_acceleration"].ge(0.0)
        & sentiment["positive_dde_share"].ge(0.50)
    )
    current_normal = normal_price & (~weak_edge | edge_confirm)

    emerging_cfg = live["emerging_trend"]
    emerging_trigger = (
        eligibility
        & emerging_rank.le(int(emerging_cfg["maximum_base_rank"]))
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
        eligibility
        & normal_rank.le(int(extension_cfg["base_rank_limit"]))
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
    entry_score = raw_score.where(~emerging, features.roc_short + 0.05 * r2.fillna(0.0))
    if theme_dedup:
        gate = theme_champions(gate, entry_score, categories)

    soft = pd.DataFrame(True, index=raw_score.index, columns=raw_score.columns)
    missing_exit = fallback_cfg["soft_exit_protection"]
    strong = (
        ~available
        & normal_rank.le(int(missing_exit["rank_limit"]))
        & features.roc_medium.ge(float(missing_exit["roc60_min"]))
        & features.above_ma
    )
    hot = live["hot_exit_protection"]
    hot_trigger = (
        available
        & sentiment["matched_count"].ge(float(hot["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(hot["hot_score_min"]))
        & sentiment["positive_dde_share"].ge(float(hot["positive_dde_share_min"]))
    )
    hot_memory = hot_trigger.rolling(int(hot["memory_days"]), min_periods=1).max().fillna(False).astype(bool)
    soft &= ~strong & ~hot_memory
    formal_decline = (
        formal_rank.gt(formal_rank.shift(int(values["rank_change_short_days"])))
        & formal_rank.gt(formal_rank.shift(int(values["rank_change_long_days"])))
    ).fillna(False)
    ruler_decline = (
        normal_rank.gt(normal_rank.shift(int(values["rank_change_short_days"])))
        & normal_rank.gt(normal_rank.shift(int(values["rank_change_long_days"])))
    ).fillna(False)
    return {
        "gate": gate.astype(bool),
        "entry_score": entry_score,
        "normal_rank": normal_rank,
        "emerging_rank": emerging_rank,
        "formal_decline": formal_decline,
        "ruler_decline": ruler_decline,
        "soft": soft.astype(bool),
        "ruler": ruler.astype(bool),
    }


def run_v5(
    name: str,
    market: dict,
    formal: dict,
    panel: dict,
    symbols: list[str],
    categories: dict[str, str],
    sentiment: dict[str, pd.DataFrame],
    available: pd.Series,
    *,
    mask_direction: bool = True,
    mask_eligibility: bool = True,
    theme_dedup: bool = False,
    ruler_exit: bool = False,
) -> dict:
    components = build_components(panel, symbols, formal)
    eligibility, _, _ = entry_eligibility(panel, symbols, formal["rules"])
    paths = build_paths(
        components, eligibility, formal, sentiment, available, categories,
        mask_direction=mask_direction, mask_eligibility=mask_eligibility,
        theme_dedup=theme_dedup,
    )
    bundle, _ = etfwin_signals(
        panel["close"][symbols], symbols, _rules(formal["rules"]),
        entry_eligibility=eligibility,
        entry_gate=paths["gate"],
        entry_ranking_score_override=paths["entry_score"],
        soft_exit_confirmation=paths["soft"],
        dual_rank_decline_override=paths["ruler_decline"] if ruler_exit else paths["formal_decline"],
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
        "paths": paths,
        "periods": {key: period_metrics(result.equity, start, end, capital) for key, (start, end) in PERIODS.items()},
    }


def row(variant: dict) -> dict:
    output = {"variant": variant["name"], **variant["result"].metrics}
    for key, values in variant["periods"].items():
        output[f"return_{key}"] = values["total_return"]
        output[f"mdd_{key}"] = values["max_drawdown"]
    gate = variant["paths"]["gate"].loc[START:]
    ruler = variant["paths"]["ruler"].loc[START:]
    output["candidate_day_rate"] = float(gate.any(axis=1).mean())
    output["directional_ruler_size_mean"] = float(ruler.sum(axis=1).mean())
    return output


def active(weights: pd.DataFrame) -> pd.Series:
    frame = weights.loc[START:].fillna(0.0)
    return frame.idxmax(axis=1).where(frame.max(axis=1).gt(0), "CASH")


def path_diff(a: pd.DataFrame, b: pd.DataFrame) -> int:
    cols = sorted(set(a.columns) | set(b.columns))
    left = a.loc[START:].reindex(columns=cols, fill_value=0.0)
    right = b.loc[START:].reindex(index=left.index, columns=cols, fill_value=0.0)
    return int((~np.isclose(left.to_numpy(), right.to_numpy(), atol=1e-12).all(axis=1)).sum())


def benchmark(key, market, formal, panel, all_symbols, core_symbols, categories, sentiment, available, symbols, mode, challengers):
    value = run_variant(
        key, market, formal, panel, all_symbols, core_symbols, categories,
        sentiment, available, symbols=symbols, mode=mode, challengers=challengers,
    )
    capital = float(market["project"]["initial_capital"])
    output = {"variant": key, **value["metrics"]}
    for name, (start, end) in PERIODS.items():
        p = period_metrics(value["equity"], start, end, capital)
        output[f"return_{name}"] = p["total_return"]
        output[f"mdd_{name}"] = p["max_drawdown"]
    return value, output


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

    core45, core_row = benchmark(
        "baseline_core_45", market, formal, panel, all_symbols, core_symbols,
        categories_all, sentiment_all, available, core_symbols, "core_anchor_challenger", [],
    )
    formal_f, formal_row = benchmark(
        "formal_champion_cash_gap_51", market, formal, panel, all_symbols, core_symbols,
        categories_all, sentiment_all, available, all_symbols, "core_champion_cash_gap", challengers,
    )
    global51, global_row = benchmark(
        "global_rank_51", market, formal, panel, all_symbols, core_symbols,
        categories_all, sentiment_all, available, all_symbols, "fixed_pool", [],
    )

    variants = {}
    specs = {
        PRIMARY: {},
        "directional_only_ruler_51": {"mask_eligibility": False},
        "eligible_only_ruler_51": {"mask_direction": False},
        "directional_eligible_theme_dedup_51": {"theme_dedup": True},
        "directional_eligible_ruler_exit_51": {"ruler_exit": True},
    }
    for name, spec in specs.items():
        variants[name] = run_v5(
            name, market, formal, panel, all_symbols, categories_all,
            sentiment_all, available, **spec,
        )
    primary = variants[PRIMARY]
    core_v5 = run_v5(
        "directional_eligible_ruler_core45", market, formal, panel, core_symbols,
        {s: categories_all[s] for s in core_symbols},
        {key: frame[core_symbols] for key, frame in sentiment_all.items()},
        available,
    )

    add_one = []
    for symbol in challengers:
        subset = core_symbols + [symbol]
        variant = run_v5(
            f"add_{symbol}", market, formal, panel, subset,
            {s: categories_all[s] for s in subset},
            {key: frame[subset] for key, frame in sentiment_all.items()}, available,
        )
        add_one.append({
            "symbol": symbol,
            "total_return": variant["result"].metrics["total_return"],
            "holding_days": int(active(variant["weights"]).eq(symbol).sum()),
            "path_difference_days_vs_v5_core45": path_diff(variant["weights"], core_v5["weights"]),
        })

    reversed_symbols = list(reversed(all_symbols))
    reordered = run_v5(
        "primary_reordered", market, formal, panel, reversed_symbols,
        {s: categories_all[s] for s in reversed_symbols},
        {key: frame[reversed_symbols] for key, frame in sentiment_all.items()}, available,
    )
    primary_active = active(primary["weights"])
    formal_active = active(formal_f["weights"])
    paths = pd.DataFrame({"v5": primary_active, "formal_f": formal_active})
    paths["different"] = paths["v5"].ne(paths["formal_f"])
    invariants = {
        "column_order_signal_difference_days": path_diff(reordered["weights"], primary["weights"]),
        "invalid_direction_or_ineligible_can_occupy_normal_rank": False,
        "primary_path_difference_days_vs_formal": int(paths["different"].sum()),
        "v5_full51_vs_v5_core45_difference_days": path_diff(primary["weights"], core_v5["weights"]),
        "challenger_holding_days": {s: int(primary_active.eq(s).sum()) for s in challengers},
    }

    metrics = pd.DataFrame([row(v) for v in variants.values()] + [row(core_v5)])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([core_row, formal_row, global_row]).to_csv(OUTPUT / "benchmarks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(add_one).to_csv(OUTPUT / "add_one_challenger.csv", index=False, encoding="utf-8-sig")
    paths.loc[paths["different"]].to_csv(OUTPUT / "path_differences_vs_formal.csv", encoding="utf-8-sig")
    payload = {
        "status": "research_evidence_v5",
        "generated_through": END,
        "formal_strategy_unchanged": True,
        "primary_predeclared": PRIMARY,
        "metrics": clean(metrics.to_dict(orient="records")),
        "benchmarks": clean([core_row, formal_row, global_row]),
        "add_one": clean(add_one),
        "invariants": clean(invariants),
        "limitations": [
            "current-survivor ETF universe",
            "all historical dates have already been examined by the project",
            "directional ruler is an architecture test, not prospective proof",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics[["variant", "total_return", "cagr", "max_drawdown", "sharpe", "candidate_day_rate", "directional_ruler_size_mean"]].to_string(index=False))
    print("benchmarks", pd.DataFrame([core_row, formal_row, global_row])[["variant", "total_return", "cagr", "max_drawdown", "sharpe"]].to_string(index=False))
    print("invariants", json.dumps(invariants, ensure_ascii=False))


if __name__ == "__main__":
    main()
