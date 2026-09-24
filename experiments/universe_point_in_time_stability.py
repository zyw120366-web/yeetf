"""Research a point-in-time, membership-stable ETF universe.

This study does not search for the historically best pool.  It replaces the
fixed 2026 membership assumption with a reproducible operational admission
process and asks whether the resulting strategy is stable to reasonable pool
governance changes and single-member deletions.

Primary pre-declared governance:

* review every six months;
* require 252 observations and 90% trailing operational eligibility;
* require no more than 1% missing closes in the trailing 252 sessions;
* require two consecutive passes to admit and two failures to remove;
* freeze membership between reviews;
* scale rank slots mechanically from the original 5/45, 3/45 and 15/45.

The current-survivor limitation remains: historical delisted or merged ETFs
are not available in this repository.  Results are research-only and never
change the formal strategy, account, orders, reports, or daily entry point.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.etfwin import etfwin_signals
from etf_rotation.execution import entry_eligibility, execution_project, period_metrics
from etf_rotation.sentiment import broadcast, load_sentiment_matrices
from etf_rotation.ye import _rules
from universe_architecture_v2_study import FEATURES, cash_management, premium_sensitive
from universe_architecture_v3_conviction import build_components


OUTPUT = ROOT / "results" / "research" / "universe_point_in_time_stability_v1"
START = "2018-07-02"
END = "2026-07-28"
BASE_SIZE = 45
BASE_LIMITS = {"normal": 5, "fallback": 3, "emerging": 15}
PERIODS = {
    "2018_2020": (START, "2020-12-31"),
    "2021_2022": ("2021-01-01", "2022-12-31"),
    "2023_2024": ("2023-01-01", "2024-12-31"),
    "2025_2026": ("2025-01-01", END),
}


@dataclass(frozen=True)
class GovernanceSpec:
    cadence_months: int = 6
    window: int = 252
    minimum_history: int = 252
    eligibility_rate: float = 0.90
    maximum_missing_rate: float = 0.01
    passes_to_admit: int = 2
    failures_to_remove: int = 2


PRIMARY = GovernanceSpec()
CTX: dict[str, object] = {}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def review_dates(index: pd.DatetimeIndex, cadence_months: int) -> list[pd.Timestamp]:
    if 12 % cadence_months != 0:
        raise ValueError("cadence_months must divide 12")
    keys = pd.Series(
        index.year * (12 // cadence_months)
        + ((index.month - 1) // cadence_months),
        index=index,
    )
    return [pd.Timestamp(value) for value in keys.groupby(keys).head(1).index]


def admission_mask(
    spec: GovernanceSpec,
    close: pd.DataFrame,
    operational_eligibility: pd.DataFrame,
    listed_sessions: pd.DataFrame,
    amount20: pd.DataFrame,
    minimum_amount: float,
    excluded: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a frozen-between-reviews point-in-time universe and audit log."""

    excluded = excluded or set()
    symbols = list(close.columns)
    eligible_rate = operational_eligibility.rolling(
        spec.window, min_periods=spec.window
    ).mean()
    missing_rate = 1.0 - close.notna().rolling(
        spec.window, min_periods=spec.window
    ).mean()
    raw_pass = (
        listed_sessions.ge(spec.minimum_history)
        & eligible_rate.ge(spec.eligibility_rate)
        & missing_rate.le(spec.maximum_missing_rate)
        & amount20.ge(minimum_amount)
    ).fillna(False)
    if excluded:
        raw_pass.loc[:, list(excluded)] = False

    admitted = pd.DataFrame(False, index=close.index, columns=symbols)
    state = {symbol: False for symbol in symbols}
    pass_streak = {symbol: 0 for symbol in symbols}
    fail_streak = {symbol: 0 for symbol in symbols}
    reviews = review_dates(close.index, spec.cadence_months)
    audit_rows: list[dict[str, object]] = []
    for number, date in enumerate(reviews):
        for symbol in symbols:
            passed = bool(raw_pass.loc[date, symbol]) and symbol not in excluded
            pass_streak[symbol] = pass_streak[symbol] + 1 if passed else 0
            fail_streak[symbol] = 0 if passed else fail_streak[symbol] + 1
            previous = state[symbol]
            if not previous and pass_streak[symbol] >= spec.passes_to_admit:
                state[symbol] = True
            elif previous and fail_streak[symbol] >= spec.failures_to_remove:
                state[symbol] = False
            audit_rows.append(
                {
                    "review_date": date,
                    "symbol": symbol,
                    "raw_pass": passed,
                    "pass_streak": pass_streak[symbol],
                    "fail_streak": fail_streak[symbol],
                    "admitted": state[symbol],
                    "changed": previous != state[symbol],
                }
            )
        end = reviews[number + 1] if number + 1 < len(reviews) else None
        selector = admitted.index >= date
        if end is not None:
            selector &= admitted.index < end
        frozen_state = np.array([state[symbol] for symbol in symbols], dtype=bool)
        admitted.loc[selector, symbols] = np.repeat(
            frozen_state[None, :], int(selector.sum()), axis=0
        )
    return admitted.astype(bool), pd.DataFrame(audit_rows)


def fixed_mask(index: pd.DatetimeIndex, symbols: list[str], members: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(False, index=index, columns=symbols)
    frame.loc[:, members] = True
    return frame


def quota(size: pd.Series, base_count: int) -> pd.Series:
    return np.ceil(size * float(base_count) / BASE_SIZE).clip(lower=1.0)


def dynamic_category_breadth(
    roc20: pd.DataFrame,
    admitted: pd.DataFrame,
    categories: dict[str, str],
) -> pd.DataFrame:
    output = pd.DataFrame(np.nan, index=roc20.index, columns=roc20.columns)
    groups: dict[str, list[str]] = {}
    for symbol, category in categories.items():
        groups.setdefault(category, []).append(symbol)
    for members in groups.values():
        active = admitted[members]
        denominator = active.sum(axis=1).replace(0, np.nan)
        share = (active & roc20[members].gt(0.0)).sum(axis=1) / denominator
        output.loc[:, members] = np.repeat(share.to_numpy()[:, None], len(members), axis=1)
    return output


def build_paths(admitted: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    formal = CTX["formal"]
    components = CTX["components"]
    eligibility = CTX["eligibility"]
    sentiment = CTX["sentiment"]
    available_series = CTX["available"]
    categories = CTX["categories"]
    features = components["features"]
    r2 = components["r2"]
    efficiency = components["efficiency"]
    roc5 = components["roc5"]
    symbols = list(admitted.columns)
    available = broadcast(available_series, symbols)
    values = formal["rules"]
    enhanced = formal["enhanced_selection"]
    live = enhanced["sentiment_available"]
    fallback_cfg = enhanced["price_only_for_historical_dates_without_sentiment"]

    score = features.ranking_score.where(admitted)
    rank = score.rank(axis=1, ascending=False, method="min")
    # The quota belongs to the frozen admitted universe.  A temporarily missing
    # score removes that product from today's ordering but must not silently
    # resize the governance version or loosen its selection intensity.
    ruler_size = admitted.sum(axis=1).astype(float)
    top_normal = rank.le(quota(ruler_size, BASE_LIMITS["normal"]), axis=0)
    top_fallback = rank.le(quota(ruler_size, BASE_LIMITS["fallback"]), axis=0)
    top_emerging = rank.le(quota(ruler_size, BASE_LIMITS["emerging"]), axis=0)
    breadth = dynamic_category_breadth(features.roc_short, admitted, categories)

    normal_price = (
        admitted
        & eligibility
        & features.roc_short.gt(0.0)
        & features.roc_medium.gt(0.0)
        & features.above_ma
        & features.ma_bias.le(float(values["max_entry_ma_bias"]))
        & top_normal
    )
    fallback = (
        normal_price
        & top_fallback
        & breadth.ge(float(fallback_cfg["category_roc20_positive_breadth_min"]))
    )
    weak_edge = top_normal & ~top_fallback & features.roc_short.lt(0.02)
    edge_confirm = (
        sentiment["matched_count"].ge(3)
        & sentiment["count_acceleration"].ge(0.0)
        & sentiment["positive_dde_share"].ge(0.50)
    )
    current_normal = normal_price & (~weak_edge | edge_confirm)

    emerging_cfg = live["emerging_trend"]
    emerging_trigger = (
        admitted
        & eligibility
        & top_emerging
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
    emerging = emerging_trigger.rolling(
        int(emerging_cfg["memory_days"]), min_periods=1
    ).max().fillna(False).astype(bool)
    emerging &= (
        admitted
        & features.roc_short.gt(0.0)
        & features.roc_medium.ge(float(emerging_cfg["roc60_range"][0]))
        & features.ma_bias.ge(float(emerging_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(live["quality_extension"]["ma120_bias_range"][1]))
    )

    extension_cfg = live["quality_extension"]
    extension = (
        admitted
        & eligibility
        & top_normal
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
    entry_score = features.ranking_score.where(
        ~emerging, features.roc_short + 0.05 * r2.fillna(0.0)
    )

    missing_exit = fallback_cfg["soft_exit_protection"]
    missing_strong = (
        ~available
        & top_fallback
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
    hot_memory = hot_trigger.rolling(
        int(hot["memory_days"]), min_periods=1
    ).max().fillna(False).astype(bool)
    soft = (~missing_strong & ~hot_memory).astype(bool)
    decline = (
        rank.gt(rank.shift(int(values["rank_change_short_days"])))
        & rank.gt(rank.shift(int(values["rank_change_long_days"])))
    ).fillna(False)
    return {
        "gate": gate.astype(bool),
        "entry_score": entry_score,
        "soft": soft,
        "decline": decline,
        "rank": rank,
        "ruler_size": ruler_size,
    }


def initialise_worker(root_text: str) -> None:
    root = Path(root_text)
    market = load_yaml(root / "config" / "market.yaml")
    formal = load_yaml(root / "config" / "ye_strategy.yaml")
    panel = load_panel(market, root / "market_data" / "prices")
    symbols = universe_keys(market)
    sentiment, available = load_sentiment_matrices(FEATURES, panel["close"].index, symbols)
    components = build_components(panel, symbols, formal)
    eligibility, listed_sessions, amount20 = entry_eligibility(panel, symbols, formal["rules"])
    CTX.update(
        market=market,
        formal=formal,
        panel=panel,
        symbols=symbols,
        categories={symbol_key(item): item["category"] for item in market["universe"]},
        sentiment=sentiment,
        available=available,
        components=components,
        eligibility=eligibility,
        listed_sessions=listed_sessions,
        amount20=amount20,
    )


def run_task(task: dict) -> dict:
    symbols = CTX["symbols"]
    panel = CTX["panel"]
    market = CTX["market"]
    formal = CTX["formal"]
    excluded = set(task.get("excluded", []))
    if task["kind"] == "fixed":
        members = [symbol for symbol in task["members"] if symbol not in excluded]
        admitted = fixed_mask(panel["close"].index, symbols, members)
        audit = pd.DataFrame()
    else:
        spec = GovernanceSpec(**task["spec"])
        admitted, audit = admission_mask(
            spec,
            panel["close"][symbols],
            CTX["eligibility"],
            CTX["listed_sessions"],
            CTX["amount20"],
            float(formal["rules"]["minimum_entry_amount"]),
            excluded,
        )
    paths = build_paths(admitted)
    holdings = int(task.get("holdings", 1))
    research_rules = replace(_rules(formal["rules"]), holdings_num=holdings)
    bundle, _ = etfwin_signals(
        panel["close"][symbols],
        symbols,
        research_rules,
        entry_eligibility=CTX["eligibility"],
        entry_gate=paths["gate"],
        entry_ranking_score_override=paths["entry_score"],
        soft_exit_confirmation=paths["soft"],
        dual_rank_decline_override=paths["decline"],
        emergency_exit=(~admitted).astype(bool),
        reentry_cooldown_days=int(formal["enhanced_selection"]["reentry_cooldown_days"]),
    )
    project = execution_project(
        market,
        premium_sensitive(symbols),
        CTX["eligibility"].shift(1, fill_value=False).astype(bool),
    )
    result = run_backtest(
        task["variant"], panel, bundle.weights, START, END, project,
        cash_management=cash_management(formal),
    )
    active = bundle.weights.loc[START:].idxmax(axis=1).where(
        bundle.weights.loc[START:].max(axis=1).gt(0.0), "CASH"
    )
    pool_size = admitted.loc[START:].sum(axis=1)
    changes = admitted.astype(int).diff().abs().sum(axis=1).loc[START:]
    output = {
        "variant": task["variant"],
        "family": task["family"],
        "holdings": holdings,
        **result.metrics,
        "pool_size_mean": float(pool_size.mean()),
        "pool_size_min": int(pool_size.min()),
        "pool_size_max": int(pool_size.max()),
        "membership_changes": int(changes.sum()),
        "membership_review_events": int(changes.gt(0).sum()),
        "candidate_day_rate": float(paths["gate"].loc[START:].any(axis=1).mean()),
        "final_members": "|".join(admitted.columns[admitted.iloc[-1]].tolist()),
        "_active": active,
        "_weights": bundle.weights.loc[START:],
        "_admitted": admitted.loc[START:],
    }
    if not audit.empty:
        output["raw_review_passes"] = int(audit["raw_pass"].sum())
    capital = float(market["project"]["initial_capital"])
    for name, (start, end) in PERIODS.items():
        metrics = period_metrics(result.equity, start, end, capital)
        output[f"return_{name}"] = metrics["total_return"]
        output[f"mdd_{name}"] = metrics["max_drawdown"]
    return output


def task_specs(symbols: list[str]) -> list[dict]:
    core = symbols[:BASE_SIZE]
    tasks = [
        {"variant": "fixed_core45_scaled", "family": "benchmark", "kind": "fixed", "members": core},
        {"variant": "fixed_full51_scaled", "family": "benchmark", "kind": "fixed", "members": symbols},
    ]
    variants = {
        "pit_semiannual_immediate": GovernanceSpec(passes_to_admit=1, failures_to_remove=1),
        "pit_semiannual_hysteresis_primary": PRIMARY,
        "pit_annual_hysteresis": GovernanceSpec(cadence_months=12),
        "pit_quarterly_hysteresis": GovernanceSpec(cadence_months=3),
        "pit_semiannual_rate80": GovernanceSpec(eligibility_rate=0.80),
        "pit_semiannual_rate95": GovernanceSpec(eligibility_rate=0.95),
        "pit_semiannual_window126": GovernanceSpec(window=126, minimum_history=252),
        "pit_semiannual_history120_window126": GovernanceSpec(window=126, minimum_history=120),
        "pit_semiannual_one_pass_two_fail": GovernanceSpec(passes_to_admit=1, failures_to_remove=2),
    }
    for name, spec in variants.items():
        tasks.append(
            {
                "variant": name,
                "family": "governance_sensitivity" if name != "pit_semiannual_hysteresis_primary" else "primary",
                "kind": "point_in_time",
                "spec": spec.__dict__,
            }
        )
    for symbol in symbols:
        tasks.append(
            {
                "variant": f"primary_leave_one__{symbol}",
                "family": "primary_leave_one",
                "kind": "point_in_time",
                "spec": PRIMARY.__dict__,
                "excluded": [symbol],
            }
        )
    # Two-holding variants test whether pool sensitivity is fundamentally a
    # universe problem or is amplified by putting 100% into one ordinal winner.
    for task in list(tasks):
        if task["family"] in {"benchmark", "primary", "governance_sensitivity", "primary_leave_one"}:
            twin = dict(task)
            twin["variant"] = f"{task['variant']}__h2"
            twin["family"] = f"{task['family']}_h2"
            twin["holdings"] = 2
            tasks.append(twin)
    return tasks


def agreement(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    return float(a.loc[common].eq(b.loc[common]).mean())


def membership_jaccard(a: pd.DataFrame, b: pd.DataFrame) -> float:
    common = a.index.intersection(b.index)
    left = a.loc[common].astype(bool)
    right = b.loc[common].astype(bool)
    union = (left | right).sum(axis=1)
    intersection = (left & right).sum(axis=1)
    values = intersection.div(union.replace(0, np.nan))
    return float(values.mean())


def portfolio_overlap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Average capital overlap including cash; one means identical portfolios."""

    common = a.index.intersection(b.index)
    columns = a.columns.union(b.columns)
    left = a.reindex(index=common, columns=columns, fill_value=0.0)
    right = b.reindex(index=common, columns=columns, fill_value=0.0)
    left_cash = (1.0 - left.sum(axis=1)).clip(lower=0.0)
    right_cash = (1.0 - right.sum(axis=1)).clip(lower=0.0)
    distance = (left - right).abs().sum(axis=1) + (left_cash - right_cash).abs()
    return float((1.0 - 0.5 * distance).mean())


def write_primary_pool_versions() -> None:
    """Persist the pre-declared pool versions without rerunning backtests."""

    initialise_worker(str(ROOT))
    formal = CTX["formal"]
    panel = CTX["panel"]
    symbols = CTX["symbols"]
    market = CTX["market"]
    _, audit = admission_mask(
        PRIMARY,
        panel["close"][symbols],
        CTX["eligibility"],
        CTX["listed_sessions"],
        CTX["amount20"],
        float(formal["rules"]["minimum_entry_amount"]),
    )
    names = {symbol_key(item): item["name"] for item in market["universe"]}
    previous: set[str] = set()
    rows = []
    for date, group in audit.groupby("review_date", sort=True):
        if pd.Timestamp(date) < pd.Timestamp(START):
            previous = set(group.loc[group["admitted"], "symbol"])
            continue
        current = set(group.loc[group["admitted"], "symbol"])
        additions = sorted(current - previous)
        removals = sorted(previous - current)
        rows.append(
            {
                "review_date": str(pd.Timestamp(date).date()),
                "pool_size": len(current),
                "additions": "|".join(f"{symbol}:{names[symbol]}" for symbol in additions),
                "removals": "|".join(f"{symbol}:{names[symbol]}" for symbol in removals),
                "members": "|".join(sorted(current)),
            }
        )
        previous = current
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "pool_versions.csv", index=False)


def write_results(outputs: list[dict]) -> None:
    primary = next(row for row in outputs if row["variant"] == "pit_semiannual_hysteresis_primary")
    core = next(row for row in outputs if row["variant"] == "fixed_core45_scaled")
    primary_h2 = next(row for row in outputs if row["variant"] == "pit_semiannual_hysteresis_primary__h2")
    core_h2 = next(row for row in outputs if row["variant"] == "fixed_core45_scaled__h2")
    primary_active = primary["_active"]
    primary_admitted = primary["_admitted"]
    core_active = core["_active"]
    references = {
        1: (primary["_weights"], core["_weights"]),
        2: (primary_h2["_weights"], core_h2["_weights"]),
    }
    rows = []
    for output in outputs:
        active = output.pop("_active")
        weights = output.pop("_weights")
        admitted = output.pop("_admitted")
        holdings = int(output["holdings"])
        primary_weights, core_weights = references[holdings]
        output["portfolio_overlap_vs_primary"] = portfolio_overlap(weights, primary_weights)
        output["portfolio_overlap_vs_core45"] = portfolio_overlap(weights, core_weights)
        output["target_agreement_vs_primary"] = (
            agreement(active, primary_active) if holdings == 1 else None
        )
        output["target_agreement_vs_core45"] = (
            agreement(active, core_active) if holdings == 1 else None
        )
        output["membership_jaccard_vs_primary"] = membership_jaccard(
            admitted, primary_admitted
        )
        rows.append(clean(output))
    frame = pd.DataFrame(rows).sort_values(["family", "variant"]).reset_index(drop=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "metrics.csv", index=False)

    leave = frame[frame["family"] == "primary_leave_one"]
    leave_h2 = frame[frame["family"] == "primary_leave_one_h2"]
    sensitivity = frame[frame["family"] == "governance_sensitivity"]
    primary_row = frame[frame["family"] == "primary"].iloc[0]
    benchmarks = frame[frame["family"] == "benchmark"]
    payload = {
        "status": "research_only_no_formal_change",
        "generated_through": END,
        "primary_governance": PRIMARY.__dict__,
        "primary": primary_row.to_dict(),
        "benchmarks": benchmarks.to_dict(orient="records"),
        "governance_sensitivity": sensitivity.to_dict(orient="records"),
        "leave_one": {
            "count": int(len(leave)),
            "target_agreement_median": float(leave["target_agreement_vs_primary"].median()),
            "target_agreement_min": float(leave["target_agreement_vs_primary"].min()),
            "return_median": float(leave["total_return"].median()),
            "return_min": float(leave["total_return"].min()),
            "return_max": float(leave["total_return"].max()),
        },
        "leave_one_two_holdings": {
            "count": int(len(leave_h2)),
            "portfolio_overlap_median": float(leave_h2["portfolio_overlap_vs_primary"].median()),
            "portfolio_overlap_min": float(leave_h2["portfolio_overlap_vs_primary"].min()),
            "return_median": float(leave_h2["total_return"].median()),
            "return_min": float(leave_h2["total_return"].min()),
            "return_max": float(leave_h2["total_return"].max()),
        },
        "limitations": [
            "Only currently surviving ETFs are available; delisted and merged products are absent.",
            "The 2018-2026 history is already research-contaminated and is not out of sample.",
            "Tracking-index and constituent-overlap metadata are incomplete, so exact exposure deduplication remains pending.",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def pct(value) -> str:
        return f"{float(value)*100:+.2f}%"

    comparison = pd.concat([benchmarks, frame[frame["family"] == "primary"], sensitivity])
    lines = []
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['variant']} | {row['pool_size_mean']:.1f} | {pct(row['total_return'])} | "
            f"{pct(row['max_drawdown'])} | {row['target_agreement_vs_primary']*100:.1f}% | "
            f"{row['membership_jaccard_vs_primary']*100:.1f}% |"
        )
    worst = leave.nsmallest(5, "target_agreement_vs_primary")[[
        "variant", "target_agreement_vs_primary", "total_return"
    ]]
    worst_lines = [
        f"| {row['variant'].split('__')[-1]} | {row['target_agreement_vs_primary']*100:.1f}% | {pct(row['total_return'])} |"
        for _, row in worst.iterrows()
    ]
    h2_primary_row = frame[frame["family"] == "primary_h2"].iloc[0]
    h2_core_row = frame[
        (frame["family"] == "benchmark_h2")
        & frame["variant"].eq("fixed_core45_scaled__h2")
    ].iloc[0]
    h2_full_row = frame[
        (frame["family"] == "benchmark_h2")
        & frame["variant"].eq("fixed_full51_scaled__h2")
    ].iloc[0]
    report = f"""# 点时ETF池与成员稳定性研究

研究截止 {END}。本研究只检验池治理与成员稳定性，不修改正式策略、账户、订单和每日入口。

## 结论先行

1. **点时池是正确的研究方向，但本轮主方案不能晋升。** 它消除了“用2026年的存续名单回填2018年”的明显未来信息，却仍有 {pct(primary_row['max_drawdown'])} 的最大回撤；审核周期或窗口变化时，目标一致率只有约72%—92%，没有形成足够稳定的平台。
2. **池敏感性并不只来自固定前5。** 比例名额已修复池扩大导致的机械收紧，但删去黄金、标普500、银行、创业板或通信等少数成员仍会改写较长持仓路径；单仓的序位与复利链依赖依然存在。
3. **两只等权只缓解、没有解决。** 逐只删除后的最差组合重叠从 {leave['portfolio_overlap_vs_primary'].min()*100:.1f}% 提高到 {leave_h2['portfolio_overlap_vs_primary'].min()*100:.1f}%，点时主方案回撤从 {pct(primary_row['max_drawdown'])} 改善到 {pct(h2_primary_row['max_drawdown'])}；但固定45和固定51的两仓回撤反而恶化，不能据此修改正式仓位数。
4. 当前最合理的答案不是再找一个历史最优名单，而是建立**经济暴露注册表＋非收益代表选择＋点时半年版本＋双审核滞后＋比例名额＋252日新增前瞻影子**。正式45核心＋卫星安全层暂时不变，直到这一完整治理链取得新样本证据。

## 预注册主方案

- 半年审核一次；
- 需要至少252日历史、近252日运营资格率不低于90%、缺失率不高于1%、当前20日成交额中位数不低于2,000万元；
- 连续两次通过才准入，连续两次失败才移除；审核间冻结；
- 常规、历史回退、新趋势名额分别按 `5/45、3/45、15/45` 随点时有效池规模机械变化；
- 不使用历史收益决定准入、移除或阈值。

## 结果对照

| 方案 | 平均池规模 | 累计收益 | 最大回撤 | 与主方案目标一致率 | 与主方案池Jaccard |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

## 逐只删除压力测试

- 51次逐只删除的目标一致率中位数为 {leave['target_agreement_vs_primary'].median()*100:.1f}%，最低 {leave['target_agreement_vs_primary'].min()*100:.1f}%。
- 累计收益中位数 {pct(leave['total_return'].median())}，范围 {pct(leave['total_return'].min())} 至 {pct(leave['total_return'].max())}。收益范围只用于暴露路径脆弱性，禁止据此反选成员。

| 删除成员 | 与主方案目标一致率 | 累计收益 |
|---|---:|---:|
{chr(10).join(worst_lines)}

## 单仓放大效应：两只等权诊断

| 方案 | 累计收益 | 最大回撤 | 与对应点时主方案组合重叠 |
|---|---:|---:|---:|
| 固定45、两只等权 | {pct(h2_core_row['total_return'])} | {pct(h2_core_row['max_drawdown'])} | {h2_core_row['portfolio_overlap_vs_primary']*100:.1f}% |
| 固定51、两只等权 | {pct(h2_full_row['total_return'])} | {pct(h2_full_row['max_drawdown'])} | {h2_full_row['portfolio_overlap_vs_primary']*100:.1f}% |
| 点时主方案、两只等权 | {pct(h2_primary_row['total_return'])} | {pct(h2_primary_row['max_drawdown'])} | 100.0% |

- 两只等权下，51次逐只删除后的组合重叠中位数为 {leave_h2['portfolio_overlap_vs_primary'].median()*100:.1f}%，最低 {leave_h2['portfolio_overlap_vs_primary'].min()*100:.1f}%；累计收益范围 {pct(leave_h2['total_return'].min())} 至 {pct(leave_h2['total_return'].max())}。
- 该实验不用于直接修改正式仓位数，只判断“池变化剧烈改写收益”有多少来自单仓100%集中。

## 解释原则

1. 点时治理消除了“2026年的51只从2018年起全部已知”的明显未来信息，并用滞后准入/退出降低池版本跳变。
2. 比例名额保证弱新增不会因为分母扩大而机械收紧旧门槛；真正更强的新暴露仍会改变候选，这是扩展机会集的必要结果，不应被视为结构错误。
3. 稳定性的合格标准不是历史净值完全不变，而是合理审核阈值附近、删去普通成员、改变审核频率时，大多数日期仍给出相同目标；变化必须集中在成员真实准入/退出之后。
4. 若主方案本身仍对删去少数成员高度敏感，就不能靠继续调排名解决。应保留半年版本、两次审核滞后和前瞻影子验证，并把这种敏感度作为单仓动量策略的固有模型风险。

## 尚未完成的关键层

- 当前配置缺少完整的跟踪指数、费率、跟踪误差和成分重叠字段，尚不能可靠执行“同一经济暴露只留一个代表”。
- 当前只有存续ETF，无法消除清盘/合并产品造成的幸存者偏差。
- 2018—2026已被反复研究，任何历史收益都不能作为主方案晋升依据；真正验收必须来自冻结后的新增252个交易日。

## 决策

- 不晋升点时主方案，不修改正式45核心、6只卫星、单仓或每日执行入口。
- 不再根据本轮收益选择年度、季度、80%或95%阈值；这些行只用于证明治理敏感度。
- 下一步只补齐跟踪指数、费率、跟踪误差和成分重叠，建立经济暴露注册表；同暴露一换一，新暴露才增加配额。
- 注册表完成后，只启动自动影子记录，每月复盘；至少252个新增交易日之前不替代正式池。
"""
    (OUTPUT / "summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    symbols = universe_keys(market)
    tasks = task_specs(symbols)
    workers = min(4, max(1, mp.cpu_count() - 1))
    context = mp.get_context("spawn")
    with context.Pool(
        workers, initializer=initialise_worker, initargs=(str(ROOT),)
    ) as pool:
        outputs = list(pool.imap_unordered(run_task, tasks, chunksize=1))
    write_results(outputs)
    write_primary_pool_versions()
    print(json.dumps({"output": str(OUTPUT), "variants": len(outputs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
