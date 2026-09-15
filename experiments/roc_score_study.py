from __future__ import annotations

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
from etf_rotation.evaluation import round_trip_timing, timing_summary
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals


STUDY = Path(__file__).with_suffix(".yaml")
OUTPUT_DIR = ROOT / "results" / "research" / "roc_score_study"
SENTIMENT_FEATURES = (
    ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def premium_sensitive(symbols: list[str]) -> list[str]:
    return [
        symbol
        for symbol in symbols
        if symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"
    ]


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def circular_block_pvalue(
    difference: np.ndarray,
    block_days: int,
    draws: int,
    rng: np.random.Generator,
) -> float:
    values = difference[np.isfinite(difference)]
    observed = float(values.mean())
    if len(values) < block_days or observed <= 0:
        return 1.0
    centered = values - observed
    length = len(centered)
    blocks = int(np.ceil(length / block_days))
    exceed = 0
    for _ in range(draws):
        starts = rng.integers(0, length, size=blocks)
        sampled = np.concatenate(
            [centered[(start + np.arange(block_days)) % length] for start in starts]
        )[:length]
        exceed += float(sampled.mean()) >= observed
    return float((exceed + 1) / (draws + 1))


def failure_rate_pvalue(
    baseline: np.ndarray,
    candidate: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> float:
    base = baseline[np.isfinite(baseline)].astype(float)
    test = candidate[np.isfinite(candidate)].astype(float)
    observed = float(base.mean() - test.mean())
    if len(base) == 0 or len(test) == 0 or observed <= 0:
        return 1.0
    pooled_rate = float(np.concatenate([base, test]).mean())
    simulated_base = rng.binomial(1, pooled_rate, size=(draws, len(base))).mean(axis=1)
    simulated_test = rng.binomial(1, pooled_rate, size=(draws, len(test))).mean(axis=1)
    return float(((simulated_base - simulated_test >= observed).sum() + 1) / (draws + 1))


def make_scores(close: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
    roc20 = close.pct_change(20, fill_method=None)
    roc60 = close.pct_change(60, fill_method=None)
    log20 = np.log1p(roc20)
    log60 = np.log1p(roc60)
    returns = close.pct_change(fill_method=None)
    vol20 = returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(20)
    vol60 = returns.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(60)
    return {
        "raw_w150": None,
        "raw_w100": roc20 + 1.00 * roc60,
        "raw_w125": roc20 + 1.25 * roc60,
        "raw_w175": roc20 + 1.75 * roc60,
        "raw_w200": roc20 + 2.00 * roc60,
        "horizon_normalized": log20 / 20.0 + 1.5 * log60 / 60.0,
        "cross_section_rank": (
            roc20.rank(axis=1, pct=True, method="average")
            + 1.5 * roc60.rank(axis=1, pct=True, method="average")
        ),
        "volatility_adjusted": (
            roc20.div(vol20.where(vol20.gt(0)))
            + 1.5 * roc60.div(vol60.where(vol60.gt(0)))
        ),
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


def main() -> None:
    study = load_yaml(STUDY)
    market = load_yaml(ROOT / "config" / "market.yaml")
    ye_config = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    categories = {
        symbol_key(item): item["category"] for item in market["universe"]
    }
    calendar = panel["close"].index
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    capital = float(market["project"]["initial_capital"])
    sentiment, available = load_sentiment_matrices(
        SENTIMENT_FEATURES, calendar, symbols
    )
    scores = make_scores(panel["close"][symbols])
    candidate_ids = [study["baseline"], *[item["id"] for item in study["candidates"]]]
    statistics = study["statistics"]
    draws = int(statistics["bootstrap_draws"])
    block_days = int(statistics["return_block_days"])
    seed = int(statistics["seed"])

    results = {}
    rows: list[dict] = []
    timing_frames: dict[str, pd.DataFrame] = {}
    for candidate_id in candidate_ids:
        bundle, _, eligibility, _, _, _ = build_ye_signals(
            panel,
            symbols,
            categories,
            ye_config,
            sentiment,
            available,
            raw_score_override=scores[candidate_id],
        )
        project = execution_project(
            market,
            premium_sensitive(symbols),
            eligibility.shift(1, fill_value=False).astype(bool),
        )
        result = run_backtest(candidate_id, panel, bundle.weights, start, end, project)
        timing = round_trip_timing(result, panel, label_end=end)
        timing_values = timing_summary(timing)
        period_2122 = period_metrics(result.equity, "2021-01-01", "2022-12-31", capital)
        period_2526 = period_metrics(result.equity, "2025-01-01", end, capital)
        results[candidate_id] = result
        timing_frames[candidate_id] = timing
        rows.append(
            {
                "candidate": candidate_id,
                **result.metrics,
                "return_2021_2022": period_2122["total_return"],
                "max_drawdown_2021_2022": period_2122["max_drawdown"],
                "return_2025_2026": period_2526["total_return"],
                "max_drawdown_2025_2026": period_2526["max_drawdown"],
                "completed_operations": timing_values["completed_round_trips"],
                "failed_operations": float(timing["failed_operation"].sum()),
                "failed_operation_rate": timing_values["failed_operation_rate"],
                "false_start_rate": timing_values["false_start_rate"],
                "material_premature_exit_rate": timing_values[
                    "material_premature_exit_rate"
                ],
            }
        )

    summary = pd.DataFrame(rows).set_index("candidate")
    baseline_id = study["baseline"]
    baseline_result = results[baseline_id]
    baseline_timing = timing_frames[baseline_id]
    return_pvalues: dict[str, float] = {}
    failure_pvalues: dict[str, float] = {}
    for offset, candidate_id in enumerate(candidate_ids[1:], start=1):
        return_difference = (
            results[candidate_id].daily_returns.reindex(baseline_result.daily_returns.index).fillna(0.0)
            - baseline_result.daily_returns.fillna(0.0)
        ).to_numpy()
        return_pvalues[candidate_id] = circular_block_pvalue(
            return_difference,
            block_days,
            draws,
            np.random.default_rng(seed + offset),
        )
        failure_pvalues[candidate_id] = failure_rate_pvalue(
            baseline_timing["failed_operation"].astype(float).to_numpy(),
            timing_frames[candidate_id]["failed_operation"].astype(float).to_numpy(),
            draws,
            np.random.default_rng(seed + 100 + offset),
        )

    adjusted_returns = holm_adjust(return_pvalues)
    adjusted_failures = holm_adjust(failure_pvalues)
    summary["return_p_raw"] = np.nan
    summary["return_p_holm"] = np.nan
    summary["failure_p_raw"] = np.nan
    summary["failure_p_holm"] = np.nan
    for candidate_id in candidate_ids[1:]:
        summary.loc[candidate_id, "return_p_raw"] = return_pvalues[candidate_id]
        summary.loc[candidate_id, "return_p_holm"] = adjusted_returns[candidate_id]
        summary.loc[candidate_id, "failure_p_raw"] = failure_pvalues[candidate_id]
        summary.loc[candidate_id, "failure_p_holm"] = adjusted_failures[candidate_id]

    reference_summary = json.loads(
        (ROOT / "results" / "etfwin_reference" / "reference_summary.json").read_text(
            encoding="utf-8"
        )
    )
    reference_periods = reference_summary["periods"]
    reference_2526 = next(
        values
        for label, values in reference_periods.items()
        if "2025" in label
    )["total_return"]
    gate = study["promotion_gate"]
    threshold = float(gate["adjusted_p_max"])
    required_2526 = (
        float(gate["period_2025_2026_min_etfwin_fraction"]) * reference_2526
    )
    base_drawdown = float(summary.loc[baseline_id, "max_drawdown"])
    max_dd_loss = float(gate["max_drawdown_deterioration_points"]) / 100.0

    weight_ids = ["raw_w100", "raw_w125", "raw_w150", "raw_w175", "raw_w200"]
    local_values = summary.loc[weight_ids]
    best_local_id = local_values["cagr"].idxmax()
    local_peak_is_interior = best_local_id not in {"raw_w100", "raw_w200"}
    local_tolerance = float(gate["local_cagr_tolerance_points"]) / 100.0
    local_platform = bool(
        local_peak_is_interior
        and abs(
            local_values.loc["raw_w125", "cagr"]
            - local_values.loc["raw_w150", "cagr"]
        )
        <= local_tolerance
        and abs(
            local_values.loc["raw_w175", "cagr"]
            - local_values.loc["raw_w150", "cagr"]
        )
        <= local_tolerance
    )

    decisions: dict[str, dict] = {}
    for candidate_id in candidate_ids[1:]:
        row = summary.loc[candidate_id]
        checks = {
            "daily_return_superiority": bool(row["return_p_holm"] <= threshold),
            "failure_rate_superiority": bool(row["failure_p_holm"] <= threshold),
            "drawdown_not_worse": bool(row["max_drawdown"] >= base_drawdown - max_dd_loss),
            "2021_2022_nonnegative": bool(
                row["return_2021_2022"]
                >= float(gate["period_2021_2022_min_return"])
            ),
            "2025_2026_reference_floor": bool(row["return_2025_2026"] >= required_2526),
            "local_parameter_platform": bool(
                local_platform if candidate_id.startswith("raw_w") else True
            ),
        }
        decisions[candidate_id] = {
            "promote": all(checks.values()),
            "checks": checks,
        }

    promoted = [name for name, item in decisions.items() if item["promote"]]
    report = {
        "study": study,
        "data_start": start,
        "data_end": end,
        "observations": int(len(baseline_result.daily_returns)),
        "etfwin_2025_2026_return": reference_2526,
        "required_2025_2026_return": required_2526,
        "local_weight_best_cagr": best_local_id,
        "local_peak_is_interior": local_peak_is_interior,
        "local_parameter_platform": local_platform,
        "promoted_candidates": promoted,
        "formal_strategy_changed": False,
        "decision": (
            "No score formula passed every preregistered promotion gate."
            if not promoted
            else "At least one score formula passed every preregistered promotion gate; manual review is required before changing the formal strategy."
        ),
        "candidate_decisions": decisions,
        "summary": summary.reset_index().to_dict(orient="records"),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(
        OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(clean(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(json.dumps(clean({key: report[key] for key in (
        "local_weight_best_cagr",
        "local_peak_is_interior",
        "local_parameter_platform",
        "promoted_candidates",
        "decision",
    )}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
