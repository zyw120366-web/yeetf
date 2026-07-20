"""Evaluate the frozen ye signals with idle cash earning a stated annual yield.

This is an independent research script.  It never changes the formal signal,
ETF universe, costs, or daily-order workflow.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.execution import entry_eligibility, execution_project, period_metrics
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals


STUDY_ID = "H-2026-07-20-01"
CASH_ANNUAL_RATE = 0.015


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def metrics_view(result) -> dict[str, float]:
    keys = (
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "annual_turnover",
        "trade_count",
        "average_exposure",
        "total_fees",
        "slippage_cost_estimate",
    )
    return {key: float(result.metrics[key]) for key in keys}


def period_ranges(end: str) -> dict[str, tuple[str, str]]:
    return {
        "2018—2020": ("2018-07-02", "2020-12-31"),
        "2021—2022": ("2021-01-01", "2022-12-31"),
        "2023—2024": ("2023-01-01", "2024-12-31"),
        "2025—2026": ("2025-01-01", end),
    }


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    config = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment, available = load_sentiment_matrices(
        ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv",
        panel["close"].index,
        symbols,
    )
    bundle, _, eligibility, _, _, _ = build_ye_signals(
        panel, symbols, categories, config, sentiment, available
    )
    project = execution_project(
        market,
        [symbol for symbol in symbols if symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"],
        eligibility.shift(1, fill_value=False).astype(bool),
    )
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    capital = float(market["project"]["initial_capital"])
    baseline = run_backtest("ye 策略｜现金不计息", panel, bundle.weights, start, end, project)
    managed = run_backtest(
        "ye 策略｜收盘宝现金管理",
        panel,
        bundle.weights,
        start,
        end,
        project,
        cash_management={
            "annual_rate": CASH_ANNUAL_RATE,
            "fee_rate": float(config["cash_management"]["fee_rate"]),
            "minimum_order": float(config["cash_management"]["minimum_order"]),
            "order_lot": float(config["cash_management"]["order_lot"]),
        },
    )
    frozen = config["validation"]
    if not np.isclose(managed.metrics["total_return"], float(frozen["total_return"]), atol=1e-10):
        raise RuntimeError("cash-managed result no longer matches the frozen formal result")

    base_metrics = metrics_view(baseline)
    managed_metrics = metrics_view(managed)
    delta = {
        "final_equity": float(managed.equity.iloc[-1] - baseline.equity.iloc[-1]),
        "total_return_pct_points": 100 * (managed_metrics["total_return"] - base_metrics["total_return"]),
        "cagr_pct_points": 100 * (managed_metrics["cagr"] - base_metrics["cagr"]),
        "max_drawdown_pct_points": 100 * (managed_metrics["max_drawdown"] - base_metrics["max_drawdown"]),
        "sharpe": managed_metrics["sharpe"] - base_metrics["sharpe"],
        "trade_count": managed_metrics["trade_count"] - base_metrics["trade_count"],
    }
    periods = {}
    for label, (period_start, period_end) in period_ranges(end).items():
        base_period = period_metrics(baseline.equity, period_start, period_end, capital)
        managed_period = period_metrics(managed.equity, period_start, period_end, capital)
        periods[label] = {
            "baseline_total_return": base_period["total_return"],
            "cash_managed_total_return": managed_period["total_return"],
            "difference_pct_points": 100 * (managed_period["total_return"] - base_period["total_return"]),
        }

    cash_weight = 1.0 - baseline.actual_weights.sum(axis=1)
    payload = {
        "study_id": STUDY_ID,
        "generated_on": date.today().isoformat(),
        "status": "incorporated_formal_backtest",
        "formal_strategy_unchanged": False,
        "assumption": {
            "product": "收盘宝（用户提供）",
            "annualized_yield": CASH_ANNUAL_RATE,
            "accrual": "Actual/365 simple interest on eligible idle cash from each trading close to the next trading open, including weekends and holidays",
            "fee_rate": float(config["cash_management"]["fee_rate"]),
            "minimum_order": float(config["cash_management"]["minimum_order"]),
            "order_lot": float(config["cash_management"]["order_lot"]),
        },
        "data": {"start": start, "end": end, "initial_capital": capital},
        "baseline": base_metrics,
        "cash_managed": {
            **managed_metrics,
            "cash_interest_income": managed.metrics["cash_interest_income"],
            "cash_management_fees": managed.metrics["cash_management_fees"],
            "cash_interest_calendar_days": managed.metrics["cash_interest_calendar_days"],
            "average_close_cash_weight": float(cash_weight.mean()),
        },
        "difference": delta,
        "periods": periods,
        "conclusion": "现金管理已纳入正式回测口径，只提高闲置资金收益，不改变任何ETF买卖信号；实盘仍以每日实际下单年化和券商回单为准。",
    }
    output_dir = ROOT / "results" / "research"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "cash_management_study.json"
    output.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean({"output": str(output), "difference": delta}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
