from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.backtest import BacktestResult, run_backtest
from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.etfwin import EtfwinRules, etfwin_signals
from etf_rotation.evaluation import realized_round_trips, round_trip_timing, timing_summary
from etf_rotation.execution import entry_eligibility, execution_project, period_metrics
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals


YE_NAME = "ye 策略"
REFERENCE_NAME = "etfwin 策略"
RESULTS = ROOT / "results"
SENTIMENT_FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"


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
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def save_frame(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding="utf-8-sig")


def periods(end: str) -> dict[str, tuple[str, str]]:
    return {
        "全区间": ("2018-07-02", end),
        "2018—2020": ("2018-07-02", "2020-12-31"),
        "2021—2022": ("2021-01-01", "2022-12-31"),
        "2023—2024": ("2023-01-01", "2024-12-31"),
        "2025—2026": ("2025-01-01", end),
    }


def portfolio(weights: pd.Series, names: dict[str, str]) -> list[dict]:
    return [
        {"symbol": str(symbol), "name": names.get(str(symbol), str(symbol)), "weight": float(weight)}
        for symbol, weight in weights[weights.gt(1e-12)].sort_values(ascending=False).items()
    ]


def premium_sensitive(symbols: list[str]) -> list[str]:
    return [
        symbol for symbol in symbols
        if symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"
    ]


def reference_bundle(
    panel: dict[str, pd.DataFrame], official: dict
):
    values = official["rules"]
    symbols = [symbol_key(item) for item in official["universe"]]
    rules = EtfwinRules(
        roc_short_days=int(values["roc_short_days"]),
        roc_medium_days=int(values["roc_medium_days"]),
        roc_short_weight=float(values["roc_short_weight"]),
        roc_medium_weight=float(values["roc_medium_weight"]),
        entry_rank_limit=int(values["entry_rank_limit"]),
        ma_days=int(values["ma_days"]),
        max_entry_ma_bias=float(values["max_entry_ma_bias"]),
        rank_change_short_days=int(values["rank_change_short_days"]),
        rank_change_long_days=int(values["rank_change_long_days"]),
        holdings_num=int(values.get("holdings_num", 1)),
        exit_on_ma_break=bool(values.get("exit_on_ma_break", True)),
        exit_on_short_roc_negative=bool(values.get("exit_on_short_roc_negative", True)),
        exit_on_dual_rank_decline=bool(values.get("exit_on_dual_rank_decline", True)),
    )
    eligibility = pd.DataFrame(True, index=panel["close"].index, columns=symbols)
    bundle, features = etfwin_signals(
        panel["close"], symbols, rules, entry_eligibility=eligibility
    )
    return symbols, bundle, features, eligibility


def metric_rows(
    name: str,
    result: BacktestResult,
    timing: dict[str, float],
    end: str,
    initial_capital: float,
) -> list[dict]:
    rows = []
    for label, (start, stop) in periods(end).items():
        values = result.metrics if label == "全区间" else period_metrics(
            result.equity, start, stop, initial_capital
        )
        rows.append({
            "strategy": name,
            "period": label,
            "start": start,
            "end": stop,
            **values,
            **(timing if label == "全区间" else {}),
        })
    return rows


def annual_rows(
    name: str, result: BacktestResult, start: str, end: str, initial_capital: float
) -> list[dict]:
    rows = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        year_start = start if year == pd.Timestamp(start).year else f"{year}-01-01"
        year_end = end if year == pd.Timestamp(end).year else f"{year}-12-31"
        rows.append({
            "strategy": name,
            "year": year,
            **period_metrics(result.equity, year_start, year_end, initial_capital),
        })
    return rows


def result_summary(
    name: str,
    result: BacktestResult,
    timing: dict[str, float],
    end: str,
    initial_capital: float,
) -> dict:
    return {
        "name": name,
        "generated_through": end,
        "metrics": clean(result.metrics),
        "periods": {
            label: clean(
                result.metrics if label == "全区间" else period_metrics(
                    result.equity, start, stop, initial_capital
                )
            )
            for label, (start, stop) in periods(end).items()
        },
        "timing": clean(timing),
    }


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    ye_config = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    official = load_yaml(ROOT / "config" / "etfwin_official.yaml")
    merged = {
        symbol_key(item): item
        for item in [*market["universe"], *official["universe"]]
    }
    data_market = {**market, "universe": list(merged.values())}
    panel = load_panel(data_market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    names = {symbol_key(item): item["name"] for item in market["universe"]}
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    calendar = panel["close"].index
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    capital = float(market["project"]["initial_capital"])

    sentiment, available = load_sentiment_matrices(
        SENTIMENT_FEATURES, calendar, symbols
    )
    (
        ye_bundle,
        ye_features,
        ye_eligibility,
        listed_sessions,
        trailing_amount,
        decision,
    ) = build_ye_signals(
        panel, symbols, categories, ye_config, sentiment, available
    )
    ye_project = execution_project(
        market,
        premium_sensitive(symbols),
        ye_eligibility.shift(1, fill_value=False).astype(bool),
    )
    cash_config = ye_config["cash_management"]
    ye_result = run_backtest(
        YE_NAME, panel, ye_bundle.weights, start, end, ye_project,
        cash_management={
            "annual_rate": float(cash_config["historical_backtest_annual_rate"]),
            "fee_rate": float(cash_config["fee_rate"]),
            "minimum_order": float(cash_config["minimum_order"]),
            "order_lot": 1000.0,
        },
    )
    ye_timing_frame = round_trip_timing(ye_result, panel, label_end=end)
    ye_timing = timing_summary(ye_timing_frame)

    reference_symbols, reference_signals, _, reference_eligibility = reference_bundle(
        panel, official
    )
    reference_project = execution_project(
        market, premium_sensitive(reference_symbols), reference_eligibility
    )
    reference_result = run_backtest(
        REFERENCE_NAME, panel, reference_signals.weights, start, end, reference_project
    )
    reference_timing_frame = round_trip_timing(reference_result, panel, label_end=end)
    reference_timing = timing_summary(reference_timing_frame)

    ye_dir = RESULTS / "ye_strategy"
    reference_dir = RESULTS / "etfwin_reference"
    comparison_dir = RESULTS / "comparison"
    for folder in (ye_dir, reference_dir, comparison_dir):
        folder.mkdir(parents=True, exist_ok=True)

    ye_result.equity.rename("equity").to_csv(ye_dir / "equity.csv", encoding="utf-8-sig")
    save_frame(ye_result.trades, ye_dir / "trades.csv")
    save_frame(realized_round_trips(ye_result.trades, calendar), ye_dir / "round_trips.csv")
    save_frame(ye_timing_frame, ye_dir / "timing.csv")
    ye_bundle.weights.to_csv(ye_dir / "signal_weights.csv", encoding="utf-8-sig")
    ye_bundle.diagnostics.to_csv(ye_dir / "signal_diagnostics.csv", encoding="utf-8-sig")
    ye_summary = result_summary(YE_NAME, ye_result, ye_timing, end, capital)
    (ye_dir / "summary.json").write_text(
        json.dumps(clean(ye_summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reference_result.equity.rename("equity").to_csv(
        reference_dir / "equity.csv", encoding="utf-8-sig"
    )
    save_frame(reference_result.trades, reference_dir / "trades.csv")
    save_frame(
        realized_round_trips(reference_result.trades, calendar),
        reference_dir / "round_trips.csv",
    )
    reference_summary = {
        **result_summary(REFERENCE_NAME, reference_result, reference_timing, end, capital),
        "status": official["status"],
        "source": official["source"],
        "note": "公开规则的本地量化代理；仅作同数据、同成本、同执行时点对照，不参与ye实盘信号。",
    }
    (reference_dir / "reference_summary.json").write_text(
        json.dumps(clean(reference_summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics = pd.DataFrame([
        *metric_rows(REFERENCE_NAME, reference_result, reference_timing, end, capital),
        *metric_rows(YE_NAME, ye_result, ye_timing, end, capital),
    ])
    annual = pd.DataFrame([
        *annual_rows(REFERENCE_NAME, reference_result, start, end, capital),
        *annual_rows(YE_NAME, ye_result, start, end, capital),
    ])
    timing = pd.DataFrame([
        {"strategy": REFERENCE_NAME, **reference_timing},
        {"strategy": YE_NAME, **ye_timing},
    ])
    equity = pd.DataFrame({
        REFERENCE_NAME: reference_result.equity.reindex(ye_result.equity.index).ffill(),
        YE_NAME: ye_result.equity,
    })
    equity.index.name = "date"
    save_frame(metrics, comparison_dir / "metrics.csv")
    save_frame(annual, comparison_dir / "annual.csv")
    save_frame(timing, comparison_dir / "timing.csv")
    save_frame(equity.reset_index(), comparison_dir / "equity.csv")
    latest = calendar[-1]
    from etf_rotation.ranking import ranking_for_day
    latest_ranking = ranking_for_day(
        panel, symbols, names, categories, ye_config, ye_bundle, ye_features,
        ye_eligibility, listed_sessions, trailing_amount, decision, sentiment, latest,
    )
    save_frame(latest_ranking, comparison_dir / "latest_ranking.csv")

    latest_signals = {
        "signal_date": str(latest.date()),
        "execution_rule": ye_config["execution"],
        "strategy": {
            "name": YE_NAME,
            "status": ye_config["status"],
            "target_portfolio": portfolio(ye_bundle.weights.loc[latest], names),
            "diagnostics": clean(ye_bundle.diagnostics.loc[latest].to_dict()),
            "live_use_allowed": bool(ye_config["validation"]["live_use_allowed"]),
        },
        "reference": {
            "name": REFERENCE_NAME,
            "status": official["status"],
            "target_portfolio": portfolio(
                reference_signals.weights.loc[latest],
                {symbol_key(item): item["name"] for item in official["universe"]},
            ),
            "live_use_allowed": False,
        },
    }
    (comparison_dir / "latest_signals.json").write_text(
        json.dumps(clean(latest_signals), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "data_end": end,
        "ye": ye_summary,
        "etfwin": reference_summary,
        "signal": latest_signals,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
