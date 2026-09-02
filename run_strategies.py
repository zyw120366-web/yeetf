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
    rank = decision["entry_rank"]
    base_pass = (
        ye_eligibility
        & rank.le(int(ye_config["rules"]["entry_rank_limit"]))
        & ye_features.roc_short.gt(0.0)
        & ye_features.roc_medium.gt(0.0)
        & ye_features.above_ma
        & ye_features.ma_bias.le(float(ye_config["rules"]["max_entry_ma_bias"]))
    )
    technical_entry_pass = decision["entry_gate"] & ye_eligibility
    core_entry_available = technical_entry_pass[decision["core_symbols"]].any(axis=1)
    priority_entry_pass = technical_entry_pass.copy()
    if decision["challenger_symbols"]:
        priority_entry_pass.loc[:, decision["challenger_symbols"]] &= (
            ~core_entry_available.to_numpy()[:, None]
        )
    latest_ranking = pd.DataFrame({
        "date": str(latest.date()),
        "symbol": symbols,
        "name": [names[symbol] for symbol in symbols],
        "category": [categories[symbol] for symbol in symbols],
        "close": panel["close"].loc[latest, symbols].to_numpy(),
        "change_1d": panel["close"][symbols].pct_change(fill_method=None).loc[latest].to_numpy(),
        "trailing_amount_20d": trailing_amount.loc[latest, symbols].to_numpy(),
        "listed_sessions": listed_sessions.loc[latest, symbols].to_numpy(),
        "roc20": ye_features.roc_short.loc[latest].to_numpy(),
        "roc60": ye_features.roc_medium.loc[latest].to_numpy(),
        "momentum_score": ye_features.raw_score.loc[latest].to_numpy(),
        "selection_score": decision["entry_score"].loc[latest].to_numpy(),
        "ma120": ye_features.moving_average.loc[latest].to_numpy(),
        "ma120_bias": ye_features.ma_bias.loc[latest].to_numpy(),
        "above_ma120": ye_features.above_ma.loc[latest].to_numpy(),
        "rank": rank.loc[latest].to_numpy(),
        "rank_5d_ago": rank.shift(int(ye_config["rules"]["rank_change_short_days"])).loc[latest].to_numpy(),
        "rank_20d_ago": rank.shift(int(ye_config["rules"]["rank_change_long_days"])).loc[latest].to_numpy(),
        "dual_rank_decline": decision["dual_rank_decline"].loc[latest].to_numpy(),
        "pool_role": [
            "challenger" if symbol in decision["challenger_symbols"] else "core"
            for symbol in symbols
        ],
        "pool_eligible": ye_eligibility.loc[latest].to_numpy(),
        "base_entry_pass": base_pass.loc[latest].to_numpy(),
        "technical_entry_pass": technical_entry_pass.loc[latest].to_numpy(),
        "final_entry_pass": priority_entry_pass.loc[latest].to_numpy(),
        "normal_entry": decision["normal"].loc[latest].to_numpy(),
        "confirmed_normal_entry": decision["current_normal"].loc[latest].to_numpy(),
        "weak_edge_confirmed": decision["weak_edge_confirmed"].loc[latest].to_numpy(),
        "historical_fallback": decision["fallback"].loc[latest].to_numpy(),
        "emerging_entry": decision["emerging"].loc[latest].to_numpy(),
        "quality_extension": decision["quality_extension"].loc[latest].to_numpy(),
        "r2_20": decision["r2_20"].loc[latest].to_numpy(),
        "efficiency20": decision["efficiency20"].loc[latest].to_numpy(),
        "category_breadth": decision["category_breadth"].loc[latest].to_numpy(),
        "sentiment_matched_count": sentiment["matched_count"].loc[latest].to_numpy(),
        "sentiment_hot_score": sentiment["hot_score"].loc[latest].to_numpy(),
        "sentiment_count_acceleration": sentiment["count_acceleration"].loc[latest].to_numpy(),
        "sentiment_positive_dde_share": sentiment["positive_dde_share"].loc[latest].to_numpy(),
        "soft_exit_confirmation": decision["soft_exit_confirmation"].loc[latest].to_numpy(),
        "hot_exit_protection": decision["hot_exit_protection"].loc[latest].to_numpy(),
        "missing_data_soft_exit_protection": decision["missing_data_soft_exit_protection"].loc[latest].to_numpy(),
        "target_weight": ye_bundle.weights.loc[latest, symbols].to_numpy(),
    }).sort_values(["rank", "momentum_score"], ascending=[True, False], na_position="last")
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
