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
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals


FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
OUTPUT = ROOT / "results" / "research" / "strategy_ablation"
VARIANTS = {
    "price_core": {
        "label": "纯价格核心",
        "components": {
            "weak_edge_filter": False,
            "emerging_trend": False,
            "quality_extension": False,
            "hot_exit_protection": False,
        },
    },
    "core_plus_weak_edge": {
        "label": "核心＋弱边缘确认",
        "components": {
            "weak_edge_filter": True,
            "emerging_trend": False,
            "quality_extension": False,
            "hot_exit_protection": False,
        },
    },
    "core_plus_entry_exceptions": {
        "label": "核心＋弱边缘＋新趋势/质量延伸",
        "components": {
            "weak_edge_filter": True,
            "emerging_trend": True,
            "quality_extension": True,
            "hot_exit_protection": False,
        },
    },
    "full_strategy": {
        "label": "当前完整策略",
        "components": {
            "weak_edge_filter": True,
            "emerging_trend": True,
            "quality_extension": True,
            "hot_exit_protection": True,
        },
    },
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def premium_sensitive(symbols: list[str]) -> list[str]:
    return [
        symbol for symbol in symbols
        if symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"
    ]


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
    market = load_yaml(ROOT / "config" / "market.yaml")
    config = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    calendar = panel["close"].index
    sentiment, available = load_sentiment_matrices(FEATURES, calendar, symbols)
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    capital = float(market["project"]["initial_capital"])
    cash = config["cash_management"]
    cash_management = {
        "annual_rate": float(cash["historical_backtest_annual_rate"]),
        "fee_rate": float(cash["fee_rate"]),
        "minimum_order": float(cash["minimum_order"]),
        "order_lot": float(cash["order_lot"]),
    }

    rows: list[dict] = []
    equity: dict[str, pd.Series] = {}
    for variant, definition in VARIANTS.items():
        bundle, _, eligibility, _, _, _ = build_ye_signals(
            panel,
            symbols,
            categories,
            config,
            sentiment,
            available,
            components=definition["components"],
        )
        project = execution_project(
            market,
            premium_sensitive(symbols),
            eligibility.shift(1, fill_value=False).astype(bool),
        )
        result = run_backtest(
            definition["label"],
            panel,
            bundle.weights,
            start,
            end,
            project,
            cash_management=cash_management,
        )
        post_review = period_metrics(result.equity, "2024-01-01", end, capital)
        rows.append({
            "variant": variant,
            "label": definition["label"],
            **result.metrics,
            "return_since_2024": post_review["total_return"],
            "cagr_since_2024": post_review["cagr"],
            "sharpe_since_2024": post_review["sharpe"],
            "max_drawdown_since_2024": post_review["max_drawdown"],
        })
        equity[variant] = result.equity

    metrics = pd.DataFrame(rows)
    baseline = metrics.set_index("variant").loc["price_core"]
    for key in ("total_return", "cagr", "sharpe", "max_drawdown", "return_since_2024"):
        metrics[f"delta_{key}_vs_price_core"] = metrics[key] - baseline[key]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    equity_frame = pd.DataFrame(equity)
    equity_frame.index.name = "date"
    equity_frame.to_csv(OUTPUT / "equity.csv", encoding="utf-8-sig")
    payload = {
        "status": "research_only",
        "generated_through": end,
        "comparison_start": start,
        "ai_attribution_window": ["2024-01-01", end],
        "same_data_cost_and_execution": True,
        "daily_execution_impact": "none",
        "variants": clean(metrics.to_dict(orient="records")),
        "interpretation_rule": "只比较同日期组件增量；结果不自动修改正式策略。",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table_lines = [
        "| 版本 | 全区间年化 | 2024年以来收益 | 2024年以来最大回撤 |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        table_lines.append(
            f"| {row['label']} | {row['cagr']:.2%} | {row['return_since_2024']:.2%} | "
            f"{row['max_drawdown_since_2024']:.2%} |"
        )
    markdown = "\n".join([
        "# ye 策略组件消融",
        "",
        "四个版本使用同一日期、ETF数据、成本和次日开盘执行模型；2024年前保持相同的历史缺失期回退规则。",
        "",
        *table_lines,
        "",
        "当前样本中，新趋势/质量延伸提供了主要机械增量；弱边缘过滤单独使用降低收益，热点退出保护略降收益并改善回撤。",
        "这些结果仍来自2024年以来同一强行情样本，只用于归因，不构成独立样本证明，也不会自动修改正式策略。",
        "",
    ])
    (OUTPUT / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
