from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.execution import execution_project
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals
from opportunity_cost_switch_study import (
    SwitchRule,
    build_state_machine,
    forward_event_returns,
    premium_sensitive,
)


FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
OUTPUT = ROOT / "results" / "research" / "switch_parameter_sensitivity"
BASE = {"rank": 5, "margin": 0.05, "confirm": 2, "hold": 5}
TESTS = {
    "rank": [3, 5, 7, 10],
    "margin": [0.03, 0.05, 0.08, 0.10],
    "confirm": [1, 2, 3],
    "hold": [3, 5, 10],
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def path_hash(weights: pd.DataFrame) -> str:
    selected = weights.idxmax(axis=1).where(weights.max(axis=1).gt(0), "CASH")
    return hashlib.sha256("|".join(selected.astype(str)).encode()).hexdigest()[:12]


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    config = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment, available = load_sentiment_matrices(
        FEATURES, panel["close"].index, symbols
    )
    official, features, eligibility, _, _, decision = build_ye_signals(
        panel, symbols, categories, config, sentiment, available
    )
    cooldown = int(config["enhanced_selection"]["reentry_cooldown_days"])
    baseline_weights, _ = build_state_machine(
        symbols,
        features,
        eligibility,
        decision,
        cooldown,
        SwitchRule("现行策略", None),
    )
    if float((baseline_weights - official.weights).abs().max().max()) > 1e-12:
        raise RuntimeError("baseline reconstruction failed")
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    cash_cfg = config["cash_management"]
    cash = {
        "annual_rate": float(cash_cfg["historical_backtest_annual_rate"]),
        "fee_rate": float(cash_cfg["fee_rate"]),
        "minimum_order": float(cash_cfg["minimum_order"]),
        "order_lot": float(cash_cfg["order_lot"]),
    }
    project = execution_project(
        market,
        premium_sensitive(symbols),
        eligibility.shift(1, fill_value=False).astype(bool),
    )
    baseline = run_backtest(
        "现行策略",
        panel,
        baseline_weights,
        start,
        end,
        project,
        cash_management=cash,
    )

    rows = []
    all_events = []
    for gate_name, switch_gate in {
        "formal_complete": decision["entry_gate"],
        "uniform_price_proxy": decision["normal"],
    }.items():
        for dimension, values in TESTS.items():
            for value in values:
                params = dict(BASE)
                params[dimension] = value
                rule = SwitchRule(
                    f"{dimension}={value}",
                    float(params["margin"]),
                    int(params["rank"]),
                    int(params["confirm"]),
                    int(params["hold"]),
                )
                weights, events = build_state_machine(
                    symbols,
                    features,
                    eligibility,
                    decision,
                    cooldown,
                    rule,
                    switch_entry_gate=switch_gate,
                )
                result = run_backtest(
                    rule.label,
                    panel,
                    weights,
                    start,
                    end,
                    project,
                    cash_management=cash,
                )
                events = forward_event_returns(events, panel["open"][symbols])
                if not events.empty:
                    events.insert(0, "gate", gate_name)
                    events.insert(1, "dimension", dimension)
                    events.insert(2, "value", value)
                    all_events.append(events)
                completed = (
                    events["new_minus_old_20d"].dropna()
                    if "new_minus_old_20d" in events
                    else pd.Series(dtype=float)
                )
                positive = completed.clip(lower=0)
                largest_share = (
                    float(positive.max() / positive.sum())
                    if len(positive) and positive.sum() > 0
                    else np.nan
                )
                rows.append(
                    {
                        "gate": gate_name,
                        "dimension": dimension,
                        "value": value,
                        "rank_cutoff": params["rank"],
                        "score_margin": params["margin"],
                        "confirmation_days": params["confirm"],
                        "minimum_hold_days": params["hold"],
                        "path_hash": path_hash(weights),
                        "event_count": int(len(events)),
                        "completed_20d_count": int(len(completed)),
                        "event_years": (
                            int(pd.to_datetime(events["signal_date"]).dt.year.nunique())
                            if len(events)
                            else 0
                        ),
                        "hit_rate_20d": (
                            float((completed > 0).mean()) if len(completed) else np.nan
                        ),
                        "median_edge_20d": (
                            float(completed.median()) if len(completed) else np.nan
                        ),
                        "largest_positive_edge_share": largest_share,
                        "total_return": result.metrics["total_return"],
                        "cagr": result.metrics["cagr"],
                        "sharpe": result.metrics["sharpe"],
                        "max_drawdown": result.metrics["max_drawdown"],
                        "trade_count": result.metrics["trade_count"],
                        "delta_total_return": result.metrics["total_return"]
                        - baseline.metrics["total_return"],
                        "delta_sharpe": result.metrics["sharpe"]
                        - baseline.metrics["sharpe"],
                        "delta_max_drawdown": result.metrics["max_drawdown"]
                        - baseline.metrics["max_drawdown"],
                    }
                )

    metrics = pd.DataFrame(rows)
    events_frame = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    events_frame.to_csv(OUTPUT / "events.csv", index=False, encoding="utf-8-sig")

    base_rows = metrics[
        ((metrics["dimension"] == "rank") & (metrics["value"] == BASE["rank"]))
        | ((metrics["dimension"] == "margin") & (metrics["value"] == BASE["margin"]))
        | ((metrics["dimension"] == "confirm") & (metrics["value"] == BASE["confirm"]))
        | ((metrics["dimension"] == "hold") & (metrics["value"] == BASE["hold"]))
    ]
    base_consistent = bool(base_rows.groupby("gate")["path_hash"].nunique().eq(1).all())
    payload = {
        "status": "research_only",
        "generated_through": end,
        "formal_strategy_changed": False,
        "base_rule": BASE,
        "one_factor_only": True,
        "base_path_consistent_across_repeated_rows": base_consistent,
        "selection_principle": "优先选择有清晰业务含义且处于相邻结果平台的数值，不按最高累计收益选择。",
        "baseline": baseline.metrics,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    lines = [
        "# 换仓参数敏感性研究",
        "",
        "固定规则结构，只分别改变一个数字；正式完整门槛与统一价格代理门槛同时报告。",
        "",
    ]
    for gate in ("formal_complete", "uniform_price_proxy"):
        lines.extend(
            [
                f"## {gate}",
                "",
                "| 改动项 | 数值 | 事件 | 年份 | 20日胜率 | 中位优势 | 累计收益 | 夏普 | 最大回撤 | 路径 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in metrics[metrics["gate"] == gate].itertuples(index=False):
            lines.append(
                f"| {row.dimension} | {row.value:g} | {int(row.completed_20d_count)} | "
                f"{int(row.event_years)} | "
                f"{row.hit_rate_20d:.1%} | {row.median_edge_20d:+.2%} | "
                f"{row.total_return:.2%} | {row.sharpe:.2f} | "
                f"{row.max_drawdown:.2%} | `{row.path_hash}` |"
            )
        lines.append("")
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
