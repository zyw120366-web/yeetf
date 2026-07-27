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
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals


FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
OUTPUT = ROOT / "results" / "research" / "universe_architecture_study"


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


def premium_sensitive(symbols: list[str]) -> list[str]:
    return [
        symbol for symbol in symbols
        if symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"
    ]


def cash_management(config: dict) -> dict:
    cash = config["cash_management"]
    return {
        "annual_rate": float(cash["historical_backtest_annual_rate"]),
        "fee_rate": float(cash["fee_rate"]),
        "minimum_order": float(cash["minimum_order"]),
        "order_lot": float(cash["order_lot"]),
    }


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    formal = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    all_symbols = universe_keys(market)
    core_size = int(formal["enhanced_selection"]["universe_architecture"]["core_pool_size"])
    core_symbols = all_symbols[:core_size]
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment, available = load_sentiment_matrices(FEATURES, panel["close"].index, all_symbols)
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    capital = float(market["project"]["initial_capital"])

    variants = {
        "core_45": (core_symbols, "core_anchor_challenger", []),
        "global_rank_51": (all_symbols, "fixed_pool", []),
        "anchor_45_plus_6": (
            all_symbols,
            "core_anchor_challenger",
            formal["enhanced_selection"]["universe_architecture"]["challenger_symbols"],
        ),
    }
    rows = []
    date_checks: dict[str, dict] = {}
    for key, (symbols, mode, challengers) in variants.items():
        config = copy.deepcopy(formal)
        architecture = config["enhanced_selection"]["universe_architecture"]
        architecture["mode"] = mode
        architecture["core_pool_size"] = core_size
        architecture["challenger_symbols"] = list(challengers)
        bundle, _, eligibility, _, _, decision = build_ye_signals(
            panel,
            symbols,
            {symbol: categories[symbol] for symbol in symbols},
            config,
            {name: frame[symbols] for name, frame in sentiment.items()},
            available,
        )
        project = execution_project(
            market,
            premium_sensitive(symbols),
            eligibility.shift(1, fill_value=False).astype(bool),
        )
        result = run_backtest(
            key,
            panel,
            bundle.weights,
            start,
            end,
            project,
            cash_management=cash_management(config),
        )
        recent = period_metrics(result.equity, "2025-01-01", end, capital)
        rows.append({
            "variant": key,
            **result.metrics,
            "return_2025_2026": recent["total_return"],
            "max_drawdown_2025_2026": recent["max_drawdown"],
        })
        for date in ("2026-04-07", "2026-04-28"):
            timestamp = pd.Timestamp(date)
            if timestamp not in decision["entry_rank"].index:
                continue
            date_checks.setdefault(date, {})[key] = {
                "communication_rank": clean(decision["entry_rank"].loc[timestamp, "515050.SH"]),
                "communication_entry_gate": bool(decision["entry_gate"].loc[timestamp, "515050.SH"]),
            }

    metrics = pd.DataFrame(rows)
    payload = {
        "status": "research_evidence_for_user_approved_change",
        "generated_through": end,
        "same_data_cost_and_execution": True,
        "variants": clean(metrics.to_dict(orient="records")),
        "communication_case": date_checks,
        "decision": "采用45只核心锚定＋6只挑战者；拒绝51只全局混排。",
        "reason": "新增ETF只能增加可选机会，不能因自身尚不可买而改变核心ETF的排名、类别宽度或退出。",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    labels = {
        "core_45": "原45只核心池",
        "global_rank_51": "51只全局混排（拒绝）",
        "anchor_45_plus_6": "45核心＋6挑战者（正式）",
    }
    lines = [
        "# ETF池排名架构对照",
        "",
        "三组使用同一价格、资讯、成本和次日开盘执行。测试目的不是挑最高收益，而是验证扩池不会让不可买的新ETF破坏原有核心信号。",
        "",
        "| 架构 | 累计收益 | 年化 | 最大回撤 | 交易数 | 2025—2026收益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        lines.append(
            f"| {labels[row['variant']]} | {row['total_return']:.2%} | {row['cagr']:.2%} | "
            f"{row['max_drawdown']:.2%} | {row['trade_count']:.0f} | {row['return_2025_2026']:.2%} |"
        )
    lines.extend([
        "",
        "结论：拒绝把51只ETF直接放进同一个前五排名。正式架构保留原45只的相对排名、类别宽度和排名退出，6只挑战者只有在完整满足入场资格后，才按相对核心池的虚拟排名竞争。",
        "",
        "这是一项排名语义修复，不是根据哪组历史收益最高来选参数；挑战者造成的绩效变化必须来自它自身真正成交。",
        "",
    ])
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
