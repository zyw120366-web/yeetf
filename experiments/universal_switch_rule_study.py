from __future__ import annotations

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
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals
from opportunity_cost_switch_study import (
    SwitchRule,
    build_state_machine,
    forward_event_returns,
    premium_sensitive,
)


FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
OUTPUT = ROOT / "results" / "research" / "universal_switch_rule"


RULES = {
    "rank5_1d": SwitchRule("掉出前5，有更强者即换", 0.0, 5, 1, 5),
    "rank5_2d": SwitchRule("掉出前5，强者持续2日", 0.0, 5, 2, 5),
    "gap3_1d": SwitchRule("前5外＋领先3点，1日", 0.03, 5, 1, 5),
    "gap3_2d": SwitchRule("前5外＋领先3点，2日", 0.03, 5, 2, 5),
    "gap5_1d": SwitchRule("前5外＋领先5点，1日", 0.05, 5, 1, 5),
    "gap5_2d": SwitchRule("前5外＋领先5点，2日", 0.05, 5, 2, 5),
    "gap10_1d": SwitchRule("前5外＋领先10点，1日", 0.10, 5, 1, 5),
    "gap10_2d": SwitchRule("前5外＋领先10点，2日", 0.10, 5, 2, 5),
    "rank8_gap5_1d": SwitchRule("前8外＋领先5点，1日", 0.05, 8, 1, 5),
    "rank8_gap5_2d": SwitchRule("前8外＋领先5点，2日", 0.05, 8, 2, 5),
    "top1_gap5_1d": SwitchRule("仅榜首领先5点，1日", 0.05, 5, 1, 5),
    "top1_gap5_2d": SwitchRule("仅榜首领先5点，2日", 0.05, 5, 2, 5),
}


PERIODS = {
    "2018_2020": ("2018-07-02", "2020-12-31"),
    "2021_2022": ("2021-01-01", "2022-12-31"),
    "2023_2024": ("2023-01-01", "2024-12-31"),
    "2025_2026": ("2025-01-01", "2026-07-29"),
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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

    normal_gate = decision["normal"]
    top1_gate = normal_gate & decision["entry_rank"].le(1)
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    capital = float(market["project"]["initial_capital"])
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
    period_rows = []
    all_events = []
    equities = {"baseline": baseline.equity}
    for key, rule in RULES.items():
        gate = top1_gate if key.startswith("top1") else normal_gate
        weights, events = build_state_machine(
            symbols,
            features,
            eligibility,
            decision,
            cooldown,
            rule,
            switch_entry_gate=gate,
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
        events.insert(0, "variant", key)
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
        years = (
            int(pd.to_datetime(events["signal_date"]).dt.year.nunique())
            if len(events) and "signal_date" in events
            else 0
        )
        hit_rate = float((completed > 0).mean()) if len(completed) else np.nan
        median_edge = float(completed.median()) if len(completed) else np.nan
        criteria = {
            "events": len(completed) >= 8,
            "years": years >= 3,
            "hit_rate": hit_rate >= 0.55 if np.isfinite(hit_rate) else False,
            "median_edge": median_edge > 0 if np.isfinite(median_edge) else False,
            "return": result.metrics["total_return"] > baseline.metrics["total_return"],
            "sharpe": result.metrics["sharpe"] > baseline.metrics["sharpe"],
            "drawdown": result.metrics["max_drawdown"] >= baseline.metrics["max_drawdown"] - 0.02,
            "concentration": largest_share <= 0.50 if np.isfinite(largest_share) else False,
        }
        rows.append(
            {
                "variant": key,
                "label": rule.label,
                "event_count": int(len(events)),
                "completed_20d_count": int(len(completed)),
                "event_years": years,
                "hit_rate_20d": hit_rate,
                "median_edge_20d": median_edge,
                "mean_edge_20d": float(completed.mean()) if len(completed) else np.nan,
                "largest_positive_edge_share": largest_share,
                **result.metrics,
                "delta_total_return": result.metrics["total_return"] - baseline.metrics["total_return"],
                "delta_sharpe": result.metrics["sharpe"] - baseline.metrics["sharpe"],
                "delta_max_drawdown": result.metrics["max_drawdown"] - baseline.metrics["max_drawdown"],
                "criteria_passed": int(sum(criteria.values())),
                "universal_pass": bool(all(criteria.values())),
            }
        )
        for period, (period_start, period_end) in PERIODS.items():
            candidate_period = period_metrics(
                result.equity, period_start, period_end, capital
            )
            baseline_period = period_metrics(
                baseline.equity, period_start, period_end, capital
            )
            period_rows.append(
                {
                    "variant": key,
                    "period": period,
                    "total_return": candidate_period["total_return"],
                    "delta_total_return": candidate_period["total_return"]
                    - baseline_period["total_return"],
                    "max_drawdown": candidate_period["max_drawdown"],
                }
            )
        equities[key] = result.equity

    metrics = pd.DataFrame(rows).sort_values(
        ["universal_pass", "criteria_passed", "event_count"],
        ascending=[False, False, False],
    )
    events_frame = pd.concat(all_events, ignore_index=True)
    periods = pd.DataFrame(period_rows)
    universal = metrics[metrics["universal_pass"]]
    best = metrics.iloc[0]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    events_frame.to_csv(OUTPUT / "events.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(OUTPUT / "period_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(equities).to_csv(OUTPUT / "equity.csv", encoding="utf-8-sig")

    payload = {
        "status": "research_only",
        "generated_through": end,
        "formal_strategy_changed": False,
        "uniform_historical_switch_gate": "常规价格条件：核心前5、双ROC为正、MA120上方、乖离不超过9%",
        "universal_acceptance": {
            "minimum_completed_events": 8,
            "minimum_years": 3,
            "minimum_20d_hit_rate": 0.55,
            "median_20d_edge_positive": True,
            "return_and_sharpe_above_baseline": True,
            "drawdown_not_worse_by_more_than_2pp": True,
            "largest_positive_event_share_at_most": 0.50,
        },
        "passed_variants": universal["variant"].tolist(),
        "best_by_criteria_not_return": best.to_dict(),
        "conclusion": (
            "存在通过全部普适性门槛的简单规则。"
            if len(universal)
            else "没有规则同时通过频率、跨年、胜率、中位优势、回撤和集中度门槛。"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    lines = [
        "# 普适换仓规则研究",
        "",
        "## 结论",
        "",
        (
            f"共有{len(universal)}条规则通过全部普适性门槛。"
            if len(universal)
            else "**没有找到能同时做到‘触发较多、跨年份、多数有效、不是靠单笔暴赚’的简单换仓规则。**"
        ),
        "",
        f"表现最接近普适要求的是“{best['label']}”：完成事件{int(best['completed_20d_count'])}次、覆盖{int(best['event_years'])}个年份、20日胜率{best['hit_rate_20d']:.1%}、中位相对收益{best['median_edge_20d']:+.2%}，通过8项标准中的{int(best['criteria_passed'])}项。",
        "",
        "## 全部规则",
        "",
        "| 规则 | 完成事件 | 年份 | 20日胜率 | 中位优势 | 累计收益 | 夏普 | 最大回撤 | 通过项 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.label} | {int(row.completed_20d_count)} | {int(row.event_years)} | "
            f"{row.hit_rate_20d:.1%} | {row.median_edge_20d:+.2%} | "
            f"{row.total_return:.2%} | {row.sharpe:.2f} | {row.max_drawdown:.2%} | "
            f"{int(row.criteria_passed)}/8 |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 统一使用价格条件，避免2024年前缺少AI资讯造成比较不公平。",
            "- 触发次数多不等于有效；如果胜率和中位优势为负，只是更频繁地追涨。",
            "- 结果不会自动修改正式策略。",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
