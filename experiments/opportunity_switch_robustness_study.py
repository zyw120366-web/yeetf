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
OUTPUT = ROOT / "results" / "research" / "opportunity_switch_robustness"
CANDIDATE = SwitchRule("10点缓冲＋前5外＋连续2日", 0.10, 5, 2, 5)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cash_management(config: dict) -> dict:
    cash = config["cash_management"]
    return {
        "annual_rate": float(cash["historical_backtest_annual_rate"]),
        "fee_rate": float(cash["fee_rate"]),
        "minimum_order": float(cash["minimum_order"]),
        "order_lot": float(cash["order_lot"]),
    }


def scaled_project(project: dict, multiple: float, capital: float | None = None) -> dict:
    output = copy.deepcopy(project)
    output["commission_rate"] *= multiple
    output["slippage_rate"] *= multiple
    output["minimum_commission"] *= multiple
    if capital is not None:
        output["initial_capital"] = capital
    for values in output["symbol_costs"].values():
        values["commission_rate"] *= multiple
        values["slippage_rate"] *= multiple
        values["minimum_commission"] *= multiple
    return output


def run_one(
    label: str,
    weights: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    start: str,
    end: str,
    project: dict,
    cash: dict,
):
    return run_backtest(
        label,
        panel,
        weights,
        start,
        end,
        project,
        cash_management=cash,
    )


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
    candidate_weights, candidate_events = build_state_machine(
        symbols,
        features,
        eligibility,
        decision,
        cooldown,
        CANDIDATE,
    )
    candidate_events = forward_event_returns(candidate_events, panel["open"][symbols])
    technical_weights, technical_events = build_state_machine(
        symbols,
        features,
        eligibility,
        decision,
        cooldown,
        CANDIDATE,
        switch_entry_gate=decision["normal"],
    )
    technical_events = forward_event_returns(
        technical_events, panel["open"][symbols]
    )
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    cash = cash_management(config)
    project = execution_project(
        market,
        premium_sensitive(symbols),
        eligibility.shift(1, fill_value=False).astype(bool),
    )

    staged50_weights = candidate_weights.copy()
    for event in candidate_events.itertuples(index=False):
        date = pd.Timestamp(event.signal_date)
        staged50_weights.loc[date] = 0.0
        staged50_weights.loc[date, str(event.from_symbol)] = 0.5
        staged50_weights.loc[date, str(event.to_symbol)] = 0.5
    blend_weights = {
        "baseline": baseline_weights,
        "sleeve25": baseline_weights * 0.75 + candidate_weights * 0.25,
        "sleeve50": baseline_weights * 0.50 + candidate_weights * 0.50,
        "sleeve75": baseline_weights * 0.25 + candidate_weights * 0.75,
        "staged50": staged50_weights,
        "full_switch": candidate_weights,
    }
    labels = {
        "baseline": "现行策略",
        "sleeve25": "25%机会成本双袖",
        "sleeve50": "50%机会成本双袖",
        "sleeve75": "75%机会成本双袖",
        "staged50": "首日50%后全仓切换",
        "full_switch": "100%直接换仓",
    }
    results = {
        key: run_one(labels[key], weights, panel, start, end, project, cash)
        for key, weights in blend_weights.items()
    }
    baseline = results["baseline"].metrics
    blend_rows = []
    for key, result in results.items():
        blend_rows.append(
            {
                "variant": key,
                "label": labels[key],
                **result.metrics,
                "delta_total_return": result.metrics["total_return"]
                - baseline["total_return"],
                "delta_max_drawdown": result.metrics["max_drawdown"]
                - baseline["max_drawdown"],
                "delta_trade_count": result.metrics["trade_count"]
                - baseline["trade_count"],
            }
        )
    blend_frame = pd.DataFrame(blend_rows)

    cost_rows = []
    for multiple in (1.0, 2.0, 3.0):
        stressed = scaled_project(project, multiple)
        base_result = run_one(
            f"基线-{multiple:g}倍成本",
            baseline_weights,
            panel,
            start,
            end,
            stressed,
            cash,
        )
        for key in ("sleeve50", "staged50", "full_switch"):
            result = run_one(
                f"{labels[key]}-{multiple:g}倍成本",
                blend_weights[key],
                panel,
                start,
                end,
                stressed,
                cash,
            )
            cost_rows.append(
                {
                    "cost_multiple": multiple,
                    "variant": key,
                    "total_return": result.metrics["total_return"],
                    "delta_total_return": result.metrics["total_return"]
                    - base_result.metrics["total_return"],
                    "max_drawdown": result.metrics["max_drawdown"],
                    "trade_count": result.metrics["trade_count"],
                }
            )
    cost_frame = pd.DataFrame(cost_rows)

    start_rows = []
    for start_year in (2018, 2021, 2023, 2024, 2025):
        window_start = max(pd.Timestamp(start), pd.Timestamp(f"{start_year}-01-01"))
        base_result = run_one(
            f"基线-{start_year}",
            baseline_weights,
            panel,
            str(window_start.date()),
            end,
            project,
            cash,
        )
        for key in ("sleeve50", "staged50", "full_switch"):
            result = run_one(
                f"{labels[key]}-{start_year}",
                blend_weights[key],
                panel,
                str(window_start.date()),
                end,
                project,
                cash,
            )
            start_rows.append(
                {
                    "start_year": start_year,
                    "variant": key,
                    "total_return": result.metrics["total_return"],
                    "delta_total_return": result.metrics["total_return"]
                    - base_result.metrics["total_return"],
                    "max_drawdown": result.metrics["max_drawdown"],
                }
            )
    start_frame = pd.DataFrame(start_rows)

    leave_rows = []
    for event in candidate_events.itertuples(index=False):
        blocked_pair = {(str(event.from_symbol), str(event.to_symbol))}
        blocked_weights, blocked_events = build_state_machine(
            symbols,
            features,
            eligibility,
            decision,
            cooldown,
            CANDIDATE,
            blocked_switch_pairs=blocked_pair,
        )
        result = run_one(
            f"剔除{event.from_symbol}->{event.to_symbol}",
            blocked_weights,
            panel,
            start,
            end,
            project,
            cash,
        )
        leave_rows.append(
            {
                "blocked_pair": f"{event.from_symbol}->{event.to_symbol}",
                "remaining_switches": len(blocked_events),
                "total_return": result.metrics["total_return"],
                "delta_total_return": result.metrics["total_return"]
                - baseline["total_return"],
                "max_drawdown": result.metrics["max_drawdown"],
            }
        )
    leave_frame = pd.DataFrame(leave_rows)

    one_day_rule = SwitchRule("一日机会事件", 0.05, 5, 1, 5)
    _, one_day_events = build_state_machine(
        symbols, features, eligibility, decision, cooldown, one_day_rule
    )
    one_day_events = forward_event_returns(one_day_events, panel["open"][symbols])
    completed_20d = one_day_events["new_minus_old_20d"].dropna()
    event_hit_rate = float((completed_20d > 0).mean()) if len(completed_20d) else np.nan
    event_median_edge = float(completed_20d.median()) if len(completed_20d) else np.nan
    technical_completed_20d = technical_events["new_minus_old_20d"].dropna()
    technical_hit_rate = (
        float((technical_completed_20d > 0).mean())
        if len(technical_completed_20d)
        else np.nan
    )
    technical_median_edge = (
        float(technical_completed_20d.median())
        if len(technical_completed_20d)
        else np.nan
    )

    small_rows = []
    small_project = scaled_project(project, 1.0, 9825.0)
    for key in ("baseline", "sleeve50", "staged50", "full_switch"):
        result = run_one(
            f"{labels[key]}-9825元",
            blend_weights[key],
            panel,
            start,
            end,
            small_project,
            cash,
        )
        small_rows.append(
            {
                "variant": key,
                "total_return": result.metrics["total_return"],
                "max_drawdown": result.metrics["max_drawdown"],
                "trade_count": result.metrics["trade_count"],
                "total_fees": result.metrics["total_fees"],
                "blocked_orders": result.metrics["blocked_order_count"],
            }
        )
    small_frame = pd.DataFrame(small_rows)

    full = blend_frame.set_index("variant").loc["full_switch"]
    half = blend_frame.set_index("variant").loc["sleeve50"]
    staged = blend_frame.set_index("variant").loc["staged50"]
    leave_positive = bool(
        not leave_frame.empty and leave_frame["delta_total_return"].gt(0).all()
    )
    enough_events = bool(
        len(candidate_events) >= 8
        and pd.to_datetime(candidate_events["signal_date"]).dt.year.nunique() >= 3
    )
    cost_stable = bool(
        cost_frame.groupby("variant")["delta_total_return"].min().gt(0).all()
    )
    start_stable = bool(
        start_frame.groupby("variant")["delta_total_return"].min().gt(0).all()
    )
    formal_acceptance = bool(
        leave_positive
        and enough_events
        and cost_stable
        and start_stable
        and event_hit_rate > 0.5
    )

    recommendation = {
        "formal_acceptance": formal_acceptance,
        "best_structural_candidate": "full_switch_with_no_trade_band",
        "rule": {
            "eligibility": "替代ETF必须通过现行完整核心买入条件",
            "no_trade_band": "当前持仓仍在核心前5时不换",
            "dominance": "替代ETF动量分至少领先10个百分点",
            "persistence": "连续2个收盘确认",
            "minimum_hold": "当前仓至少持有5个交易日",
            "transition": "确认后下一开盘一次性换仓；不增加双袖或分步状态",
        },
        "reason": (
            "缓冲带符合交易成本下的无交易区原则；本项目里分步方案未改善最大回撤，"
            "反而增加交易腿，因此按简单优先淘汰。历史事件数量仍不足，只能作为影子候选。"
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    blend_frame.to_csv(OUTPUT / "allocation_comparison.csv", index=False, encoding="utf-8-sig")
    cost_frame.to_csv(OUTPUT / "cost_stress.csv", index=False, encoding="utf-8-sig")
    start_frame.to_csv(OUTPUT / "start_date_stability.csv", index=False, encoding="utf-8-sig")
    leave_frame.to_csv(OUTPUT / "leave_one_event_out.csv", index=False, encoding="utf-8-sig")
    small_frame.to_csv(OUTPUT / "small_account.csv", index=False, encoding="utf-8-sig")
    candidate_events.to_csv(OUTPUT / "candidate_events.csv", index=False, encoding="utf-8-sig")
    technical_events.to_csv(
        OUTPUT / "technical_proxy_events.csv", index=False, encoding="utf-8-sig"
    )
    one_day_events.to_csv(OUTPUT / "one_day_opportunities.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({key: result.equity for key, result in results.items()}).to_csv(
        OUTPUT / "equity.csv", encoding="utf-8-sig"
    )

    payload = {
        "status": "research_only",
        "generated_through": end,
        "formal_strategy_changed": False,
        "research_principles": [
            "候选必须通过原策略完整买入门槛",
            "使用持仓缓冲带避免每次榜首变化都交易",
            "用分步调整降低一次判断错误的损失",
            "最高历史收益不作为选择标准",
        ],
        "recommendation": recommendation,
        "evidence": {
            "full_switch": full.to_dict(),
            "sleeve50": half.to_dict(),
            "staged50": staged.to_dict(),
            "candidate_event_count": len(candidate_events),
            "candidate_event_years": int(
                pd.to_datetime(candidate_events["signal_date"]).dt.year.nunique()
            ),
            "one_day_completed_event_count": int(len(completed_20d)),
            "one_day_20d_hit_rate": event_hit_rate,
            "one_day_median_20d_edge": event_median_edge,
            "leave_one_event_out_all_positive": leave_positive,
            "cost_stable": cost_stable,
            "start_date_stable": start_stable,
            "enough_events": enough_events,
            "technical_proxy_event_count": int(len(technical_events)),
            "technical_proxy_20d_hit_rate": technical_hit_rate,
            "technical_proxy_median_20d_edge": technical_median_edge,
        },
        "limitations": [
            "两日候选全历史仍只有两次事件，无法统计确认其胜率。",
            "一次2024年金融科技事件解释了大部分回测增益。",
            "50%分步方案改变单仓约束，正式化前必须由用户明确批准。",
            "所有历史均已被研究，真正证据只能来自冻结后的前瞻影子。",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    blend_index = blend_frame.set_index("variant")
    lines = [
        "# 机会成本换仓稳健性研究",
        "",
        "## 最终结论",
        "",
        "**没有足够证据把任何换仓规则直接加入正式策略。**",
        "",
        "当前最合理的结构候选不是看见榜首就追入，而是简单的“无交易缓冲带”：旧仓掉出前5、完整合格核心候选领先10个百分点、连续2日、旧仓至少持有5日后，下一开盘一次性换仓。分步和双袖都不进入候选。",
        "",
        f"- 100%换仓累计收益{full['total_return']:.2%}，相对基线{full['delta_total_return']:+.2%}；首日50%过渡累计{staged['total_return']:.2%}。两者最大回撤相同，但分步多3个交易腿，因此分步没有净结构优势。",
        f"- 一日版本共有{len(completed_20d)}个完成20日观察的事件，强者胜率{event_hit_rate:.1%}、相对收益中位数{event_median_edge:+.2%}，并没有稳定超过50%。",
        f"- 两日版本仍只有{len(candidate_events)}次事件，分布在{pd.to_datetime(candidate_events['signal_date']).dt.year.nunique()}个年份；不能把两次成功当成普遍规律。",
        f"- 为避免AI历史覆盖不足造成假稀疏，另用统一价格常规门槛作结构代理，共得到{len(technical_events)}次完成事件；20日胜率{technical_hit_rate:.1%}、中位相对收益{technical_median_edge:+.2%}。均值被金融科技单笔拉高，仍没有普遍优势。",
        f"- 去掉任一历史换仓后其余路径仍优于基线：{'是' if leave_positive else '否'}；成本压力稳定：{'是' if cost_stable else '否'}；不同起点稳定：{'是' if start_stable else '否'}。这些只能说明结构不脆弱，不能弥补事件数量不足。",
        "",
        "## 仓位方式",
        "",
        "| 方式 | 累计收益 | 年化 | 最大回撤 | 夏普 | 交易腿增量 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in blend_weights:
        row = blend_index.loc[key]
        lines.append(
            f"| {labels[key]} | {row['total_return']:.2%} | {row['cagr']:.2%} | "
            f"{row['max_drawdown']:.2%} | {row['sharpe']:.2f} | "
            f"{int(row['delta_trade_count']):+d} |"
        )
    lines.extend(
        [
            "",
            "## 决策",
            "",
            "- 正式策略、账户与订单不变。",
            "- 研究影子只记录是否触发，不接入每日买卖入口。",
            "- 分步换仓和双袖方案因没有改善回撤且增加交易，按奥卡姆剃刀淘汰。",
            "- 每月复盘新增事件；达到至少8次、跨3个年份且去单笔后仍稳定，再讨论正式化。",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
