from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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
OUTPUT = ROOT / "results" / "research" / "opportunity_cost_switch"


@dataclass(frozen=True)
class SwitchRule:
    label: str
    score_margin: float | None
    current_rank_above: int | None = None
    confirmation_days: int = 1
    minimum_hold_days: int = 0
    roc20_margin: float | None = None


RULES = {
    "baseline": SwitchRule("现行退出后再换仓", None),
    "rank_2d": SwitchRule("持仓掉出前5且强者持续2日", 0.0, 5, 2, 5),
    "gap5_2d": SwitchRule("动量分领先5个百分点且持续2日", 0.05, None, 2, 5),
    "combined5_1d": SwitchRule("前5外＋领先5个百分点（1日）", 0.05, 5, 1, 5),
    "combined5_2d": SwitchRule("前5外＋领先5个百分点（2日）", 0.05, 5, 2, 5),
    "combined5_3d": SwitchRule("前5外＋领先5个百分点（3日）", 0.05, 5, 3, 5),
    "combined8_2d": SwitchRule("前5外＋领先8个百分点（2日）", 0.08, 5, 2, 5),
    "combined10_2d": SwitchRule("前5外＋领先10个百分点（2日）", 0.10, 5, 2, 5),
    "combined5_roc3_2d": SwitchRule(
        "前5外＋分数领先5点＋ROC20领先3点（2日）", 0.05, 5, 2, 5, 0.03
    ),
}


PERIODS = {
    "2018_2020": ("2018-07-02", "2020-12-31"),
    "2021_2022": ("2021-01-01", "2022-12-31"),
    "2023_2024": ("2023-01-01", "2024-12-31"),
    "2025_2026": ("2025-01-01", "2026-07-29"),
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def premium_sensitive(symbols: list[str]) -> list[str]:
    return [
        symbol
        for symbol in symbols
        if symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"
    ]


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_state_machine(
    symbols: list[str],
    features,
    eligibility: pd.DataFrame,
    decision: dict,
    reentry_cooldown_days: int,
    rule: SwitchRule,
    blocked_switch_dates: set[pd.Timestamp] | None = None,
    blocked_switch_pairs: set[tuple[str, str]] | None = None,
    switch_entry_gate: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the formal one-position state machine, then optionally pre-empt.

    A switch candidate must already pass the complete formal entry gate. Core
    holdings can only be displaced by another core ETF; satellites retain the
    formal rule that any qualified core candidate has priority.
    """

    calendar = eligibility.index
    entry = (
        eligibility
        & decision["entry_gate"]
        & features.raw_score.notna()
    ).fillna(False)
    switch_entry = (
        eligibility
        & (decision["entry_gate"] if switch_entry_gate is None else switch_entry_gate)
        & features.raw_score.notna()
    ).fillna(False)
    entry_score = decision["entry_score"]
    decision_rank = decision["entry_rank"]
    soft_exit = decision["soft_exit_confirmation"]
    dual_rank_decline = decision["dual_rank_decline"]
    core = set(decision["core_symbols"])
    challenger = set(decision["challenger_symbols"])

    weights = pd.DataFrame(0.0, index=calendar, columns=symbols)
    last_exit = {symbol: -10_000_000 for symbol in symbols}
    selected: str | None = None
    entered_at = -10_000_000
    streak_symbol: str | None = None
    streak_count = 0
    events: list[dict] = []
    blocked_switch_dates = blocked_switch_dates or set()
    blocked_switch_pairs = blocked_switch_pairs or set()

    for location, date in enumerate(calendar):
        exited_today: set[str] = set()
        forced_entry_candidate: str | None = None
        core_available = [
            symbol
            for symbol in decision["core_symbols"]
            if bool(entry.loc[date, symbol])
            and location - last_exit[symbol] > reentry_cooldown_days
        ]

        regular_reason: str | None = None
        if selected is not None:
            if selected in challenger and core_available:
                regular_reason = "核心候选恢复"
            elif not bool(features.above_ma.loc[date, selected]):
                regular_reason = "跌破MA120"
            elif (
                float(features.roc_short.loc[date, selected]) < 0.0
                and bool(soft_exit.loc[date, selected])
            ):
                regular_reason = "ROC20转负"
            elif (
                bool(dual_rank_decline.loc[date, selected])
                and bool(soft_exit.loc[date, selected])
            ):
                regular_reason = "5日与20日排名同时下滑"

        if regular_reason is not None and selected is not None:
            exited_today.add(selected)
            last_exit[selected] = location
            selected = None
            streak_symbol = None
            streak_count = 0

        if selected is not None and rule.score_margin is not None:
            candidates = [
                symbol
                for symbol in symbols
                if symbol != selected
                and bool(switch_entry.loc[date, symbol])
                and symbol not in exited_today
                and location - last_exit[symbol] > reentry_cooldown_days
            ]
            if selected in core:
                candidates = [symbol for symbol in candidates if symbol in core]
            elif core_available:
                candidates = [symbol for symbol in candidates if symbol in core]
            candidate = (
                str(entry_score.loc[date, candidates].sort_values(ascending=False).index[0])
                if candidates
                else None
            )
            qualifies = candidate is not None
            score_gap = np.nan
            roc_gap = np.nan
            if candidate is not None:
                score_gap = float(
                    features.raw_score.loc[date, candidate]
                    - features.raw_score.loc[date, selected]
                )
                roc_gap = float(
                    features.roc_short.loc[date, candidate]
                    - features.roc_short.loc[date, selected]
                )
                qualifies &= score_gap >= rule.score_margin
                if rule.current_rank_above is not None:
                    qualifies &= bool(
                        decision_rank.loc[date, selected] > rule.current_rank_above
                    )
                if rule.roc20_margin is not None:
                    qualifies &= roc_gap >= rule.roc20_margin
                qualifies &= location - entered_at >= rule.minimum_hold_days

            if qualifies and candidate is not None:
                if streak_symbol == candidate:
                    streak_count += 1
                else:
                    streak_symbol = candidate
                    streak_count = 1
            else:
                streak_symbol = None
                streak_count = 0

            if (
                candidate is not None
                and qualifies
                and streak_count >= rule.confirmation_days
                and date not in blocked_switch_dates
                and (selected, candidate) not in blocked_switch_pairs
            ):
                old = selected
                exited_today.add(old)
                last_exit[old] = location
                selected = None
                forced_entry_candidate = candidate
                events.append(
                    {
                        "signal_date": date,
                        "from_symbol": old,
                        "to_symbol": candidate,
                        "from_rank": float(decision_rank.loc[date, old]),
                        "to_rank": float(decision_rank.loc[date, candidate]),
                        "score_gap": score_gap,
                        "roc20_gap": roc_gap,
                        "confirmation_days": rule.confirmation_days,
                    }
                )
                streak_symbol = None
                streak_count = 0

        if selected is None:
            if forced_entry_candidate is not None:
                selected = forced_entry_candidate
                entered_at = location
            candidates = [
                symbol
                for symbol in symbols
                if bool(entry.loc[date, symbol])
                and symbol not in exited_today
                and location - last_exit[symbol] > reentry_cooldown_days
            ]
            core_candidates = [symbol for symbol in candidates if symbol in core]
            if core_candidates:
                candidates = core_candidates
            if selected is None and candidates:
                selected = str(
                    entry_score.loc[date, candidates]
                    .sort_values(ascending=False)
                    .index[0]
                )
                entered_at = location

        if selected is not None:
            weights.loc[date, selected] = 1.0

    return weights, pd.DataFrame(events)


def forward_event_returns(
    events: pd.DataFrame, open_prices: pd.DataFrame
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    calendar = open_prices.index
    output = events.copy()
    for horizon in (5, 10, 20):
        advantages = []
        for row in events.itertuples(index=False):
            signal_loc = int(calendar.get_loc(pd.Timestamp(row.signal_date)))
            entry_loc = signal_loc + 1
            end_loc = min(entry_loc + horizon, len(calendar) - 1)
            if entry_loc >= len(calendar) or end_loc <= entry_loc:
                advantages.append(np.nan)
                continue
            old_start = float(open_prices.iloc[entry_loc][row.from_symbol])
            new_start = float(open_prices.iloc[entry_loc][row.to_symbol])
            old_end = float(open_prices.iloc[end_loc][row.from_symbol])
            new_end = float(open_prices.iloc[end_loc][row.to_symbol])
            advantages.append((new_end / new_start) - (old_end / old_start))
        output[f"new_minus_old_{horizon}d"] = advantages
    return output


def actual_live_switch_snapshot(
    account: dict,
    features,
    eligibility: pd.DataFrame,
    decision: dict,
    rule: SwitchRule,
    end: str,
) -> dict:
    """Evaluate the research trigger against the confirmed live holding only."""

    positions = account.get("positions", [])
    if len(positions) != 1:
        return {"applicable": False, "reason": "实盘不是单一持仓"}
    holding = str(positions[0]["symbol"])
    start = pd.Timestamp(account["performance"]["strategy_start_date"])
    finish = pd.Timestamp(end)
    calendar = eligibility.index
    start_loc = int(calendar.get_indexer([start], method="bfill")[0])
    finish_loc = int(calendar.get_loc(finish))
    core = set(decision["core_symbols"])
    candidate_pool = decision["core_symbols"] if holding in core else list(eligibility.columns)
    entry = (
        eligibility
        & decision["entry_gate"]
        & features.raw_score.notna()
    ).fillna(False)
    streak_symbol = None
    streak_count = 0
    qualifying_dates = []
    latest = None

    for location in range(start_loc, finish_loc + 1):
        date = calendar[location]
        candidates = [
            symbol
            for symbol in candidate_pool
            if symbol != holding and bool(entry.loc[date, symbol])
        ]
        candidate = (
            str(
                decision["entry_score"]
                .loc[date, candidates]
                .sort_values(ascending=False)
                .index[0]
            )
            if candidates
            else None
        )
        qualifies = candidate is not None
        score_gap = np.nan
        roc_gap = np.nan
        if candidate is not None:
            score_gap = float(
                features.raw_score.loc[date, candidate]
                - features.raw_score.loc[date, holding]
            )
            roc_gap = float(
                features.roc_short.loc[date, candidate]
                - features.roc_short.loc[date, holding]
            )
            qualifies &= score_gap >= float(rule.score_margin or 0.0)
            if rule.current_rank_above is not None:
                qualifies &= bool(
                    decision["entry_rank"].loc[date, holding]
                    > rule.current_rank_above
                )
            if rule.roc20_margin is not None:
                qualifies &= roc_gap >= rule.roc20_margin
            qualifies &= location - start_loc >= rule.minimum_hold_days
        if qualifies and candidate is not None:
            if streak_symbol == candidate:
                streak_count += 1
            else:
                streak_symbol = candidate
                streak_count = 1
            qualifying_dates.append(str(date.date()))
        else:
            streak_symbol = None
            streak_count = 0
        latest = {
            "date": str(date.date()),
            "candidate": candidate,
            "holding_rank": clean(decision["entry_rank"].loc[date, holding]),
            "candidate_rank": (
                clean(decision["entry_rank"].loc[date, candidate])
                if candidate is not None
                else None
            ),
            "score_gap": clean(score_gap),
            "roc20_gap": clean(roc_gap),
            "qualifies_today": bool(qualifies),
        }
    return {
        "applicable": True,
        "holding": holding,
        "holding_since": str(start.date()),
        **(latest or {}),
        "same_candidate_streak": streak_count,
        "required_streak": rule.confirmation_days,
        "qualifying_dates": qualifying_dates,
        "would_switch_next_open_if_rule_were_formal": bool(
            streak_symbol is not None and streak_count >= rule.confirmation_days
        ),
        "formal_order_unchanged": True,
    }


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    config = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    categories = {
        symbol_key(item): item["category"] for item in market["universe"]
    }
    sentiment, available = load_sentiment_matrices(
        FEATURES, panel["close"].index, symbols
    )
    official, features, eligibility, _, _, decision = build_ye_signals(
        panel, symbols, categories, config, sentiment, available
    )
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    initial_capital = float(market["project"]["initial_capital"])
    cash = config["cash_management"]
    cash_management = {
        "annual_rate": float(cash["historical_backtest_annual_rate"]),
        "fee_rate": float(cash["fee_rate"]),
        "minimum_order": float(cash["minimum_order"]),
        "order_lot": float(cash["order_lot"]),
    }
    project = execution_project(
        market,
        premium_sensitive(symbols),
        eligibility.shift(1, fill_value=False).astype(bool),
    )

    rows = []
    period_rows = []
    equities = {}
    all_events = []
    generated_weights = {}
    for key, rule in RULES.items():
        weights, events = build_state_machine(
            symbols,
            features,
            eligibility,
            decision,
            int(config["enhanced_selection"]["reentry_cooldown_days"]),
            rule,
        )
        generated_weights[key] = weights
        if key == "baseline":
            mismatch = (weights - official.weights).abs().max().max()
            if float(mismatch) > 1e-12:
                mismatch_dates = weights.index[
                    (weights - official.weights).abs().max(axis=1).gt(1e-12)
                ]
                raise RuntimeError(
                    f"baseline state machine mismatch on {len(mismatch_dates)} dates; "
                    f"first={mismatch_dates[0]}"
                )
        result = run_backtest(
            rule.label,
            panel,
            weights,
            start,
            end,
            project,
            cash_management=cash_management,
        )
        equities[key] = result.equity
        rows.append(
            {
                "variant": key,
                "label": rule.label,
                **result.metrics,
                "preemptive_switch_count": int(len(events)),
            }
        )
        for period, (period_start, period_end) in PERIODS.items():
            period_rows.append(
                {
                    "variant": key,
                    "period": period,
                    **period_metrics(
                        result.equity,
                        period_start,
                        period_end,
                        initial_capital,
                    ),
                }
            )
        if not events.empty:
            events = forward_event_returns(events, panel["open"][symbols])
            events.insert(0, "variant", key)
            all_events.append(events)

    metrics = pd.DataFrame(rows)
    baseline = metrics.set_index("variant").loc["baseline"]
    for field in (
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "trade_count",
        "total_fees",
        "slippage_cost_estimate",
    ):
        metrics[f"delta_{field}"] = metrics[field] - float(baseline[field])
    period_frame = pd.DataFrame(period_rows)
    period_baseline = (
        period_frame[period_frame["variant"] == "baseline"]
        .set_index("period")["total_return"]
    )
    period_frame["delta_total_return"] = period_frame.apply(
        lambda row: row["total_return"] - period_baseline.loc[row["period"]], axis=1
    )
    events_frame = (
        pd.concat(all_events, ignore_index=True)
        if all_events
        else pd.DataFrame()
    )
    equity_frame = pd.DataFrame(equities)
    equity_frame.index.name = "date"

    neighbor_keys = [
        "combined5_1d",
        "combined5_2d",
        "combined5_3d",
        "combined8_2d",
        "combined10_2d",
        "combined5_roc3_2d",
    ]
    neighbor = metrics.set_index("variant").loc[neighbor_keys]
    stable_better_count = int((neighbor["delta_total_return"] > 0).sum())
    drawdown_not_worse_count = int(
        (neighbor["delta_max_drawdown"] >= -0.02).sum()
    )
    candidate = metrics.set_index("variant").loc["combined5_2d"]
    candidate_periods = period_frame[
        period_frame["variant"] == "combined5_2d"
    ]
    materially_worse_periods = int(
        (candidate_periods["delta_total_return"] < -0.05).sum()
    )
    candidate_events = events_frame[
        events_frame["variant"] == "combined5_2d"
    ].copy()
    candidate_event_years = int(
        pd.to_datetime(candidate_events["signal_date"]).dt.year.nunique()
    )
    positive_edges = candidate_events["new_minus_old_20d"].dropna().clip(lower=0)
    largest_event_edge_share = (
        float(positive_edges.max() / positive_edges.sum())
        if len(positive_edges) and positive_edges.sum() > 0
        else np.nan
    )
    enough_independent_events = bool(
        len(candidate_events) >= 8
        and candidate_event_years >= 3
        and np.isfinite(largest_event_edge_share)
        and largest_event_edge_share <= 0.60
    )
    accepted = bool(
        candidate["delta_total_return"] > 0
        and stable_better_count >= 4
        and drawdown_not_worse_count >= 5
        and materially_worse_periods == 0
        and candidate["preemptive_switch_count"] <= 20
        and enough_independent_events
    )

    today = pd.Timestamp(end)
    today_snapshot = {
        "date": end,
        "formal_holding": str(
            official.weights.loc[today][official.weights.loc[today].gt(0)].index[0]
        )
        if official.weights.loc[today].gt(0).any()
        else None,
        "formal_entry_candidates": [
            str(symbol)
            for symbol in decision["entry_score"].loc[today]
            .where((eligibility & decision["entry_gate"]).loc[today])
            .dropna()
            .sort_values(ascending=False)
            .index
        ],
        "candidate_rule_target": str(
            generated_weights["combined5_2d"].loc[today][
                generated_weights["combined5_2d"].loc[today].gt(0)
            ].index[0]
        )
        if generated_weights["combined5_2d"].loc[today].gt(0).any()
        else None,
        "candidate_switched_today": bool(
            not events_frame.empty
            and (
                (events_frame["variant"] == "combined5_2d")
                & (pd.to_datetime(events_frame["signal_date"]) == today)
            ).any()
        ),
    }
    account = json.loads(
        (ROOT / "results" / "live" / "account_state.json").read_text(
            encoding="utf-8"
        )
    )
    live_snapshot = actual_live_switch_snapshot(
        account,
        features,
        eligibility,
        decision,
        RULES["combined5_2d"],
        end,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    period_frame.to_csv(
        OUTPUT / "period_metrics.csv", index=False, encoding="utf-8-sig"
    )
    events_frame.to_csv(
        OUTPUT / "switch_events.csv", index=False, encoding="utf-8-sig"
    )
    equity_frame.to_csv(OUTPUT / "equity.csv", encoding="utf-8-sig")

    payload = {
        "status": "research_only",
        "generated_through": end,
        "formal_strategy_changed": False,
        "baseline_reproduced_exactly": True,
        "hypothesis": (
            "当前核心持仓仍未触发退出时，若另一个完整合格核心候选持续显著领先，"
            "允许机会成本换仓可能比一律等待旧仓退出更有效。"
        ),
        "pre_registered_candidate": {
            "rule": "combined5_2d",
            "definition": RULES["combined5_2d"].__dict__,
            "acceptance": {
                "candidate_total_return_above_baseline": True,
                "at_least_four_of_six_neighbors_above_baseline": True,
                "at_least_five_of_six_neighbors_drawdown_not_worse_by_over_2pp": True,
                "no_period_underperformance_worse_than_5pp": True,
                "preemptive_switches_no_more_than_20": True,
                "at_least_eight_events_across_three_years": True,
                "largest_positive_event_edge_no_more_than_60pct": True,
            },
            "accepted": accepted,
        },
        "robustness": {
            "neighbors_above_baseline": stable_better_count,
            "neighbors_drawdown_not_worse_by_over_2pp": drawdown_not_worse_count,
            "candidate_materially_worse_periods": materially_worse_periods,
            "candidate_event_count": int(len(candidate_events)),
            "candidate_event_years": candidate_event_years,
            "largest_positive_20d_event_edge_share": clean(
                largest_event_edge_share
            ),
            "enough_independent_events": enough_independent_events,
        },
        "historical_path_today": today_snapshot,
        "confirmed_live_holding_today": live_snapshot,
        "metrics": clean(metrics.to_dict(orient="records")),
        "period_metrics": clean(period_frame.to_dict(orient="records")),
        "limitations": [
            "截至2026-07-29的历史已反复研究，结果不是样本外证据。",
            "阈值只用于检验结构稳定性，不能按历史最高收益挑选正式参数。",
            "若候选通过，也应先进入前瞻影子而非直接改变实盘订单。",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_index = metrics.set_index("variant")
    lines = [
        "# 机会成本换仓研究",
        "",
        "## 结论",
        "",
        (
            "预先指定的候选规则为：持仓已掉出核心前5、另一个完整合格核心候选的"
            "动量分领先至少5个百分点、连续2个收盘确认、且原仓至少持有5个交易日。"
        ),
        "",
        f"- 验收结果：**{'通过研究门槛' if accepted else '未通过研究门槛'}**；正式策略未修改。",
        f"- 当前规则累计收益：{baseline['total_return']:.2%}；候选：{candidate['total_return']:.2%}，差异{candidate['delta_total_return']:+.2%}。",
        f"- 当前规则最大回撤：{baseline['max_drawdown']:.2%}；候选：{candidate['max_drawdown']:.2%}。",
        f"- 候选新增主动换仓：{int(candidate['preemptive_switch_count'])}次；交易腿增加{int(candidate['delta_trade_count'])}笔。",
        f"- 六个邻近版本中，{stable_better_count}/6累计收益高于基线，{drawdown_not_worse_count}/6的最大回撤没有恶化超过2个百分点。",
        f"- 但候选全历史只有{len(candidate_events)}次主动换仓，且只发生在{candidate_event_years}个年份；最大一笔占20日正向相对优势的{largest_event_edge_share:.1%}，独立证据不足。",
        f"- 确认实盘持仓口径下，医药相对银行已连续{live_snapshot.get('same_candidate_streak', 0)}日满足该研究触发；若规则早已冻结，下一开盘会换仓，但本次研究不能追溯修改正式订单。",
        "",
        "## 全区间比较",
        "",
        "| 版本 | 累计收益 | 年化 | 最大回撤 | 夏普 | 主动换仓 | 交易腿 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, rule in RULES.items():
        row = metric_index.loc[key]
        lines.append(
            f"| {rule.label} | {row['total_return']:.2%} | {row['cagr']:.2%} | "
            f"{row['max_drawdown']:.2%} | {row['sharpe']:.2f} | "
            f"{int(row['preemptive_switch_count'])} | {int(row['trade_count'])} |"
        )
    lines.extend(
        [
            "",
            "## 分期比较（相对现行规则的累计收益差）",
            "",
            "| 版本 | 2018—20 | 2021—22 | 2023—24 | 2025—26 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in RULES:
        subset = period_frame[period_frame["variant"] == key].set_index("period")
        lines.append(
            f"| {RULES[key].label} | "
            + " | ".join(
                f"{subset.loc[period, 'delta_total_return']:+.2%}" for period in PERIODS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 新标的必须先通过原策略完整买入条件；资讯或技术门槛不降低。",
            "- 核心持仓不能被卫星抢走；卫星仍只在核心空档参与。",
            "- 该研究只检验是否值得建立前瞻影子，不允许直接覆盖现有实盘计划。",
            "- 表面增益主要来自2024-09的一次金融科技行情，必须用未来独立事件验证。",
            "- 当前历史不是样本外，不能按表中最高收益版本倒推正式阈值。",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
