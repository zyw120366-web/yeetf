"""One preregistered rank-exit comparison; never writes formal/live artifacts."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_panel, universe_keys, symbol_key
from etf_rotation.etfwin import etfwin_signals, EtfwinOpportunitySwitch
from etf_rotation.evaluation import realized_round_trips
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.sentiment import load_sentiment_matrices, sentiment_matrices_from_frame
from etf_rotation.ye import build_ye_signals, _rules
from scripts.build_sentiment_features import build, source_regimes

OUT = ROOT / "results/research/rank_exit_review_20260910"
PROTOCOL = ROOT / "docs/2026-09-10_执行修复与排名退出研究计划.md"
STARTS = ["2018-07-02", "2021-01-04", "2023-01-03", "2024-01-02", "2025-01-02"]
PERIODS = [("2018—20", "2018-07-02", "2020-12-31"),
           ("2021—22", "2021-01-01", "2022-12-31"),
           ("2023—24", "2023-01-01", "2024-12-31"),
           ("2025以后", "2025-01-01", "2026-09-10")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signals(panel, symbols, config, context, eligibility, start, candidate):
    """Reuse frozen entry gates; reset to cash without losing indicator warmup."""
    gate = context["entry_gate"].copy()
    gate.loc[gate.index < pd.Timestamp(start)] = False
    dual = context["dual_rank_decline"].copy()
    if candidate:
        dual &= context["entry_rank"].gt(5)
    values = config["enhanced_selection"]["opportunity_switch"]
    switch = EtfwinOpportunitySwitch(
        held_rank_must_exceed=values["held_rank_must_exceed"],
        minimum_score_advantage=values["minimum_score_advantage"],
        confirmation_days=values["confirmation_days"], minimum_hold_days=values["minimum_hold_days"]
    ) if values["enabled"] else None
    raw = (panel["close"][symbols].pct_change(20, fill_method=None)
           + 1.5 * panel["close"][symbols].pct_change(60, fill_method=None))
    bundle, _ = etfwin_signals(
        panel["close"][symbols], symbols, _rules(config["rules"]),
        entry_eligibility=eligibility, entry_gate=gate,
        entry_ranking_score_override=context["entry_score"],
        soft_exit_confirmation=context["soft_exit_confirmation"],
        dual_rank_decline_override=dual,
        reentry_cooldown_days=config["enhanced_selection"]["reentry_cooldown_days"],
        priority_symbols=context["core_symbols"], preempt_for_priority_entry=True,
        opportunity_switch=switch, opportunity_switch_rank=context["entry_rank"],
        opportunity_switch_score=raw, opportunity_switch_symbols=context["core_symbols"],
    )
    return bundle.weights


def remove_best_round(result, capital):
    trips = realized_round_trips(result.trades, result.equity.index)
    returns = result.equity.pct_change()
    returns.iloc[0] = result.equity.iloc[0] / capital - 1
    if trips.empty:
        return float((1 + returns).prod() - 1), None
    best = trips.sort_values("net_pnl", ascending=False).iloc[0]
    # This intentionally removes entire execution-day returns, including any
    # other position on transition days: stress attribution, NOT a new strategy.
    returns.loc[best["entry_date"]:best["exit_completed_date"]] = 0
    return float((1 + returns).prod() - 1), {
        "symbol": best["symbol"], "entry": str(best["entry_date"].date()),
        "exit": str(best["exit_completed_date"].date()), "net_pnl": float(best["net_pnl"]),
    }


def paired(rows, track, start=STARTS[0], cost=1, capital=100000):
    subset = [r for r in rows if r["track"] == track and r["start"] == start
              and r["cost_multiplier"] == cost and r["capital"] == capital]
    return next(r for r in subset if not r["candidate"]), next(r for r in subset if r["candidate"])


def main():
    market = yaml.safe_load((ROOT / "config/market.yaml").read_text())
    config = yaml.safe_load((ROOT / "config/ye_strategy.yaml").read_text())
    end = str(market["project"]["data_end"])
    if end != "2026-09-10":
        raise ValueError("本次预设研究固定截止2026-09-10；不得静默扩展区间")
    guarded = [ROOT / "config/ye_strategy.yaml", ROOT / "results/live/account_state.json",
               ROOT / f"results/live/{end}_order_plan.json", ROOT / "results/ye_strategy/signal_weights.csv"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in guarded}
    panel = load_panel(market, ROOT / "market_data/prices")
    symbols = universe_keys(market)
    categories = {symbol_key(x): x["category"] for x in market["universe"]}
    calendar = panel["close"].index
    features = ROOT / "market_data/sentiment/features/symbol_daily.csv"
    tracks = {"mixed": load_sentiment_matrices(features, calendar, symbols),
              "uniform_keyword": sentiment_matrices_from_frame(build(use_ai_reviews=False), calendar, symbols)}
    cash = config["cash_management"]
    cash_model = {"annual_rate": cash["historical_backtest_annual_rate"], "fee_rate": cash["fee_rate"],
                  "minimum_order": cash["minimum_order"], "order_lot": cash["order_lot"]}
    premium = [s for s in symbols if s.startswith("513") or s == "159941.SZ"]
    metrics, periods, trade_rows, differences, events = [], [], [], [], []
    OUT.mkdir(parents=True, exist_ok=True)
    for track, (sentiment, available) in tracks.items():
        original, _, eligibility, _, _, context = build_ye_signals(
            panel, symbols, categories, config, sentiment, available)
        full_results = {}
        for start in STARTS:
            for candidate in (False, True):
                weights = signals(panel, symbols, config, context, eligibility, start, candidate)
                if track == "mixed" and start == STARTS[0] and not candidate:
                    assert np.allclose(weights.loc[start:], original.weights.loc[start:]), "对照信号未复现正式策略"
                scenarios = [(1, 100000), (2, 100000)]
                if start == STARTS[0]:
                    scenarios.append((1, 9825))
                for multiplier, capital in scenarios:
                    project = execution_project(market, premium, eligibility.shift(1, fill_value=False).astype(bool))
                    project["initial_capital"] = capital
                    for costs in [project, *project["symbol_costs"].values()]:
                        for key in ("commission_rate", "slippage_rate", "minimum_commission"):
                            costs[key] *= multiplier
                    result = run_backtest("candidate" if candidate else "baseline", panel, weights,
                                          start, end, project, cash_management=cash_model)
                    removed, winner = remove_best_round(result, capital)
                    row = {"track": track, "start": start, "candidate": candidate,
                           "cost_multiplier": multiplier, "capital": capital, **result.metrics,
                           "without_best_round_return": removed, "removed_round": winner}
                    metrics.append(row)
                    if start == STARTS[0] and multiplier == 1 and capital == 100000:
                        full_results[candidate] = result
                        for label, first, last in PERIODS:
                            periods.append({"track": track, "candidate": candidate, "period": label,
                                            **period_metrics(result.equity, first, last, capital)})
                        for trip in realized_round_trips(result.trades, result.equity.index).to_dict("records"):
                            trade_rows.append({"track": track, "candidate": candidate, **trip})
                        if track == "mixed" and not candidate:
                            formal = json.loads((ROOT / "results/ye_strategy/summary.json").read_text())
                            assert abs(result.metrics["total_return"] - formal["metrics"]["total_return"]) < 1e-9
                print(f"{track} {start} candidate={candidate}: completed", flush=True)
        left, right = full_results[False], full_results[True]
        def held(weights):
            return weights.apply(lambda row: "|".join(sorted(row.index[row > 1e-8])) or "cash", axis=1)
        positions = pd.DataFrame({"baseline": held(left.actual_weights), "candidate": held(right.actual_weights)})
        for day, row in positions[positions["baseline"] != positions["candidate"]].iterrows():
            differences.append({"track": track, "date": day, **row.to_dict()})
        for day in left.actual_weights.index:
            prior = left.actual_weights.index[left.actual_weights.index < day]
            if not len(prior):
                continue
            for symbol in left.actual_weights.columns[left.actual_weights.loc[prior[-1]] > 1e-8]:
                if (context["dual_rank_decline"].at[day, symbol]
                        and context["entry_rank"].at[day, symbol] <= 5
                        and context["soft_exit_confirmation"].at[day, symbol]):
                    events.append({"track": track, "date": day, "symbol": symbol,
                                   "rank": context["entry_rank"].at[day, symbol],
                                   "roc20": panel["close"][symbol].pct_change(20, fill_method=None).at[day],
                                   "candidate_position": positions.at[day, "candidate"]})
    checks = {}
    for track in tracks:
        a, b = paired(metrics, track)
        checks[track + "_return_sharpe"] = b["total_return"] > a["total_return"] and b["sharpe"] > a["sharpe"]
        checks[track + "_drawdown"] = b["max_drawdown"] >= a["max_drawdown"] - .02
        a2, b2 = paired(metrics, track, cost=2)
        checks[track + "_double_cost"] = b2["total_return"] > a2["total_return"]
        starts_won = sum(paired(metrics, track, start=s)[1]["total_return"] >
                         paired(metrics, track, start=s)[0]["total_return"] for s in STARTS)
        checks[track + "_starts_4_of_5"] = starts_won >= 4
        phase = pd.DataFrame([r for r in periods if r["track"] == track]).pivot(index="period", columns="candidate", values="total_return")
        checks[track + "_periods_3_of_4"] = int((phase[True] > phase[False]).sum()) >= 3
        checks[track + "_without_best_round"] = b["without_best_round_return"] > a["without_best_round_return"]
        small_a, small_b = paired(metrics, track, capital=9825)
        checks[track + "_small_account"] = small_b["total_return"] > small_a["total_return"]
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in guarded}, "研究改动了正式结果"
    payload = {"status": "shadow_candidate" if all(checks.values()) else "rejected",
               "generated_through": end, "checks": checks, "metrics": metrics,
               "formal_unchanged": before, "protocol_sha256": digest(PROTOCOL),
               "caveat": "回看诊断，不是样本外；两数据轨道共享大部分历史，不是两份独立证据",
               "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                    [features, Path(__file__), ROOT / "config/market.yaml", ROOT / "config/sentiment.yaml",
                     ROOT / "scripts/build_sentiment_features.py", ROOT / "src/etf_rotation/sentiment.py",
                     ROOT / "src/etf_rotation/sentiment_ai.py", ROOT / "src/etf_rotation/execution.py",
                     ROOT / "src/etf_rotation/evaluation.py", ROOT / "src/etf_rotation/data.py",
                     ROOT / "src/etf_rotation/ye.py", ROOT / "src/etf_rotation/etfwin.py", ROOT / "src/etf_rotation/backtest.py"]}}
    payload["source_snapshot_sha256"] = {
        str(folder.relative_to(ROOT)): hashlib.sha256("\n".join(
            f"{p.name}:{digest(p)}" for p in sorted(folder.glob(pattern))).encode()).hexdigest()
        for folder, pattern in [(ROOT / "market_data/prices", "*.csv"),
                                (ROOT / "market_data/sentiment/ths_hot_reason", "*.json"),
                                (ROOT / "market_data/sentiment/ai_review", "*.json")]
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    for name, rows in [("metrics", metrics), ("periods", periods), ("trades", trade_rows),
                       ("position_differences", differences), ("rank_exit_events", events)]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    source_regimes(calendar[calendar >= pd.Timestamp(STARTS[0])]).to_csv(OUT / "data_regimes.csv", index=False)
    lines = ["# 排名退出单因素研究", "", f"截止{end}；结论：{payload['status']}。正式策略和账户未修改。",
             "", "唯一变化：排名双降退出额外要求当前排名大于5。其余规则固定；没有扫描更多阈值。",
             "", "| 数据轨道 | 版本 | 累计收益 | 年化 | 夏普 | 最大回撤 | 成交笔数 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for track in tracks:
        for row in paired(metrics, track):
            lines.append(f"| {track} | {'前五外才排名退出' if row['candidate'] else '正式对照'} | {row['total_return']:.2%} | {row['cagr']:.2%} | {row['sharpe']:.2f} | {row['max_drawdown']:.2%} | {row['trade_count']:.0f} |")
    lines += ["", "## 预设验收", ""] + [f"- {key}: {'通过' if value else '未通过'}" for key, value in checks.items()]
    lines += ["", "mixed包含价格回退、关键词代理、AI审核；uniform_keyword在2024年后统一使用历史关键词映射，不冒充AI。两轨道不是独立样本。",
              "分期和不同空仓起点、双倍成本、小资金结果见CSV。trades包含两版本完整成交轮次，position_differences逐日定位持仓差异。",
              "去最大赢家是将各自最大盈利完成持仓执行期间的组合日收益剔除，不是重新运行无该交易的策略；过渡日可能同时剔除其他持仓收益，仅作压力。",
              "本研究未通过则关闭，不围绕该结果继续寻找更优数字；即便通过也不直接晋升实盘。", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"status": payload["status"], "checks": checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
