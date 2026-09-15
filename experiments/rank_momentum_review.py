"""One isolated follow-up: protect only improving own momentum, not top-five rank."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments import rank_exit_review as ref

ROOT = ref.ROOT
OUT = ROOT / "results/research/rank_momentum_review_20260910"
PROTOCOL = ROOT / "docs/2026-09-10_排名与自身动量分离研究计划.md"


def own_momentum_protection(close: pd.DataFrame, rules: dict) -> pd.DataFrame:
    score = (close.pct_change(rules["roc_short_days"], fill_method=None) * rules["roc_short_weight"]
             + close.pct_change(rules["roc_medium_days"], fill_method=None) * rules["roc_medium_weight"])
    finite = pd.DataFrame(np.isfinite(score), index=score.index, columns=score.columns)
    result = finite.copy()
    for window in (rules["rank_change_short_days"], rules["rank_change_long_days"]):
        result &= finite.shift(window, fill_value=False) & score.ge(score.shift(window))
    return result


def episode_intervals(left: pd.DataFrame, right: pd.DataFrame) -> list[dict]:
    """Signal-target divergence episodes, not independent completed trades."""
    different = left.gt(0).ne(right.gt(0)).any(axis=1)
    dates = left.index
    result, start = [], None
    for i, value in enumerate(different):
        if value and start is None:
            start = i
        if not value and start is not None:
            if i + 1 < len(dates):
                result.append({"signal_start": dates[start], "execute_start": dates[start + 1],
                               "execute_end": dates[i + 1], "completed": True})
            else:
                result.append({"signal_start": dates[start], "execute_start": dates[min(start + 1, i)],
                               "execute_end": dates[i], "completed": False})
            start = None
    if start is not None and start + 1 < len(dates):
        result.append({"signal_start": dates[start], "execute_start": dates[start + 1],
                       "execute_end": dates[-1], "completed": False})
    return result


def old_exit_comparison() -> pd.DataFrame:
    old = pd.read_csv(ROOT / "results/research/rank_exit_review_20260910/trades.csv")
    old = old.loc[old.track.eq("mixed")]
    joined = old.loc[~old.candidate].merge(old.loc[old.candidate], on=["symbol", "entry_date"], suffixes=("_base", "_top5"))
    joined = joined.loc[joined.exit_date_base.ne(joined.exit_date_top5)].copy()
    joined["delta_pp"] = (joined.net_return_top5 - joined.net_return_base) * 100
    return joined[["symbol", "entry_date", "exit_date_base", "exit_date_top5", "net_return_base", "net_return_top5", "delta_pp"]].sort_values("delta_pp")


def run():
    market = yaml.safe_load((ROOT / "config/market.yaml").read_text())
    config = yaml.safe_load((ROOT / "config/ye_strategy.yaml").read_text())
    end = str(market["project"]["data_end"])
    if end != "2026-09-10":
        raise ValueError("本研究固定截止2026-09-10，不得自动延长区间")
    guard_paths = [ROOT / "config/ye_strategy.yaml", ROOT / "config/market.yaml",
                   ROOT / "results/live/account_state.json", ROOT / f"results/live/{end}_order_plan.json",
                   ROOT / "results/ye_strategy/signal_weights.csv", ROOT / "results/ye_strategy/summary.json",
                   ROOT / "outputs/ETF轮动策略_今日日报.html"]
    guard = {str(p.relative_to(ROOT)): ref.digest(p) for p in guard_paths}
    panel = ref.load_panel(market, ROOT / "market_data/prices")
    symbols = ref.universe_keys(market)
    categories = {ref.symbol_key(x): x["category"] for x in market["universe"]}
    calendar = panel["close"].index
    features = ROOT / "market_data/sentiment/features/symbol_daily.csv"
    tracks = {"mixed": ref.load_sentiment_matrices(features, calendar, symbols),
              "uniform_keyword": ref.sentiment_matrices_from_frame(ref.build(use_ai_reviews=False), calendar, symbols)}
    protection = own_momentum_protection(panel["close"][symbols], config["rules"])
    cash = config["cash_management"]
    cash_model = {"annual_rate": cash["historical_backtest_annual_rate"], "fee_rate": cash["fee_rate"],
                  "minimum_order": cash["minimum_order"], "order_lot": cash["order_lot"]}
    premium = [s for s in symbols if s.startswith("513") or s == "159941.SZ"]
    rows, phases, episodes, event_rows, curves = [], [], [], [], {}
    for track, (sentiment, available) in tracks.items():
        original, _, eligibility, _, _, context = ref.build_ye_signals(panel, symbols, categories, config, sentiment, available)
        changed_context = {**context, "dual_rank_decline": context["dual_rank_decline"] & ~protection}
        score = panel["close"][symbols].pct_change(20, fill_method=None) + 1.5 * panel["close"][symbols].pct_change(60, fill_method=None)
        full, target = {}, {}
        for start in ref.STARTS:
            for candidate in (False, True):
                weights = ref.signals(panel, symbols, config, changed_context if candidate else context,
                                      eligibility, start, False)
                if start == ref.STARTS[0] and not candidate:
                    assert np.allclose(weights.loc[start:], original.weights.loc[start:])
                scenarios = [(1, 100000), (2, 100000)] + ([(1, 9825)] if start == ref.STARTS[0] else [])
                for multiplier, capital in scenarios:
                    project = ref.execution_project(market, premium, eligibility.shift(1, fill_value=False).astype(bool))
                    project["initial_capital"] = capital
                    for costs in [project, *project["symbol_costs"].values()]:
                        for key in ("commission_rate", "slippage_rate", "minimum_commission"):
                            costs[key] *= multiplier
                    result = ref.run_backtest("own_momentum" if candidate else "baseline", panel, weights,
                                              start, end, project, cash_management=cash_model)
                    removed, winner = ref.remove_best_round(result, capital)
                    rows.append({"track": track, "start": start, "candidate": candidate, "cost_multiplier": multiplier,
                                 "capital": capital, **result.metrics, "without_best_round_return": removed,
                                 "removed_round": winner})
                    if start == ref.STARTS[0] and multiplier == 1 and capital == 100000:
                        full[candidate], target[candidate] = result, weights.loc[start:end]
                        curves[f"{track}_{candidate}"] = result.equity
                        if track == "mixed" and not candidate:
                            expected = json.loads((ROOT / "results/ye_strategy/summary.json").read_text())["metrics"]["total_return"]
                            assert abs(result.metrics["total_return"] - expected) < 1e-9
                        for label, first, last in ref.PERIODS:
                            phases.append({"track": track, "candidate": candidate, "period": label,
                                           **ref.period_metrics(result.equity, first, last, capital)})
                print(f"{track} {start} candidate={candidate} completed", flush=True)
        for episode in episode_intervals(target[False], target[True]):
            returns = [ref.period_metrics(full[c].equity, str(episode["execute_start"].date()),
                        str(episode["execute_end"].date()), 100000)["total_return"] for c in (False, True)]
            episodes.append({"track": track, **episode, "baseline_return": returns[0],
                             "candidate_return": returns[1], "delta": returns[1] - returns[0]})
        previous_held = target[False].shift(1).gt(0)
        exited = previous_held & ~target[False].gt(0)
        for day, symbol in exited.stack().loc[lambda x: x].index:
            if (context["dual_rank_decline"].at[day, symbol]
                    and context["soft_exit_confirmation"].at[day, symbol]):
                event_rows.append({"track": track, "date": day, "symbol": symbol,
                                   "rank": context["entry_rank"].at[day, symbol],
                                   "score_change5": score.diff(5).at[day, symbol],
                                   "score_change20": score.diff(20).at[day, symbol],
                                   "own_momentum_protection": bool(protection.at[day, symbol]),
                                   "soft_exit_allowed": bool(context["soft_exit_confirmation"].at[day, symbol])})
    checks = {}
    for track in tracks:
        a, b = ref.paired(rows, track)
        checks[track + "_return_sharpe"] = b["total_return"] > a["total_return"] and b["sharpe"] > a["sharpe"]
        checks[track + "_drawdown"] = b["max_drawdown"] >= a["max_drawdown"] - .02
        a2, b2 = ref.paired(rows, track, cost=2)
        checks[track + "_double_cost"] = b2["total_return"] > a2["total_return"]
        checks[track + "_starts_4_of_5"] = sum(ref.paired(rows, track, start=s)[1]["total_return"] > ref.paired(rows, track, start=s)[0]["total_return"] for s in ref.STARTS) >= 4
        phase = pd.DataFrame([r for r in phases if r["track"] == track]).pivot(index="period", columns="candidate", values="total_return")
        checks[track + "_periods_3_of_4"] = int((phase[True] > phase[False]).sum()) >= 3
        checks[track + "_without_best_round"] = b["without_best_round_return"] > a["without_best_round_return"]
        small_a, small_b = ref.paired(rows, track, capital=9825)
        checks[track + "_small_account"] = small_b["total_return"] > small_a["total_return"]
        completed = [r for r in episodes if r["track"] == track and r["completed"]]
        checks[track + "_events_8_years_3"] = len(completed) >= 8 and len({r["signal_start"].year for r in completed}) >= 3
        checks[track + "_median_event_positive"] = bool(completed) and float(np.median([r["delta"] for r in completed])) > 0
    assert guard == {str(p.relative_to(ROOT)): ref.digest(p) for p in guard_paths}, "正式产物被研究改动"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in [("metrics", rows), ("periods", phases), ("signal_episodes", episodes), ("rank_exit_diagnosis", event_rows)]:
        pd.DataFrame(data).to_csv(OUT / f"{name}.csv", index=False)
    old_exit_comparison().to_csv(OUT / "previous_top5_matched_exits.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "equity.csv")
    sources = [Path(__file__), Path(ref.__file__), PROTOCOL, features, ROOT / "config/market.yaml",
               ROOT / "config/ye_strategy.yaml", ROOT / "config/sentiment.yaml", ROOT / "scripts/build_sentiment_features.py"]
    sources += sorted((ROOT / "src/etf_rotation").glob("*.py"))
    payload = {"status": "forward_observation_candidate" if all(checks.values()) else "not_promoted",
               "generated_through": end, "scenario_count": len(rows), "checks": checks,
               "full_metrics": [r for r in rows if r["start"] == ref.STARTS[0] and r["capital"] == 100000 and r["cost_multiplier"] == 1],
               "formal_unchanged": guard, "input_sha256": {str(p.relative_to(ROOT)): ref.digest(p) for p in sources},
               "caveat": "受到已观察历史启发的回看诊断，不是独立样本外；分叉区间不是独立成交样本，两轨道共享大部分历史"}
    payload["source_snapshot_sha256"] = {
        str(folder.relative_to(ROOT)): ref.hashlib.sha256("\n".join(f"{p.name}:{ref.digest(p)}" for p in sorted(folder.glob(pattern))).encode()).hexdigest()
        for folder, pattern in [(ROOT / "market_data/prices", "*.csv"), (ROOT / "market_data/sentiment/ths_hot_reason", "*.json"),
                                (ROOT / "market_data/sentiment/ai_review", "*.json")]}
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    lines = ["# 排名下降与自身动量分离研究", "", f"截至{end}；{len(rows)}次固定对照；结论：{payload['status']}。正式策略未改变。",
             "", "只在排名双降、但自身动量分相比5日前和20日前均未下降时，暂缓排名卖出；其他退出照旧。",
             "", "| 口径 | 版本 | 累计收益 | 年化 | 夏普 | 最大回撤 |", "|---|---|---:|---:|---:|---:|"]
    for r in payload["full_metrics"]:
        lines.append(f"| {r['track']} | {'自身动量保护' if r['candidate'] else '正式对照'} | {r['total_return']:.2%} | {r['cagr']:.2%} | {r['sharpe']:.2f} | {r['max_drawdown']:.2%} |")
    lines += ["", "## 预设验收", ""] + [f"- {key}：{'通过' if value else '未通过'}" for key, value in checks.items()]
    lines += ["", "## 触发与适用性", ""]
    for track in tracks:
        done = [r for r in episodes if r["track"] == track and r["completed"]]
        protected = [r for r in event_rows if r["track"] == track and r["own_momentum_protection"]]
        median = float(np.median([r["delta"] for r in done])) if done else float("nan")
        lines.append(f"- {track}：原规则排名退出中{len(protected)}次自身双窗口动量未降；完整策略形成{len(done)}个已完成信号分叉区间，涉及{len({r['signal_start'].year for r in done})}个开始年份，区间收益差中位数{median:+.2%}。")
    lines += ["", "上一轮同入场完成交易的出口差异见previous_top5_matched_exits.csv，匹配样本不能覆盖全部路径差。",
              "signal_episodes按信号目标不同到重新相同划分，并整体后移到次日开盘所在日，比较两条真实回测净值的同日历区间收益；部分成交可能跨过区间边界，不能据此做独立成交因果归因。",
              "候选原规则触发点不等于它自己完整持仓路径的事件数。不同起点、两数据轨道、区间样本均非相互独立。去最大赢家沿用上一轮整个执行期日收益剔除，只作压力，非重新回测反事实。",
              "历史已反复观察，不称样本外。未通过不晋升、不再围绕此候选改5/20窗口；只保留证据，不增加每日步骤。", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"status": payload["status"], "checks": checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
