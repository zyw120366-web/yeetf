"""80%主策略资金＋20%扩展策略资金的执行可行性研究。

不改任何动量、AI、退出或ETF池规则，只回答四个实际问题：

1. 两部分资金历史上有多少时间选同一只、不同ETF或现金；
2. 分歧由新增ETF直接造成多少，之后的持仓路径延续造成多少；
3. 1.2万元、100股整手和最低佣金下，两只ETF能否实际买到；
4. 交易次数和成本增加多少。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from universe_architecture_v2_study import (  # noqa: E402
    cash_management,
    clean,
    load_yaml,
    premium_sensitive,
    run_variant,
)
from universe_robustness_grand_study import blend_weights  # noqa: E402
from etf_rotation.backtest import run_backtest  # noqa: E402
from etf_rotation.data import load_panel, symbol_key, universe_keys  # noqa: E402
from etf_rotation.execution import execution_project  # noqa: E402
from etf_rotation.etfwin import etfwin_signals  # noqa: E402
from etf_rotation.sentiment import load_sentiment_matrices  # noqa: E402
from etf_rotation.ye import _rules  # noqa: E402


FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
OUTPUT = ROOT / "results" / "research" / "two_part_allocation_execution"
MAIN_SHARE = 0.80


def selected_symbol(row: pd.Series) -> str:
    held = row[row > 0.0].index.tolist()
    if not held:
        return "CASH"
    if len(held) != 1:
        raise AssertionError(f"expected at most one selection, got {held}")
    return str(held[0])


def allocation_state(main_symbol: str, expanded_symbol: str) -> str:
    if main_symbol == "CASH" and expanded_symbol == "CASH":
        return "both_cash"
    if main_symbol == expanded_symbol:
        return "same_etf"
    if main_symbol == "CASH":
        return "expanded_only"
    if expanded_symbol == "CASH":
        return "main_only"
    return "different_etfs"


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    formal = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    all_symbols = universe_keys(market)
    core_size = int(
        formal["enhanced_selection"]["universe_architecture"]["core_pool_size"]
    )
    core_symbols = all_symbols[:core_size]
    challengers = [
        str(symbol)
        for symbol in formal["enhanced_selection"]["universe_architecture"][
            "challenger_symbols"
        ]
    ]
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment, available = load_sentiment_matrices(
        FEATURES, panel["close"].index, all_symbols
    )

    def build(key: str, symbols: list[str], challengers_for_run: list[str]):
        return run_variant(
            key,
            market,
            formal,
            panel,
            all_symbols,
            core_symbols,
            categories,
            sentiment,
            available,
            symbols=symbols,
            mode="core_anchor_challenger",
            challengers=challengers_for_run,
        )

    main_strategy = build("main_strategy", core_symbols, [])
    expanded_strategy = build("expanded_strategy", all_symbols, challengers)

    # 独立新增ETF信号：使用完整51只计算出的核心锚定排名、技术门、AI门和退出，
    # 但持仓状态机只持有新增ETF，不让一条看不见的核心影子持仓阻塞20%资金。
    challenger_rules = _rules(formal["rules"])
    challenger_bundle, _ = etfwin_signals(
        panel["close"][challengers],
        challengers,
        challenger_rules,
        entry_eligibility=expanded_strategy["eligibility"][challengers],
        entry_gate=expanded_strategy["decision"]["entry_gate"][challengers],
        entry_ranking_score_override=expanded_strategy["decision"]["entry_score"][
            challengers
        ],
        soft_exit_confirmation=expanded_strategy["decision"][
            "soft_exit_confirmation"
        ][challengers],
        dual_rank_decline_override=expanded_strategy["decision"][
            "dual_rank_decline"
        ][challengers],
        reentry_cooldown_days=int(
            formal["enhanced_selection"]["reentry_cooldown_days"]
        ),
    )
    # 透明的无参数收紧：新增ETF必须比主策略当前持仓的选择分更高。
    # 主策略为空仓时，只需通过原有完整门槛；不新增固定阈值。
    main_selected = main_strategy["weights"].idxmax(axis=1)
    main_has_position = main_strategy["weights"].sum(axis=1).gt(0.0)
    main_current_score = pd.Series(np.nan, index=main_selected.index, dtype=float)
    for date in main_selected.index[main_has_position]:
        symbol = str(main_selected.loc[date])
        main_current_score.loc[date] = float(
            expanded_strategy["decision"]["entry_score"].loc[date, symbol]
        )
    dominance_gate = expanded_strategy["decision"]["entry_gate"][challengers].copy()
    for symbol in challengers:
        dominance_gate[symbol] &= (
            ~main_has_position
            | expanded_strategy["decision"]["entry_score"][symbol].gt(
                main_current_score
            )
        )
    dominant_challenger_bundle, _ = etfwin_signals(
        panel["close"][challengers],
        challengers,
        challenger_rules,
        entry_eligibility=expanded_strategy["eligibility"][challengers],
        entry_gate=dominance_gate,
        entry_ranking_score_override=expanded_strategy["decision"]["entry_score"][
            challengers
        ],
        soft_exit_confirmation=expanded_strategy["decision"][
            "soft_exit_confirmation"
        ][challengers],
        dual_rank_decline_override=expanded_strategy["decision"][
            "dual_rank_decline"
        ][challengers],
        reentry_cooldown_days=int(
            formal["enhanced_selection"]["reentry_cooldown_days"]
        ),
    )
    target_weights = blend_weights(
        main_strategy["weights"],
        expanded_strategy["weights"],
        all_symbols,
        MAIN_SHARE,
    )
    # 更简洁的结构候选：扩展资金只在扩展策略当前持有“新增ETF”时启用；
    # 其余日期100%跟随主策略，不让新增ETF退出后的另一条核心持仓路径继续分叉。
    expanded_is_new = expanded_strategy["weights"][challengers].sum(axis=1).gt(0.0)
    new_etf_only_weights = main_strategy["weights"].reindex(
        columns=all_symbols, fill_value=0.0
    ).copy()
    new_etf_only_weights.loc[expanded_is_new] = (
        main_strategy["weights"].reindex(columns=all_symbols, fill_value=0.0).loc[
            expanded_is_new
        ]
        * MAIN_SHARE
        + expanded_strategy["weights"].reindex(columns=all_symbols, fill_value=0.0).loc[
            expanded_is_new
        ]
        * (1.0 - MAIN_SHARE)
    )
    independent_new_active = challenger_bundle.weights.sum(axis=1).gt(0.0)
    independent_new_weights = main_strategy["weights"].reindex(
        columns=all_symbols, fill_value=0.0
    ).copy()
    independent_new_weights.loc[independent_new_active] = (
        main_strategy["weights"].reindex(columns=all_symbols, fill_value=0.0).loc[
            independent_new_active
        ]
        * MAIN_SHARE
        + challenger_bundle.weights.reindex(columns=all_symbols, fill_value=0.0).loc[
            independent_new_active
        ]
        * (1.0 - MAIN_SHARE)
    )
    dominant_new_active = dominant_challenger_bundle.weights.sum(axis=1).gt(0.0)
    dominant_new_weights = main_strategy["weights"].reindex(
        columns=all_symbols, fill_value=0.0
    ).copy()
    dominant_new_weights.loc[dominant_new_active] = (
        main_strategy["weights"].reindex(columns=all_symbols, fill_value=0.0).loc[
            dominant_new_active
        ]
        * MAIN_SHARE
        + dominant_challenger_bundle.weights.reindex(
            columns=all_symbols, fill_value=0.0
        ).loc[dominant_new_active]
        * (1.0 - MAIN_SHARE)
    )
    eligibility = expanded_strategy["eligibility"].reindex(
        columns=all_symbols, fill_value=False
    )
    project = execution_project(
        market,
        premium_sensitive(all_symbols),
        eligibility.shift(1, fill_value=False).astype(bool),
    )
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    standard = run_backtest(
        "two_part_100000",
        panel,
        target_weights,
        start,
        end,
        project,
        cash_management=cash_management(formal),
    )
    new_etf_only_standard = run_backtest(
        "new_etf_only_100000",
        panel,
        new_etf_only_weights,
        start,
        end,
        project,
        cash_management=cash_management(formal),
    )
    independent_new_standard = run_backtest(
        "independent_new_100000",
        panel,
        independent_new_weights,
        start,
        end,
        project,
        cash_management=cash_management(formal),
    )
    dominant_new_standard = run_backtest(
        "dominant_new_100000",
        panel,
        dominant_new_weights,
        start,
        end,
        project,
        cash_management=cash_management(formal),
    )
    small_project = dict(project)
    small_project["initial_capital"] = 12_000.0
    small = run_backtest(
        "two_part_12000",
        panel,
        target_weights,
        start,
        end,
        small_project,
        cash_management=cash_management(formal),
    )
    new_etf_only_small = run_backtest(
        "new_etf_only_12000",
        panel,
        new_etf_only_weights,
        start,
        end,
        small_project,
        cash_management=cash_management(formal),
    )
    independent_new_small = run_backtest(
        "independent_new_12000",
        panel,
        independent_new_weights,
        start,
        end,
        small_project,
        cash_management=cash_management(formal),
    )
    dominant_new_small = run_backtest(
        "dominant_new_12000",
        panel,
        dominant_new_weights,
        start,
        end,
        small_project,
        cash_management=cash_management(formal),
    )
    small_main = run_backtest(
        "main_only_12000",
        panel,
        main_strategy["weights"].reindex(columns=all_symbols, fill_value=0.0),
        start,
        end,
        small_project,
        cash_management=cash_management(formal),
    )

    calendar = target_weights.loc[pd.Timestamp(start) : pd.Timestamp(end)].index
    main_weights = main_strategy["weights"].reindex(
        index=calendar, columns=all_symbols, fill_value=0.0
    )
    expanded_weights = expanded_strategy["weights"].reindex(
        index=calendar, columns=all_symbols, fill_value=0.0
    )
    rows = []
    for date in calendar:
        main_symbol = selected_symbol(main_weights.loc[date])
        expanded_symbol = selected_symbol(expanded_weights.loc[date])
        state = allocation_state(main_symbol, expanded_symbol)
        rows.append(
            {
                "signal_date": date,
                "main_strategy_target": main_symbol,
                "expanded_strategy_target": expanded_symbol,
                "state": state,
                "expanded_target_is_new_etf": expanded_symbol in challengers,
                "combined_target_count": int((target_weights.loc[date] > 0).sum()),
            }
        )
    path = pd.DataFrame(rows).set_index("signal_date")
    calendar_all = panel["close"].index

    # 将连续的分歧日期合并成事件，避免把同一段持仓误解成许多独立发现。
    divergent = ~path["state"].isin(["same_etf", "both_cash"])
    episode_rows = []
    episode_groups = divergent.ne(divergent.shift(fill_value=False)).cumsum()
    for _, episode in path[divergent].groupby(episode_groups[divergent]):
        signal_start = episode.index[0]
        signal_end = episode.index[-1]
        start_location = calendar_all.get_loc(signal_start)
        end_location = calendar_all.get_loc(signal_end)
        if start_location + 1 >= len(calendar_all):
            continue
        execution_start = calendar_all[start_location + 1]
        execution_end = calendar_all[min(end_location + 1, len(calendar_all) - 1)]

        def episode_return(equity: pd.Series) -> float:
            before = equity[equity.index < execution_start]
            window = equity.loc[execution_start:execution_end]
            if before.empty or window.empty:
                return np.nan
            return float(window.iloc[-1] / before.iloc[-1] - 1.0)

        main_return = episode_return(main_strategy["equity"])
        combined_return = episode_return(standard.equity)
        new_symbols = sorted(
            set(episode.loc[episode["expanded_target_is_new_etf"], "expanded_strategy_target"])
        )
        episode_rows.append(
            {
                "signal_start": signal_start,
                "signal_end": signal_end,
                "execution_start": execution_start,
                "execution_end": execution_end,
                "signal_days": int(len(episode)),
                "states": "|".join(sorted(set(episode["state"]))),
                "main_targets": "|".join(sorted(set(episode["main_strategy_target"]))),
                "expanded_targets": "|".join(
                    sorted(set(episode["expanded_strategy_target"]))
                ),
                "new_etfs": "|".join(new_symbols),
                "main_return": main_return,
                "combined_return": combined_return,
                "combined_minus_main_percentage_points": float(
                    (combined_return - main_return) * 100.0
                ),
            }
        )
    episodes = pd.DataFrame(episode_rows)

    # T日信号在T+1开盘执行。只检查目标为两只ETF的日期，判断1.2万元账户
    # 次日收盘时是否实际同时持有两只；收盘权重会包含当日价格漂移。
    feasibility_rows = []
    for signal_date, row in path[path["state"] == "different_etfs"].iterrows():
        location = calendar_all.get_loc(signal_date)
        if location + 1 >= len(calendar_all):
            continue
        execution_date = calendar_all[location + 1]
        if execution_date not in small.actual_weights.index:
            continue
        actual = small.actual_weights.loc[execution_date].fillna(0.0)
        main_symbol = str(row["main_strategy_target"])
        expanded_symbol = str(row["expanded_strategy_target"])
        feasibility_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "main_symbol": main_symbol,
                "expanded_symbol": expanded_symbol,
                "main_actual_weight_close": float(actual.get(main_symbol, 0.0)),
                "expanded_actual_weight_close": float(actual.get(expanded_symbol, 0.0)),
                "both_positions_present": bool(
                    actual.get(main_symbol, 0.0) > 0.0
                    and actual.get(expanded_symbol, 0.0) > 0.0
                ),
            }
        )
    feasibility = pd.DataFrame(feasibility_rows)

    selected_new_feasibility_rows = []
    for signal_date in expanded_is_new[expanded_is_new].index:
        if signal_date not in calendar:
            continue
        location = calendar_all.get_loc(signal_date)
        if location + 1 >= len(calendar_all):
            continue
        execution_date = calendar_all[location + 1]
        if execution_date not in new_etf_only_small.actual_weights.index:
            continue
        actual = new_etf_only_small.actual_weights.loc[execution_date].fillna(0.0)
        new_symbol = selected_symbol(expanded_strategy["weights"].loc[signal_date])
        main_symbol = selected_symbol(main_strategy["weights"].loc[signal_date])
        new_present = bool(actual.get(new_symbol, 0.0) > 0.0)
        main_required = main_symbol != "CASH"
        main_present = bool(actual.get(main_symbol, 0.0) > 0.0) if main_required else True
        selected_new_feasibility_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "main_symbol": main_symbol,
                "new_symbol": new_symbol,
                "main_required": main_required,
                "main_actual_weight_close": float(actual.get(main_symbol, 0.0))
                if main_required
                else 0.0,
                "new_actual_weight_close": float(actual.get(new_symbol, 0.0)),
                "required_positions_present": bool(main_present and new_present),
            }
        )
    selected_new_feasibility = pd.DataFrame(selected_new_feasibility_rows)

    state_counts = path["state"].value_counts().to_dict()
    total_days = len(path)
    direct_new_days = int(path["expanded_target_is_new_etf"].sum())
    different_days = int((path["state"] == "different_etfs").sum())
    different_core_days = int(
        (
            (path["state"] == "different_etfs")
            & ~path["expanded_target_is_new_etf"]
        ).sum()
    )
    target_changes = int(
        target_weights.loc[calendar]
        .fillna(0.0)
        .ne(target_weights.loc[calendar].shift().fillna(0.0))
        .any(axis=1)
        .sum()
    )
    main_target_changes = int(
        main_weights.fillna(0.0)
        .ne(main_weights.shift().fillna(0.0))
        .any(axis=1)
        .sum()
    )
    expanded_target_changes = int(
        expanded_weights.fillna(0.0)
        .ne(expanded_weights.shift().fillna(0.0))
        .any(axis=1)
        .sum()
    )
    challenger_days = {
        symbol: int((path["expanded_strategy_target"] == symbol).sum())
        for symbol in challengers
    }
    challenger_days = {symbol: days for symbol, days in challenger_days.items() if days}

    execution_summary = {
        "status": "research_only_not_formal_strategy",
        "generated_through": end,
        "plain_language_name": "80%主策略资金＋20%扩展策略资金",
        "total_signal_days": total_days,
        "state_days": {str(key): int(value) for key, value in state_counts.items()},
        "state_share": {
            str(key): float(value / total_days) for key, value in state_counts.items()
        },
        "different_etf_days": different_days,
        "all_divergence_days": int(divergent.sum()),
        "direct_new_etf_days": direct_new_days,
        "different_core_etf_path_days": different_core_days,
        "new_etf_selected_days": challenger_days,
        "divergence_episode_count": int(len(episodes)),
        "main_target_change_days": main_target_changes,
        "expanded_target_change_days": expanded_target_changes,
        "combined_target_change_days": target_changes,
        "small_capital_two_position_checks": int(len(feasibility)),
        "small_capital_both_positions_success_days": int(
            feasibility["both_positions_present"].sum()
        ) if not feasibility.empty else 0,
        "small_capital_both_positions_success_rate": float(
            feasibility["both_positions_present"].mean()
        ) if not feasibility.empty else None,
        "small_capital_expanded_weight_close_median": float(
            feasibility["expanded_actual_weight_close"].median()
        ) if not feasibility.empty else None,
        "small_capital_expanded_weight_close_p10": float(
            feasibility["expanded_actual_weight_close"].quantile(0.10)
        ) if not feasibility.empty else None,
        "small_capital_expanded_weight_close_p90": float(
            feasibility["expanded_actual_weight_close"].quantile(0.90)
        ) if not feasibility.empty else None,
        "small_capital_main_only": clean(small_main.metrics),
        "small_capital_two_part": clean(small.metrics),
        "small_capital_new_etf_only": clean(new_etf_only_small.metrics),
        "selected_new_small_capital_checks": int(len(selected_new_feasibility)),
        "selected_new_small_capital_success_days": int(
            selected_new_feasibility["required_positions_present"].sum()
        ) if not selected_new_feasibility.empty else 0,
        "selected_new_small_capital_success_rate": float(
            selected_new_feasibility["required_positions_present"].mean()
        ) if not selected_new_feasibility.empty else None,
        "selected_new_small_capital_weight_median": float(
            selected_new_feasibility["new_actual_weight_close"].median()
        ) if not selected_new_feasibility.empty else None,
        "small_capital_independent_new": clean(independent_new_small.metrics),
        "small_capital_dominant_new": clean(dominant_new_small.metrics),
        "small_capital_extra_trade_count": int(
            small.metrics["trade_count"] - small_main.metrics["trade_count"]
        ),
        "small_capital_extra_fees": float(
            small.metrics["total_fees"] - small_main.metrics["total_fees"]
        ),
        "small_capital_extra_slippage_estimate": float(
            small.metrics["slippage_cost_estimate"]
            - small_main.metrics["slippage_cost_estimate"]
        ),
        "standard_capital_two_part": clean(standard.metrics),
        "standard_capital_new_etf_only": clean(new_etf_only_standard.metrics),
        "standard_capital_independent_new": clean(independent_new_standard.metrics),
        "standard_capital_dominant_new": clean(dominant_new_standard.metrics),
        "independent_new_selected_days": {
            symbol: int((challenger_bundle.weights[symbol] > 0.0).sum())
            for symbol in challengers
            if int((challenger_bundle.weights[symbol] > 0.0).sum()) > 0
        },
        "dominant_new_selected_days": {
            symbol: int((dominant_challenger_bundle.weights[symbol] > 0.0).sum())
            for symbol in challengers
            if int((dominant_challenger_bundle.weights[symbol] > 0.0).sum()) > 0
        },
        "new_etf_only_rule": (
            "Use 20% expanded-strategy capital only while its selected target is "
            "a newly added ETF. Otherwise the whole account follows the main strategy."
        ),
        "daily_rule": (
            "Only trade when the combined target changes. Do not rebalance tiny "
            "80/20 price drift while both target ETFs remain unchanged."
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path.to_csv(OUTPUT / "allocation_path.csv", encoding="utf-8-sig")
    feasibility.to_csv(
        OUTPUT / "small_capital_two_position_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    episodes.to_csv(
        OUTPUT / "divergence_episodes.csv", index=False, encoding="utf-8-sig"
    )
    selected_new_feasibility.to_csv(
        OUTPUT / "selected_new_small_capital_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean(execution_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(clean(execution_summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
