"""验收55只扩展观察池及其近5年买卖差异。

冻结原45只主策略与全部动量、AI、退出、成本和T+1执行规则，仅比较：

1. 当前51只扩展观察：原45只 + 现有6只新增ETF；
2. 候选55只扩展观察：再加入工业母机、央企红利、东南亚科技、标普油气；
3. 两者都只在扩展观察最终选中新增ETF时使用20%资金，其余时间100%跟随主策略。

用法：
    PYTHONPATH=src python3 experiments/universe_55_acceptance_study.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from universe_architecture_v2_study import (  # noqa: E402
    cash_management,
    clean,
    load_yaml,
    run_variant,
)
from etf_rotation.backtest import run_backtest  # noqa: E402
from etf_rotation.data import FIELDS, fetch_easy_tdx, symbol_key  # noqa: E402
from etf_rotation.evaluation import realized_round_trips  # noqa: E402
from etf_rotation.execution import execution_project, period_metrics  # noqa: E402
from etf_rotation.sentiment import load_sentiment_matrices  # noqa: E402


FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
BASE_PRICES = ROOT / "market_data" / "prices"
RESEARCH_PRICES = ROOT / "scratch" / "universe_55_prices"
OUTPUT = ROOT / "results" / "research" / "universe_55_acceptance"
MAIN_SHARE = 0.80
FIVE_YEAR_START = "2021-07-29"

EXTRA_ITEMS = [
    {"code": "159667", "market": "SZ", "name": "工业母机ETF国泰", "category": "高端制造"},
    {"code": "561580", "market": "SH", "name": "央企红利ETF华泰柏瑞", "category": "风格红利"},
    {"code": "513730", "market": "SH", "name": "东南亚科技ETF华泰柏瑞", "category": "港股与亚洲"},
    {"code": "159518", "market": "SZ", "name": "标普油气ETF嘉实", "category": "周期资源"},
]


def load_combined_panel(market: dict, extra_items: list[dict]) -> dict[str, pd.DataFrame]:
    fetch_config = {
        "project": {**market["project"], "data_count": 3000},
        "universe": extra_items,
        "market_proxies": [],
        "benchmark": market["benchmark"],
    }
    fetch_easy_tdx(fetch_config, RESEARCH_PRICES, force=False)
    extra_symbols = {symbol_key(item) for item in extra_items}
    instruments = {
        symbol_key(item): item
        for item in [*market["universe"], *extra_items, market["benchmark"]]
    }
    raw: dict[str, pd.DataFrame] = {}
    for symbol in instruments:
        folder = RESEARCH_PRICES if symbol in extra_symbols else BASE_PRICES
        frame = pd.read_csv(folder / f"{symbol}.csv", parse_dates=["datetime"])
        raw[symbol] = frame.drop_duplicates("datetime").set_index("datetime").sort_index()

    benchmark = symbol_key(market["benchmark"])
    calendar = raw[benchmark].index
    calendar = calendar[
        (calendar >= pd.Timestamp(market["project"]["warmup_start"]))
        & (calendar <= pd.Timestamp(market["project"]["data_end"]))
    ]
    return {
        field: pd.DataFrame(
            {symbol: frame[field].reindex(calendar) for symbol, frame in raw.items()},
            index=calendar,
        ).astype(float)
        for field in FIELDS
    }


def align_weights(weights: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    return weights.reindex(columns=symbols, fill_value=0.0).fillna(0.0)


def new_etf_only_weights(
    core_weights: pd.DataFrame,
    expanded_weights: pd.DataFrame,
    new_symbols: list[str],
    all_symbols: list[str],
    main_share: float = MAIN_SHARE,
) -> pd.DataFrame:
    core = align_weights(core_weights, all_symbols)
    expanded = align_weights(expanded_weights, all_symbols)
    active = expanded[new_symbols].sum(axis=1).gt(0.0)
    output = core.copy()
    output.loc[active] = (
        core.loc[active] * main_share
        + expanded.loc[active] * (1.0 - main_share)
    )
    if float(output.sum(axis=1).max()) > 1.0 + 1e-10:
        raise AssertionError("combined target exceeds 100%")
    return output


def selected_symbol(weights: pd.DataFrame) -> pd.Series:
    def select(row: pd.Series) -> str:
        held = row[row > 0.0]
        return "CASH" if held.empty else str(held.idxmax())

    return weights.apply(select, axis=1)


def target_description(row: pd.Series) -> str:
    held = row[row > 1e-10]
    if held.empty:
        return "100%现金"
    return "+".join(f"{symbol.split('.')[0]} {weight:.0%}" for symbol, weight in held.items())


def target_difference_episodes(
    current: pd.DataFrame,
    proposed: pd.DataFrame,
    expanded_current: pd.DataFrame,
    expanded_proposed: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    start: str,
) -> pd.DataFrame:
    dates = current.index[current.index >= pd.Timestamp(start)]
    current = current.loc[dates]
    proposed = proposed.loc[dates]
    current_pick = selected_symbol(expanded_current.loc[dates])
    proposed_pick = selected_symbol(expanded_proposed.loc[dates])
    different = (~current.round(10).eq(proposed.round(10))).any(axis=1)
    state = pd.DataFrame(
        {
            "different": different,
            "current_pick": current_pick,
            "proposed_pick": proposed_pick,
            "current_target": current.apply(target_description, axis=1),
            "proposed_target": proposed.apply(target_description, axis=1),
        },
        index=dates,
    )
    state_key = (
        state["different"].astype(str)
        + "|" + state["current_pick"]
        + "|" + state["proposed_pick"]
        + "|" + state["current_target"]
        + "|" + state["proposed_target"]
    )
    groups = state_key.ne(state_key.shift()).cumsum()
    rows = []
    for _, episode in state[state["different"]].groupby(groups[state["different"]]):
        signal_start = episode.index[0]
        signal_end = episode.index[-1]
        start_loc = calendar.get_loc(signal_start)
        end_loc = calendar.get_loc(signal_end)
        rows.append(
            {
                "signal_start": signal_start,
                "signal_end": signal_end,
                "execution_start": calendar[min(start_loc + 1, len(calendar) - 1)],
                "execution_end": calendar[min(end_loc + 1, len(calendar) - 1)],
                "signal_days": int(len(episode)),
                "current_expanded_pick": episode["current_pick"].iloc[0],
                "proposed_expanded_pick": episode["proposed_pick"].iloc[0],
                "current_combined_target": episode["current_target"].iloc[0],
                "proposed_combined_target": episode["proposed_target"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def metrics_row(name: str, result, initial_capital: float, end: str) -> dict:
    five = period_metrics(result.equity, FIVE_YEAR_START, end, initial_capital)
    return {
        "variant": name,
        **result.metrics,
        "five_year_total_return": five["total_return"],
        "five_year_cagr": five["cagr"],
        "five_year_max_drawdown": five["max_drawdown"],
        "five_year_sharpe": five["sharpe"],
    }


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    formal = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    research_market = copy.deepcopy(market)
    research_market["universe"] = [*market["universe"], *EXTRA_ITEMS]
    panel = load_combined_panel(market, EXTRA_ITEMS)

    core_size = int(formal["enhanced_selection"]["universe_architecture"]["core_pool_size"])
    current_items = market["universe"]
    core_symbols = [symbol_key(item) for item in current_items[:core_size]]
    current_new = [
        str(symbol)
        for symbol in formal["enhanced_selection"]["universe_architecture"]["challenger_symbols"]
    ]
    extra_symbols = [symbol_key(item) for item in EXTRA_ITEMS]
    proposed_new = [*current_new, *extra_symbols]
    current_symbols = [*core_symbols, *current_new]
    proposed_symbols = [*core_symbols, *proposed_new]
    all_symbols = proposed_symbols
    categories = {symbol_key(item): item["category"] for item in research_market["universe"]}
    sentiment, available = load_sentiment_matrices(FEATURES, panel["close"].index, all_symbols)

    def build(key: str, symbols: list[str], challengers: list[str]):
        return run_variant(
            key,
            research_market,
            formal,
            panel,
            all_symbols,
            core_symbols,
            categories,
            sentiment,
            available,
            symbols=symbols,
            mode="core_anchor_challenger",
            challengers=challengers,
        )

    core = build("core45", core_symbols, [])
    expanded51 = build("expanded51", current_symbols, current_new)
    expanded55 = build("expanded55", proposed_symbols, proposed_new)
    weights51 = new_etf_only_weights(core["weights"], expanded51["weights"], current_new, all_symbols)
    weights55 = new_etf_only_weights(core["weights"], expanded55["weights"], proposed_new, all_symbols)

    eligibility = expanded55["eligibility"].reindex(columns=all_symbols, fill_value=False)
    premium_symbols = [
        symbol for symbol in all_symbols
        if symbol.split(".")[0].startswith("513") or symbol in {"159941.SZ", "159518.SZ"}
    ]
    project = execution_project(
        research_market,
        premium_symbols,
        eligibility.shift(1, fill_value=False).astype(bool),
    )
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    capital = float(market["project"]["initial_capital"])

    core_result = run_backtest(
        "core45", panel, align_weights(core["weights"], all_symbols), start, end, project,
        cash_management=cash_management(formal),
    )
    current_result = run_backtest(
        "current51_new20", panel, weights51, start, end, project,
        cash_management=cash_management(formal),
    )
    proposed_result = run_backtest(
        "proposed55_new20", panel, weights55, start, end, project,
        cash_management=cash_management(formal),
    )

    small_project = copy.deepcopy(project)
    small_project["initial_capital"] = 12_000.0
    current_small = run_backtest(
        "current51_new20_12000", panel, weights51, start, end, small_project,
        cash_management=cash_management(formal),
    )
    proposed_small = run_backtest(
        "proposed55_new20_12000", panel, weights55, start, end, small_project,
        cash_management=cash_management(formal),
    )
    core_small = run_backtest(
        "core45_12000",
        panel,
        align_weights(core["weights"], all_symbols),
        start,
        end,
        small_project,
        cash_management=cash_management(formal),
    )

    allocation_rows = []
    expanded55_pick_for_grid = selected_symbol(expanded55["weights"])
    active_signal_dates = expanded55_pick_for_grid.loc[FIVE_YEAR_START:end]
    active_signal_dates = active_signal_dates[
        active_signal_dates.isin(proposed_new)
    ]
    calendar_for_grid = panel["close"].index
    for satellite_share in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        grid_weights = new_etf_only_weights(
            core["weights"],
            expanded55["weights"],
            proposed_new,
            all_symbols,
            main_share=1.0 - satellite_share,
        )
        standard_grid = run_backtest(
            f"proposed55_new{int(satellite_share * 100)}",
            panel,
            grid_weights,
            start,
            end,
            project,
            cash_management=cash_management(formal),
        )
        small_grid = run_backtest(
            f"proposed55_new{int(satellite_share * 100)}_12000",
            panel,
            grid_weights,
            start,
            end,
            small_project,
            cash_management=cash_management(formal),
        )
        actual_weights = []
        success = []
        for signal_date, symbol in active_signal_dates.items():
            location = calendar_for_grid.get_loc(signal_date)
            if location + 1 >= len(calendar_for_grid):
                continue
            execution_date = calendar_for_grid[location + 1]
            actual = float(
                small_grid.actual_weights.loc[execution_date].get(str(symbol), 0.0)
            )
            actual_weights.append(actual)
            success.append(actual > 0.0)
        standard_row = metrics_row(
            "standard", standard_grid, capital, end
        )
        small_row = metrics_row("small", small_grid, 12_000.0, end)
        allocation_rows.append(
            {
                "satellite_share": satellite_share,
                "standard_total_return": standard_row["total_return"],
                "standard_cagr": standard_row["cagr"],
                "standard_max_drawdown": standard_row["max_drawdown"],
                "standard_sharpe": standard_row["sharpe"],
                "standard_five_year_total_return": standard_row["five_year_total_return"],
                "standard_five_year_cagr": standard_row["five_year_cagr"],
                "standard_five_year_max_drawdown": standard_row["five_year_max_drawdown"],
                "standard_terminal_wealth_gap_vs_core_pct": float(
                    standard_grid.equity.iloc[-1] / core_result.equity.iloc[-1] - 1.0
                ),
                "small_total_return": small_row["total_return"],
                "small_five_year_total_return": small_row["five_year_total_return"],
                "small_max_drawdown": small_row["max_drawdown"],
                "small_terminal_wealth_gap_vs_core_pct": float(
                    small_grid.equity.iloc[-1] / core_small.equity.iloc[-1] - 1.0
                ),
                "small_trade_count": small_grid.metrics["trade_count"],
                "small_total_fees": small_grid.metrics["total_fees"],
                "small_extra_fees_vs_core": float(
                    small_grid.metrics["total_fees"] - core_small.metrics["total_fees"]
                ),
                "small_signal_execution_success_rate": float(pd.Series(success).mean()),
                "small_actual_satellite_weight_median": float(pd.Series(actual_weights).median()),
                "small_weight_shortfall_median_pp": float(
                    (satellite_share - pd.Series(actual_weights)).median() * 100.0
                ),
            }
        )
    allocation_sensitivity = pd.DataFrame(allocation_rows)

    metrics = pd.DataFrame(
        [
            metrics_row("core45", core_result, capital, end),
            metrics_row("current51_new20", current_result, capital, end),
            metrics_row("proposed55_new20", proposed_result, capital, end),
            metrics_row("current51_new20_12000", current_small, 12_000.0, end),
            metrics_row("proposed55_new20_12000", proposed_small, 12_000.0, end),
        ]
    )

    episodes = target_difference_episodes(
        weights51,
        weights55,
        expanded51["weights"],
        expanded55["weights"],
        panel["close"].index,
        FIVE_YEAR_START,
    )
    if not episodes.empty:
        for index, episode in episodes.iterrows():
            before_current = current_result.equity[current_result.equity.index < episode["execution_start"]]
            before_proposed = proposed_result.equity[proposed_result.equity.index < episode["execution_start"]]
            current_window = current_result.equity.loc[episode["execution_start"] : episode["execution_end"]]
            proposed_window = proposed_result.equity.loc[episode["execution_start"] : episode["execution_end"]]
            current_return = (
                float(current_window.iloc[-1] / before_current.iloc[-1] - 1.0)
                if not before_current.empty and not current_window.empty else float("nan")
            )
            proposed_return = (
                float(proposed_window.iloc[-1] / before_proposed.iloc[-1] - 1.0)
                if not before_proposed.empty and not proposed_window.empty else float("nan")
            )
            episodes.loc[index, "current_return"] = current_return
            episodes.loc[index, "proposed_return"] = proposed_return
            episodes.loc[index, "proposed_minus_current_pp"] = (
                proposed_return - current_return
            ) * 100.0

    round_trips = realized_round_trips(proposed_result.trades, panel["close"].index)
    extra_round_trips = round_trips[
        round_trips["symbol"].isin(extra_symbols)
        & (pd.to_datetime(round_trips["entry_date"]) >= pd.Timestamp(FIVE_YEAR_START))
    ].copy()
    small_round_trips = realized_round_trips(
        proposed_small.trades, panel["close"].index
    )
    small_extra_round_trips = small_round_trips[
        small_round_trips["symbol"].isin(extra_symbols)
        & (pd.to_datetime(small_round_trips["entry_date"]) >= pd.Timestamp(FIVE_YEAR_START))
    ].copy()
    expanded55_pick = selected_symbol(expanded55["weights"])
    selected_days = {
        symbol: int((expanded55_pick == symbol).loc[FIVE_YEAR_START:end].sum())
        for symbol in proposed_new
        if int((expanded55_pick == symbol).loc[FIVE_YEAR_START:end].sum()) > 0
    }
    feasibility_rows = []
    calendar = panel["close"].index
    for signal_date in expanded55_pick.loc[FIVE_YEAR_START:end].index:
        symbol = str(expanded55_pick.loc[signal_date])
        if symbol not in extra_symbols:
            continue
        location = calendar.get_loc(signal_date)
        if location + 1 >= len(calendar):
            continue
        execution_date = calendar[location + 1]
        actual_weight = float(
            proposed_small.actual_weights.loc[execution_date].get(symbol, 0.0)
        )
        feasibility_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "symbol": symbol,
                "actual_weight_close": actual_weight,
                "position_present": actual_weight > 0.0,
            }
        )
    feasibility = pd.DataFrame(feasibility_rows)

    metrics_current = metrics.set_index("variant").loc["current51_new20"]
    metrics_proposed = metrics.set_index("variant").loc["proposed55_new20"]
    summary = {
        "status": "research_only_formal_strategy_unchanged",
        "data_through": end,
        "comparison": "current 51 vs proposed 55, both only use 20% when expanded selection picks a newly added ETF",
        "five_year_start": FIVE_YEAR_START,
        "extra_symbols": extra_symbols,
        "selected_days_last5y": selected_days,
        "target_difference_episode_count_last5y": int(len(episodes)),
        "completed_extra_round_trips_last5y": int(len(extra_round_trips)),
        "small_capital_completed_extra_round_trips_last5y": int(
            len(small_extra_round_trips)
        ),
        "small_capital_extra_signal_checks": int(len(feasibility)),
        "small_capital_extra_position_success_rate": float(
            feasibility["position_present"].mean()
        ) if not feasibility.empty else None,
        "small_capital_extra_weight_median": float(
            feasibility["actual_weight_close"].median()
        ) if not feasibility.empty else None,
        "full_period_total_return_change_pp": float(
            (metrics_proposed["total_return"] - metrics_current["total_return"]) * 100.0
        ),
        "full_period_max_drawdown_change_pp": float(
            (metrics_proposed["max_drawdown"] - metrics_current["max_drawdown"]) * 100.0
        ),
        "five_year_total_return_change_pp": float(
            (metrics_proposed["five_year_total_return"] - metrics_current["five_year_total_return"]) * 100.0
        ),
        "five_year_max_drawdown_change_pp": float(
            (metrics_proposed["five_year_max_drawdown"] - metrics_current["five_year_max_drawdown"]) * 100.0
        ),
        "trade_count_change": int(
            metrics_proposed["trade_count"] - metrics_current["trade_count"]
        ),
        "small_capital_trade_count_change": int(
            proposed_small.metrics["trade_count"] - current_small.metrics["trade_count"]
        ),
        "interpretation_rule": "Do not accept or reject membership because historical return improved; judge coverage, risk containment and executability.",
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    allocation_sensitivity.to_csv(
        OUTPUT / "allocation_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    episodes.to_csv(OUTPUT / "target_difference_episodes_last5y.csv", index=False, encoding="utf-8-sig")
    extra_round_trips.to_csv(OUTPUT / "extra_etf_round_trips_last5y.csv", index=False, encoding="utf-8-sig")
    small_extra_round_trips.to_csv(
        OUTPUT / "extra_etf_round_trips_last5y_12000.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feasibility.to_csv(
        OUTPUT / "extra_etf_execution_checks_12000.csv",
        index=False,
        encoding="utf-8-sig",
    )
    current_result.trades.to_csv(OUTPUT / "current51_trades.csv", index=False, encoding="utf-8-sig")
    proposed_result.trades.to_csv(OUTPUT / "proposed55_trades.csv", index=False, encoding="utf-8-sig")
    current_small.trades.to_csv(
        OUTPUT / "current51_trades_12000.csv", index=False, encoding="utf-8-sig"
    )
    proposed_small.trades.to_csv(
        OUTPUT / "proposed55_trades_12000.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
    print(metrics.to_string(index=False))
    print("\nAllocation sensitivity")
    print(allocation_sensitivity.to_string(index=False))
    print("\nTarget difference episodes")
    print(episodes.to_string(index=False) if not episodes.empty else "none")
    print("\nExtra ETF round trips")
    print(extra_round_trips.to_string(index=False) if not extra_round_trips.empty else "none")
    print("\nExtra ETF round trips at 12,000")
    print(
        small_extra_round_trips.to_string(index=False)
        if not small_extra_round_trips.empty else "none"
    )


if __name__ == "__main__":
    main()
