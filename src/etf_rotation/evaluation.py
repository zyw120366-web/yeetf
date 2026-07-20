from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult


def realized_round_trips(
    trades: pd.DataFrame,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Aggregate partial fills into completed, fee-aware buy/sell rounds.

    A round starts when one symbol's position changes from zero to positive and
    ends when it returns to zero.  Multiple liquidity-limited fills are kept as
    one economic decision.  Prices already include simulated slippage; fees are
    added to buy cost and deducted from sell proceeds before return is measured.
    Open positions are intentionally excluded because they do not yet have a
    realised win or loss.
    """

    columns = [
        "symbol",
        "entry_date",
        "entry_completed_date",
        "exit_date",
        "exit_completed_date",
        "buy_fill_count",
        "sell_fill_count",
        "quantity",
        "average_buy_price",
        "average_sell_price",
        "buy_cost",
        "sell_proceeds",
        "total_fees",
        "net_pnl",
        "net_return",
        "holding_days",
        "outcome",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    calendar_locations = (
        {pd.Timestamp(date): index for index, date in enumerate(calendar)}
        if calendar is not None
        else {}
    )
    states: dict[str, dict[str, object]] = {}
    completed: list[dict[str, object]] = []
    ordered = trades.sort_values("date", kind="stable")

    for trade in ordered.itertuples(index=False):
        symbol = str(trade.symbol)
        side = str(trade.side).upper()
        date = pd.Timestamp(trade.date)
        quantity = float(trade.qty)
        price = float(trade.price)
        fee = float(getattr(trade, "fee", 0.0))
        if quantity <= 0:
            continue

        if side == "BUY":
            state = states.get(symbol)
            if state is None:
                state = {
                    "position": 0.0,
                    "entry_date": date,
                    "entry_completed_date": date,
                    "exit_date": None,
                    "exit_completed_date": None,
                    "buy_fill_count": 0,
                    "sell_fill_count": 0,
                    "buy_quantity": 0.0,
                    "sell_quantity": 0.0,
                    "buy_notional": 0.0,
                    "sell_notional": 0.0,
                    "buy_fees": 0.0,
                    "sell_fees": 0.0,
                }
                states[symbol] = state
            state["position"] = float(state["position"]) + quantity
            state["entry_completed_date"] = date
            state["buy_fill_count"] = int(state["buy_fill_count"]) + 1
            state["buy_quantity"] = float(state["buy_quantity"]) + quantity
            state["buy_notional"] = float(state["buy_notional"]) + quantity * price
            state["buy_fees"] = float(state["buy_fees"]) + fee
            continue

        if side != "SELL" or symbol not in states:
            continue
        state = states[symbol]
        position = float(state["position"])
        sold = min(quantity, position)
        if sold <= 0:
            continue
        allocated_fee = fee * sold / quantity
        if state["exit_date"] is None:
            state["exit_date"] = date
        state["exit_completed_date"] = date
        state["sell_fill_count"] = int(state["sell_fill_count"]) + 1
        state["sell_quantity"] = float(state["sell_quantity"]) + sold
        state["sell_notional"] = float(state["sell_notional"]) + sold * price
        state["sell_fees"] = float(state["sell_fees"]) + allocated_fee
        state["position"] = max(0.0, position - sold)
        if float(state["position"]) > 1e-8:
            continue

        buy_quantity = float(state["buy_quantity"])
        sell_quantity = float(state["sell_quantity"])
        buy_notional = float(state["buy_notional"])
        sell_notional = float(state["sell_notional"])
        buy_fees = float(state["buy_fees"])
        sell_fees = float(state["sell_fees"])
        buy_cost = buy_notional + buy_fees
        sell_proceeds = sell_notional - sell_fees
        net_pnl = sell_proceeds - buy_cost
        net_return = sell_proceeds / buy_cost - 1.0 if buy_cost > 0 else np.nan
        entry_date = pd.Timestamp(state["entry_date"])
        exit_completed_date = pd.Timestamp(state["exit_completed_date"])
        if entry_date in calendar_locations and exit_completed_date in calendar_locations:
            holding_days = calendar_locations[exit_completed_date] - calendar_locations[entry_date]
        else:
            holding_days = (exit_completed_date - entry_date).days
        completed.append(
            {
                "symbol": symbol,
                "entry_date": entry_date,
                "entry_completed_date": pd.Timestamp(state["entry_completed_date"]),
                "exit_date": pd.Timestamp(state["exit_date"]),
                "exit_completed_date": exit_completed_date,
                "buy_fill_count": int(state["buy_fill_count"]),
                "sell_fill_count": int(state["sell_fill_count"]),
                "quantity": min(buy_quantity, sell_quantity),
                "average_buy_price": buy_notional / buy_quantity,
                "average_sell_price": sell_notional / sell_quantity,
                "buy_cost": buy_cost,
                "sell_proceeds": sell_proceeds,
                "total_fees": buy_fees + sell_fees,
                "net_pnl": net_pnl,
                "net_return": net_return,
                "holding_days": int(holding_days),
                "outcome": "胜" if net_pnl > 0 else "负" if net_pnl < 0 else "平",
            }
        )
        states.pop(symbol, None)

    return pd.DataFrame(completed, columns=columns)


def round_trip_timing(
    result: BacktestResult,
    panel: dict[str, pd.DataFrame],
    startup_lookback: int = 20,
    peak_lookahead: int = 10,
    entry_success_lookahead: int = 20,
    material_move_threshold: float = 0.05,
    label_end: str | pd.Timestamp | None = None,
    require_full_lookahead: bool = True,
) -> pd.DataFrame:
    """Evaluate entry/exit timing against hindsight anchors.

    Hindsight lows and peaks are labels used only after the backtest.  They never
    feed the signal.  Positive exit_delay_days means the sale happened after the
    local peak; negative values identify premature exits.
    """
    columns = [
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "startup_anchor_date",
        "peak_date",
        "entry_delay_days",
        "exit_delay_days",
        "holding_days",
        "entry_markup_from_base",
        "post_entry_peak_return",
        "peak_giveback",
        "move_capture_ratio",
        "false_start",
        "entry_forward_max_return_20d",
        "entry_forward_return_20d",
        "post_exit_max_return_10d",
        "fixed_horizon_false_buy",
        "material_premature_exit",
        "failed_operation",
    ]
    if result.trades.empty:
        return pd.DataFrame(columns=columns)

    calendar = panel["close"].index
    label_end_ts = pd.Timestamp(label_end) if label_end is not None else pd.Timestamp(result.equity.index[-1])
    eligible_label_dates = calendar[calendar <= label_end_ts]
    label_end_loc = int(calendar.get_loc(eligible_label_dates[-1]))
    close = panel["close"]
    positions: dict[str, float] = {}
    entries: dict[str, tuple[pd.Timestamp, float]] = {}
    completed: list[dict] = []
    for trade in result.trades.sort_values("date", kind="stable").itertuples(index=False):
        symbol = str(trade.symbol)
        before = positions.get(symbol, 0.0)
        qty = float(trade.qty)
        after = before + qty if trade.side == "BUY" else before - qty
        if before <= 0 and after > 0:
            entries[symbol] = (pd.Timestamp(trade.date), float(trade.price))
        if before > 0 and after <= 1e-9 and symbol in entries:
            entry_date, entry_price = entries.pop(symbol)
            exit_date = pd.Timestamp(trade.date)
            entry_loc = int(calendar.get_loc(entry_date))
            exit_loc = int(calendar.get_loc(exit_date))
            last_required_loc = max(
                exit_loc + peak_lookahead,
                entry_loc + entry_success_lookahead,
            )
            if require_full_lookahead and last_required_loc > label_end_loc:
                positions[symbol] = max(0.0, after)
                continue
            series = close[symbol]

            base_slice = series.iloc[max(0, entry_loc - startup_lookback) : entry_loc + 1].dropna()
            peak_slice = series.iloc[
                entry_loc : min(label_end_loc + 1, exit_loc + peak_lookahead + 1)
            ].dropna()
            entry_forward_slice = series.iloc[
                entry_loc : min(
                    label_end_loc + 1,
                    entry_loc + entry_success_lookahead + 1,
                )
            ].dropna()
            post_exit_slice = series.iloc[
                exit_loc + 1 : min(
                    label_end_loc + 1,
                    exit_loc + peak_lookahead + 1,
                )
            ].dropna()
            if base_slice.empty or peak_slice.empty or entry_forward_slice.empty:
                positions[symbol] = max(0.0, after)
                continue
            base_date = pd.Timestamp(base_slice.idxmin())
            base_price = float(base_slice.min())
            peak_date = pd.Timestamp(peak_slice.idxmax())
            peak_price = float(peak_slice.max())
            base_loc = int(calendar.get_loc(base_date))
            peak_loc = int(calendar.get_loc(peak_date))
            exit_price = float(trade.price)
            entry_forward_max_return = (
                float(entry_forward_slice.max()) / entry_price - 1.0
            )
            entry_forward_return = (
                float(entry_forward_slice.iloc[-1]) / entry_price - 1.0
            )
            post_exit_max_return = (
                float(post_exit_slice.max()) / exit_price - 1.0
                if not post_exit_slice.empty
                else np.nan
            )
            full_move = peak_price / base_price - 1.0
            realised_move = exit_price / base_price - 1.0
            false_buy = entry_forward_max_return < material_move_threshold
            premature_exit = post_exit_max_return > material_move_threshold
            completed.append(
                {
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "startup_anchor_date": base_date,
                    "peak_date": peak_date,
                    "entry_delay_days": entry_loc - base_loc,
                    "exit_delay_days": exit_loc - peak_loc,
                    "holding_days": exit_loc - entry_loc,
                    "entry_markup_from_base": entry_price / base_price - 1.0,
                    "post_entry_peak_return": peak_price / entry_price - 1.0,
                    "peak_giveback": exit_price / peak_price - 1.0,
                    "move_capture_ratio": realised_move / full_move if full_move > 1e-9 else np.nan,
                    "false_start": peak_price / entry_price - 1.0 < 0.05,
                    "entry_forward_max_return_20d": entry_forward_max_return,
                    "entry_forward_return_20d": entry_forward_return,
                    "post_exit_max_return_10d": post_exit_max_return,
                    "fixed_horizon_false_buy": false_buy,
                    "material_premature_exit": premature_exit,
                    # 一笔完整交易只记一次失败：买入后的上涨空间不足，或
                    # 卖出后很快继续上涨。两个标签都只用于事后评价，不参与信号。
                    "failed_operation": bool(false_buy or premature_exit),
                }
            )
        positions[symbol] = max(0.0, after)
    return pd.DataFrame(completed, columns=columns)


def timing_summary(timing: pd.DataFrame) -> dict[str, float]:
    if timing.empty:
        return {
            "completed_round_trips": 0.0,
            "median_entry_delay_days": np.nan,
            "median_exit_delay_days": np.nan,
            "median_peak_giveback": np.nan,
            "median_move_capture_ratio": np.nan,
            "false_start_rate": np.nan,
            "premature_exit_rate": np.nan,
            "fixed_horizon_false_buy_rate": np.nan,
            "material_premature_exit_rate": np.nan,
            "failed_operation_rate": np.nan,
        }
    return {
        "completed_round_trips": float(len(timing)),
        "median_entry_delay_days": float(timing["entry_delay_days"].median()),
        "median_exit_delay_days": float(timing["exit_delay_days"].median()),
        "median_peak_giveback": float(timing["peak_giveback"].median()),
        "median_move_capture_ratio": float(timing["move_capture_ratio"].clip(-2, 2).median()),
        "false_start_rate": float(timing["false_start"].mean()),
        "premature_exit_rate": float((timing["exit_delay_days"] < 0).mean()),
        "fixed_horizon_false_buy_rate": float(
            timing["fixed_horizon_false_buy"].mean()
        ),
        "material_premature_exit_rate": float(
            timing["material_premature_exit"].mean()
        ),
        "failed_operation_rate": float(timing["failed_operation"].mean()),
    }
