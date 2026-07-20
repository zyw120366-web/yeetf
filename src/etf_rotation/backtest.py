from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    daily_returns: pd.Series
    trades: pd.DataFrame
    actual_weights: pd.DataFrame
    metrics: dict[str, float]


def _metrics(
    equity: pd.Series,
    turnover: float,
    trade_count: int,
    avg_exposure: float,
    initial_equity: float,
) -> dict[str, float]:
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / initial_equity - 1
    returns = returns.dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 252)
    total = equity.iloc[-1] / initial_equity - 1
    cagr = (equity.iloc[-1] / initial_equity) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    downside = returns.clip(upper=0).std() * np.sqrt(252)
    sortino = returns.mean() * 252 / downside if downside > 0 else 0.0
    drawdown = equity / equity.cummax().clip(lower=initial_equity) - 1
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "annual_volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "annual_turnover": float(turnover / years),
        "trade_count": float(trade_count),
        "average_exposure": float(avg_exposure),
    }


def run_backtest(
    name: str,
    panel: dict[str, pd.DataFrame],
    signal_weights: pd.DataFrame,
    start: str,
    end: str,
    project: dict,
    *,
    cash_annual_rate: float = 0.0,
    cash_management: dict | None = None,
) -> BacktestResult:
    calendar = panel["close"].index
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    test_dates = calendar[(calendar >= start_ts) & (calendar <= end_ts)]
    symbols = list(signal_weights.columns)
    close = panel["close"][symbols].ffill()
    open_px = panel["open"][symbols]
    capital = float(project["initial_capital"])
    lot = int(project["lot_size"])
    commission = float(project["commission_rate"])
    min_commission = float(project["minimum_commission"])
    slippage = float(project["slippage_rate"])
    symbol_costs = project.get("symbol_costs", {})
    detect_locked_limits = bool(project.get("detect_locked_limits", False))
    locked_limit_threshold = float(project.get("locked_limit_threshold", 0.095))
    max_participation_rate = project.get("max_participation_rate")
    max_participation_rate = (
        float(max_participation_rate) if max_participation_rate is not None else None
    )
    liquidity_lookback_days = int(project.get("liquidity_lookback_days", 20))
    buy_eligibility = project.get("buy_eligibility")
    if buy_eligibility is not None:
        if not isinstance(buy_eligibility, pd.DataFrame):
            raise TypeError("buy_eligibility must be a DataFrame")
        if set(symbols) - set(buy_eligibility.columns):
            raise KeyError("buy_eligibility is missing signal symbols")
    if not np.isfinite(cash_annual_rate) or cash_annual_rate < 0:
        raise ValueError("cash annual rate must be a finite non-negative value")
    if cash_management is not None:
        cash_annual_rate = float(cash_management["annual_rate"])
        if not np.isfinite(cash_annual_rate) or cash_annual_rate < 0:
            raise ValueError("cash-management annual rate must be finite and non-negative")

    shares = pd.Series(0.0, index=symbols)
    cash = capital
    last_close = close.loc[test_dates[0]].copy()
    equity_records: list[tuple[pd.Timestamp, float]] = []
    weight_records: list[pd.Series] = []
    trade_records: list[dict] = []
    total_turnover = 0.0
    total_fees = 0.0
    slippage_cost_estimate = 0.0
    blocked_order_count = 0
    limit_blocked_order_count = 0
    eligibility_blocked_order_count = 0
    liquidity_limited_order_count = 0
    cash_interest_income = 0.0
    cash_management_fees = 0.0
    cash_interest_calendar_days = 0
    last_applied_target = pd.Series(np.nan, index=symbols)
    previous_date: pd.Timestamp | None = None

    for date in test_dates:
        # Idle cash is assumed to enter the cash-management product after the
        # previous close and to be available again before this market open.
        # Use Actual/365 simple interest so weekends and market holidays earn
        # interest without introducing an intraday ETF return assumption.
        if previous_date is not None and cash > 0:
            calendar_days = int((date - previous_date).days)
            if calendar_days > 0 and cash_annual_rate > 0:
                invested = cash
                fee = 0.0
                if cash_management is not None:
                    order_lot = float(cash_management["order_lot"])
                    minimum = float(cash_management["minimum_order"])
                    fee_rate = float(cash_management["fee_rate"])
                    invested = (
                        np.floor((cash / (1.0 + fee_rate)) / order_lot) * order_lot
                        if cash > minimum else 0.0
                    )
                    fee = invested * fee_rate
                interest = invested * cash_annual_rate * calendar_days / 365.0
                cash += interest - fee
                cash_interest_income += interest
                cash_management_fees += fee
                cash_interest_calendar_days += calendar_days
        loc = calendar.get_loc(date)
        raw_open = open_px.loc[date]
        mark_open = raw_open.fillna(last_close)
        equity_open = float(cash + (shares * mark_open).sum())
        signal_date = calendar[loc - 1] if loc > 0 else date
        target_w = signal_weights.loc[signal_date].fillna(0).clip(lower=0)
        if target_w.sum() > 1:
            target_w /= target_w.sum()

        target_changed = last_applied_target.isna().any() or not np.allclose(
            target_w.to_numpy(), last_applied_target.to_numpy(), atol=1e-10, rtol=0
        )
        desired_shares = shares.copy()
        tradeable = raw_open.notna() & (raw_open > 0)
        if target_changed:
            desired_shares.loc[tradeable] = (
                np.floor((target_w.loc[tradeable] * equity_open / raw_open.loc[tradeable]) / lot) * lot
            )
        requested_delta = desired_shares - shares

        # A daily bar with open == high == low at (roughly) a 10% or 20%
        # price limit is not safely executable in the locked direction.  A
        # buy at a locked limit-up and a sell at a locked limit-down remain
        # pending instead of being filled at a price that was only quoted.
        buyable = tradeable.copy()
        sellable = tradeable.copy()
        locked_up = pd.Series(False, index=symbols)
        locked_down = pd.Series(False, index=symbols)
        if detect_locked_limits and loc > 0 and {"high", "low"} <= set(panel):
            high = panel["high"].loc[date, symbols]
            low = panel["low"].loc[date, symbols]
            previous_close = close.iloc[loc - 1]
            one_price = (
                raw_open.notna()
                & high.notna()
                & low.notna()
                & np.isclose(raw_open, high, atol=5e-5, rtol=0)
                & np.isclose(raw_open, low, atol=5e-5, rtol=0)
            )
            open_return = raw_open / previous_close - 1.0
            locked_up = one_price & (open_return >= locked_limit_threshold)
            locked_down = one_price & (open_return <= -locked_limit_threshold)
            buyable &= ~locked_up
            sellable &= ~locked_down

        eligibility_blocked = pd.Series(False, index=symbols)
        if buy_eligibility is not None:
            allowed_today = (
                buy_eligibility.reindex(index=[date], columns=symbols)
                .iloc[0]
                .fillna(False)
                .astype(bool)
            )
            eligibility_blocked = tradeable & ~allowed_today
            buyable &= allowed_today

        pending_execution = bool(
            ((requested_delta > 0) & ~buyable).any()
            or ((requested_delta < 0) & ~sellable).any()
        )
        newly_blocked = int(
            (((requested_delta > 0) & ~buyable) | ((requested_delta < 0) & ~sellable)).sum()
        )
        blocked_order_count += newly_blocked
        limit_blocked_order_count += int(
            (((requested_delta > 0) & locked_up) | ((requested_delta < 0) & locked_down)).sum()
        )
        eligibility_blocked_order_count += int(
            ((requested_delta > 0) & eligibility_blocked).sum()
        )

        delta = requested_delta.copy()
        delta[(delta > 0) & ~buyable] = 0.0
        delta[(delta < 0) & ~sellable] = 0.0
        day_notional = 0.0

        capacity_notional = pd.Series(np.inf, index=symbols)
        if max_participation_rate is not None and max_participation_rate > 0 and "amount" in panel:
            history_start = max(0, loc - liquidity_lookback_days)
            known_amount = panel["amount"].iloc[history_start:loc][symbols]
            if not known_amount.empty:
                capacity_notional = (
                    known_amount.median(skipna=True).fillna(0.0) * max_participation_rate
                )

        def costs_for(symbol: str) -> tuple[float, float, float]:
            overrides = symbol_costs.get(symbol, {})
            return (
                float(overrides.get("commission_rate", commission)),
                float(overrides.get("slippage_rate", slippage)),
                float(overrides.get("minimum_commission", min_commission)),
            )

        def capacity_qty(symbol: str, raw_price: float) -> float:
            nonlocal liquidity_limited_order_count, pending_execution
            if not np.isfinite(capacity_notional[symbol]):
                return np.inf
            quantity = (
                np.floor((max(0.0, capacity_notional[symbol]) / raw_price) / lot) * lot
            )
            return float(quantity)

        # Sells first; ETF transactions have no stamp duty.
        for symbol in delta[delta < 0].index:
            wanted = float(-delta[symbol])
            cap = capacity_qty(symbol, float(raw_open[symbol]))
            qty = min(wanted, cap)
            if qty + 1e-9 < wanted:
                liquidity_limited_order_count += 1
                pending_execution = True
            if qty <= 0:
                continue
            symbol_commission, symbol_slippage, symbol_min_commission = costs_for(symbol)
            price = float(raw_open[symbol]) * (1 - symbol_slippage)
            gross = qty * price
            fee = max(symbol_min_commission, gross * symbol_commission)
            cash += gross - fee
            shares[symbol] -= qty
            day_notional += gross
            total_fees += fee
            slippage_cost_estimate += qty * float(raw_open[symbol]) * symbol_slippage
            trade_records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "side": "SELL",
                    "qty": qty,
                    "price": price,
                    "raw_open": float(raw_open[symbol]),
                    "fee": fee,
                    "commission_rate": symbol_commission,
                    "slippage_rate": symbol_slippage,
                }
            )

        # Buys are clipped to available cash and rounded to board lots.
        for symbol in delta[delta > 0].sort_values(ascending=False).index:
            wanted = float(delta[symbol])
            symbol_commission, symbol_slippage, symbol_min_commission = costs_for(symbol)
            price = float(raw_open[symbol]) * (1 + symbol_slippage)
            cash_after_min_fee = max(0.0, cash - symbol_min_commission)
            affordable = (
                np.floor((cash_after_min_fee / (price * (1 + symbol_commission))) / lot) * lot
            )
            cap = capacity_qty(symbol, float(raw_open[symbol]))
            qty = max(0.0, min(wanted, affordable, cap))
            if qty + 1e-9 < min(wanted, affordable):
                liquidity_limited_order_count += 1
                pending_execution = True
            if qty <= 0:
                continue
            gross = qty * price
            fee = max(symbol_min_commission, gross * symbol_commission)
            if gross + fee > cash:
                continue
            cash -= gross + fee
            shares[symbol] += qty
            day_notional += gross
            total_fees += fee
            slippage_cost_estimate += qty * float(raw_open[symbol]) * symbol_slippage
            trade_records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "side": "BUY",
                    "qty": qty,
                    "price": price,
                    "raw_open": float(raw_open[symbol]),
                    "fee": fee,
                    "commission_rate": symbol_commission,
                    "slippage_rate": symbol_slippage,
                }
            )

        total_turnover += day_notional / max(equity_open, 1.0)
        if target_changed:
            # If a required leg could not trade, retain a pending state so the
            # order is retried on the next open instead of silently disappearing.
            current_open_w = shares * mark_open / max(float(cash + (shares * mark_open).sum()), 1.0)
            pending_untradeable = (~tradeable) & ((target_w - current_open_w).abs() > 1e-6)
            if pending_untradeable.any() or pending_execution:
                last_applied_target[:] = np.nan
            else:
                last_applied_target = target_w.copy()
        mark_close = close.loc[date].fillna(mark_open)
        equity_close = float(cash + (shares * mark_close).sum())
        actual_w = shares * mark_close / max(equity_close, 1.0)
        actual_w.name = date
        weight_records.append(actual_w)
        equity_records.append((date, equity_close))
        last_close = mark_close
        previous_date = date

    equity = pd.Series(dict(equity_records), name=name).sort_index()
    actual_weights = pd.DataFrame(weight_records)
    trades = pd.DataFrame(trade_records)
    avg_exposure = float(actual_weights.sum(axis=1).mean())
    metrics = _metrics(equity, total_turnover, len(trades), avg_exposure, capital)
    metrics.update(
        {
            "total_fees": float(total_fees),
            "slippage_cost_estimate": float(slippage_cost_estimate),
            "blocked_order_count": float(blocked_order_count),
            "limit_blocked_order_count": float(limit_blocked_order_count),
            "eligibility_blocked_order_count": float(eligibility_blocked_order_count),
            "liquidity_limited_order_count": float(liquidity_limited_order_count),
        }
    )
    if cash_annual_rate > 0:
        metrics.update(
            {
                "cash_management_annual_rate": float(cash_annual_rate),
                "cash_interest_income": float(cash_interest_income),
                "cash_interest_calendar_days": float(cash_interest_calendar_days),
                "cash_management_fees": float(cash_management_fees),
            }
        )
    return BacktestResult(name, equity, equity.pct_change().fillna(0), trades, actual_weights, metrics)


def buy_and_hold_benchmark(
    name: str, close: pd.Series, start: str, end: str, initial_capital: float
) -> BacktestResult:
    px = close.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
    equity = initial_capital * px / px.iloc[0]
    metrics = _metrics(equity, 0.0, 1, 1.0, initial_capital)
    weights = pd.DataFrame({close.name: 1.0}, index=equity.index)
    return BacktestResult(name, equity.rename(name), equity.pct_change().fillna(0), pd.DataFrame(), weights, metrics)
