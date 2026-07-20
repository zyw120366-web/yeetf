from __future__ import annotations

import numpy as np
import pandas as pd


def entry_eligibility(
    panel: dict[str, pd.DataFrame], symbols: list[str], rules: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the point-in-time listing and liquidity gate for new entries."""

    close = panel["close"][symbols]
    amount = panel["amount"][symbols]
    listed_sessions = close.notna().cumsum()
    window = int(rules["liquidity_lookback_days"])
    trailing_amount = amount.rolling(window, min_periods=window).median()
    eligible = (
        listed_sessions.ge(int(rules["listing_warmup_days"]))
        & trailing_amount.ge(float(rules["minimum_entry_amount"]))
    ).fillna(False)
    return eligible, listed_sessions, trailing_amount


def execution_project(
    market: dict,
    premium_sensitive: list[str],
    buy_eligibility: pd.DataFrame,
) -> dict:
    """Build the one frozen cost and execution model used everywhere."""

    execution = market["execution"]
    default = execution["fixed_default"]
    special = execution["fixed_premium_sensitive"]
    minimum = float(execution["minimum_commission"])
    return {
        "initial_capital": float(market["project"]["initial_capital"]),
        "lot_size": int(market["project"]["lot_size"]),
        "commission_rate": float(default["commission_rate"]),
        "slippage_rate": float(default["slippage_rate"]),
        "minimum_commission": minimum,
        "symbol_costs": {
            symbol: {
                "commission_rate": float(special["commission_rate"]),
                "slippage_rate": float(special["slippage_rate"]),
                "minimum_commission": minimum,
            }
            for symbol in premium_sensitive
        },
        "detect_locked_limits": bool(execution["detect_locked_limits"]),
        "locked_limit_threshold": float(execution["locked_limit_threshold"]),
        "max_participation_rate": float(execution["max_participation_rate"]),
        "liquidity_lookback_days": int(execution["liquidity_lookback_days"]),
        "buy_eligibility": buy_eligibility,
    }


def period_metrics(
    equity: pd.Series, start: str, end: str, initial_capital: float
) -> dict[str, float]:
    """Calculate a sub-period from the equity immediately before it starts."""

    window = equity.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if window.empty:
        return {key: np.nan for key in (
            "total_return", "cagr", "annual_volatility", "sharpe", "max_drawdown"
        )}
    previous = equity[equity.index < window.index[0]]
    base = float(previous.iloc[-1]) if not previous.empty else float(initial_capital)
    anchor = pd.Series([base], index=[window.index[0] - pd.Timedelta(days=1)])
    path = pd.concat([anchor, window])
    returns = path.pct_change().dropna()
    total = float(window.iloc[-1] / base - 1.0)
    years = max((window.index[-1] - window.index[0]).days / 365.25, 1 / 252)
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else 0.0
    )
    return {
        "total_return": total,
        "cagr": float((1.0 + total) ** (1.0 / years) - 1.0),
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float((path / path.cummax() - 1.0).min()),
    }
