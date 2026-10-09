from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals
from run_strategies import load_yaml
from etf_rotation.execution import period_metrics


BEST = ROOT / "results" / "ye_strategy"
OUTPUT = BEST / "trade_audit.json"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if pd.isna(value):
        return None
    return value


def value(frame: pd.DataFrame, date: pd.Timestamp, symbol: str):
    result = frame.at[date, symbol]
    if pd.isna(result):
        return None
    if isinstance(result, (np.bool_, bool)):
        return bool(result)
    return float(result)


def fmt_number(item, digits: int = 2) -> str:
    return "缺失" if item is None else f"{item:.{digits}f}"


def fmt_percent(item, digits: int = 2) -> str:
    return "缺失" if item is None else f"{item:.{digits}%}"


def previous_session(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    pos = int(calendar.searchsorted(date, side="left"))
    if pos <= 0:
        raise ValueError(f"no prior trading session for {date.date()}")
    return calendar[pos - 1]


def annual_rows(equity: pd.Series, rounds: pd.DataFrame, initial_capital: float) -> list[dict]:
    rows: list[dict] = []
    for year in sorted(equity.index.year.unique()):
        year_equity = equity[equity.index.year == year]
        prior = equity[equity.index < year_equity.index[0]]
        base = float(prior.iloc[-1]) if not prior.empty else initial_capital
        path = pd.concat(
            [pd.Series([base], index=[year_equity.index[0] - pd.Timedelta(days=1)]), year_equity]
        )
        drawdown = path / path.cummax() - 1.0
        year_rounds = rounds[pd.to_datetime(rounds["exit_completed_date"]).dt.year == year]
        rows.append(
            {
                "year": int(year),
                "start_equity": base,
                "end_equity": float(year_equity.iloc[-1]),
                "return": float(year_equity.iloc[-1] / base - 1.0),
                "max_drawdown": float(drawdown.min()),
                "completed_trades": int(len(year_rounds)),
                "winning_trades": int((year_rounds["net_pnl"] > 0).sum()),
                "losing_trades": int((year_rounds["net_pnl"] <= 0).sum()),
                "net_pnl": float(year_rounds["net_pnl"].sum()),
            }
        )
    return rows


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    ye = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    close = panel["close"][symbols]
    calendar = close.index
    names = {
        symbol_key(item): {"name": str(item["name"]), "category": str(item["category"])}
        for item in market["universe"]
    }
    categories = {symbol: names[symbol]["category"] for symbol in symbols}

    sentiment, available = load_sentiment_matrices(
        ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv",
        calendar,
        symbols,
    )
    _, base, _, _, _, decision = build_ye_signals(
        panel, symbols, categories, ye, sentiment, available
    )
    r2 = decision["r2_20"]
    efficiency = decision["efficiency20"]
    breadth = decision["category_breadth"]
    gate = decision["entry_gate"]
    current_normal = decision["current_normal"]
    emerging = decision["emerging"]
    quality_extension = decision["quality_extension"]
    hot_exit_protect = decision["hot_exit_protection"]
    missing_strong = decision["missing_data_soft_exit_protection"]
    strategy_rank = decision["entry_rank"]

    timing = pd.read_csv(BEST / "timing.csv")
    rounds = pd.read_csv(BEST / "round_trips.csv").merge(
        timing[[
            "symbol", "entry_date", "exit_date",
            "entry_forward_max_return_20d", "entry_forward_return_20d",
            "post_exit_max_return_10d", "fixed_horizon_false_buy",
            "material_premature_exit", "failed_operation",
        ]],
        left_on=["symbol", "entry_date", "exit_completed_date"],
        right_on=["symbol", "entry_date", "exit_date"],
        how="left",
    )
    fills = pd.read_csv(BEST / "trades.csv")
    diagnostics = pd.read_csv(BEST / "signal_diagnostics.csv", parse_dates=["date"]).set_index("date")
    equity_frame = pd.read_csv(BEST / "equity.csv")
    equity_frame.columns = ["date", "equity"]
    equity_frame["date"] = pd.to_datetime(equity_frame["date"])
    equity = equity_frame.set_index("date")["equity"].astype(float)
    weights = pd.read_csv(BEST / "signal_weights.csv", index_col=0)
    weights.index = pd.to_datetime(weights.index)

    for column in ["entry_date", "entry_completed_date", "exit_date_x", "exit_completed_date"]:
        rounds[column] = pd.to_datetime(rounds[column])
    timing["entry_date"] = pd.to_datetime(timing["entry_date"])
    timing["exit_date"] = pd.to_datetime(timing["exit_date"])
    timing_lookup = timing.set_index(["symbol", "entry_date", "exit_date"])

    cumulative_pnl = 0.0
    audit_rows: list[dict] = []
    for number, trade in rounds.iterrows():
        symbol = str(trade["symbol"])
        entry_signal = previous_session(calendar, trade["entry_date"])
        exit_signal = previous_session(calendar, trade["exit_date_x"])
        sentiment_available = bool(available.at[entry_signal])
        if not sentiment_available:
            entry_type = "价格回退：前3名+主题广度"
        elif bool(value(quality_extension, entry_signal, symbol)):
            entry_type = "高质量趋势延伸"
        elif bool(value(emerging, entry_signal, symbol)):
            entry_type = "情绪确认的新趋势"
        elif bool(value(current_normal, entry_signal, symbol)):
            entry_type = "常规动量"
        else:
            entry_type = "其他合格路径"

        diag_entry = diagnostics.loc[entry_signal]
        diag_exit = diagnostics.loc[exit_signal]
        exit_reason_text = str(diag_exit.get("exit_reasons") or "")
        exit_reason = exit_reason_text.split(":", 1)[1] if ":" in exit_reason_text else exit_reason_text
        if not exit_reason:
            exit_reason = "换仓/期末处理"

        price_path = close.loc[trade["entry_date"] : trade["exit_completed_date"], symbol].dropna()
        average_buy_price = float(trade["average_buy_price"])
        mfe = float(price_path.max() / average_buy_price - 1.0) if not price_path.empty else None
        mae = float(price_path.min() / average_buy_price - 1.0) if not price_path.empty else None
        peak_date = price_path.idxmax() if not price_path.empty else None
        trough_date = price_path.idxmin() if not price_path.empty else None
        cumulative_pnl += float(trade["net_pnl"])

        timing_key = (symbol, trade["entry_date"], trade["exit_completed_date"])
        timing_row = timing_lookup.loc[timing_key] if timing_key in timing_lookup.index else None
        failed = trade.get("failed_operation")
        false_buy = trade.get("fixed_horizon_false_buy")
        premature = trade.get("material_premature_exit")
        if pd.isna(failed):
            failure_type = "未满前瞻窗口"
        elif bool(false_buy) and bool(premature):
            failure_type = "假启动+卖早"
        elif bool(false_buy):
            failure_type = "假启动"
        elif bool(premature):
            failure_type = "卖早"
        else:
            failure_type = "否"

        entry_reason = (
            f"{entry_type}；排名{fmt_number(value(strategy_rank, entry_signal, symbol), 0)}，"
            f"ROC20={fmt_percent(value(base.roc_short, entry_signal, symbol))}，"
            f"ROC60={fmt_percent(value(base.roc_medium, entry_signal, symbol))}，"
            f"MA120乖离={fmt_percent(value(base.ma_bias, entry_signal, symbol))}"
        )
        if sentiment_available:
            entry_reason += (
                f"；题材股{fmt_number(value(sentiment['matched_count'], entry_signal, symbol), 0)}只，"
                f"热度={fmt_number(value(sentiment['hot_score'], entry_signal, symbol))}，"
                f"加速={fmt_number(value(sentiment['count_acceleration'], entry_signal, symbol))}"
            )
        else:
            entry_reason += f"；同主题ROC20正向广度={fmt_percent(value(breadth, entry_signal, symbol), 0)}"

        audit_rows.append(
            {
                "trade_no": int(number + 1),
                "symbol": symbol,
                "name": names[symbol]["name"],
                "category": names[symbol]["category"],
                "entry_signal_date": entry_signal,
                "entry_start_date": trade["entry_date"],
                "entry_completed_date": trade["entry_completed_date"],
                "exit_signal_date": exit_signal,
                "exit_start_date": trade["exit_date_x"],
                "exit_completed_date": trade["exit_completed_date"],
                "entry_type": entry_type,
                "entry_reason": entry_reason,
                "exit_reason": exit_reason,
                "buy_fill_count": int(trade["buy_fill_count"]),
                "sell_fill_count": int(trade["sell_fill_count"]),
                "quantity": float(trade["quantity"]),
                "average_buy_price": average_buy_price,
                "average_sell_price": float(trade["average_sell_price"]),
                "buy_cost": float(trade["buy_cost"]),
                "sell_proceeds": float(trade["sell_proceeds"]),
                "total_fees": float(trade["total_fees"]),
                "net_pnl": float(trade["net_pnl"]),
                "net_return": float(trade["net_return"]),
                "cumulative_realized_pnl": cumulative_pnl,
                "holding_days": int(trade["holding_days"]),
                "outcome": str(trade["outcome"]),
                "holding_mfe": mfe,
                "holding_mae": mae,
                "peak_date": peak_date,
                "trough_date": trough_date,
                "entry_rank": value(strategy_rank, entry_signal, symbol),
                "entry_roc20": value(base.roc_short, entry_signal, symbol),
                "entry_roc60": value(base.roc_medium, entry_signal, symbol),
                "entry_ma120_bias": value(base.ma_bias, entry_signal, symbol),
                "entry_r2_20": value(r2, entry_signal, symbol),
                "entry_efficiency20": value(efficiency, entry_signal, symbol),
                "entry_category_breadth": value(breadth, entry_signal, symbol),
                "entry_sentiment_available": sentiment_available,
                "entry_matched_hot_stocks": value(sentiment["matched_count"], entry_signal, symbol),
                "entry_hot_score": value(sentiment["hot_score"], entry_signal, symbol),
                "entry_count_acceleration": value(sentiment["count_acceleration"], entry_signal, symbol),
                "entry_positive_dde_share": value(sentiment["positive_dde_share"], entry_signal, symbol),
                "entry_gate_pass": value(gate, entry_signal, symbol),
                "entry_eligible_count": int(diag_entry["entry_eligible_count"]),
                "exit_rank": value(strategy_rank, exit_signal, symbol),
                "exit_roc20": value(base.roc_short, exit_signal, symbol),
                "exit_roc60": value(base.roc_medium, exit_signal, symbol),
                "exit_ma120_bias": value(base.ma_bias, exit_signal, symbol),
                "exit_sentiment_available": bool(available.at[exit_signal]),
                "exit_matched_hot_stocks": value(sentiment["matched_count"], exit_signal, symbol),
                "exit_hot_score": value(sentiment["hot_score"], exit_signal, symbol),
                "exit_hot_protected": value(hot_exit_protect, exit_signal, symbol),
                "exit_missing_data_strong_trend": value(missing_strong, exit_signal, symbol),
                "entry_forward_max_return_20d": clean(trade.get("entry_forward_max_return_20d")),
                "entry_forward_return_20d": clean(trade.get("entry_forward_return_20d")),
                "post_exit_max_return_10d": clean(trade.get("post_exit_max_return_10d")),
                "failure_type": failure_type,
                "failed_operation": clean(failed),
                "entry_delay_days": clean(timing_row.get("entry_delay_days")) if timing_row is not None else None,
                "exit_delay_days": clean(timing_row.get("exit_delay_days")) if timing_row is not None else None,
                "peak_giveback": clean(timing_row.get("peak_giveback")) if timing_row is not None else None,
                "move_capture_ratio": clean(timing_row.get("move_capture_ratio")) if timing_row is not None else None,
            }
        )

    fill_rows: list[dict] = []
    for number, fill in fills.iterrows():
        symbol = str(fill["symbol"])
        side = str(fill["side"])
        slippage_cost = abs(float(fill["price"]) - float(fill["raw_open"])) * float(fill["qty"])
        fill_rows.append(
            {
                "fill_no": int(number + 1),
                "date": pd.Timestamp(fill["date"]),
                "symbol": symbol,
                "name": names[symbol]["name"],
                "category": names[symbol]["category"],
                "side": "买入" if side == "BUY" else "卖出",
                "quantity": float(fill["qty"]),
                "raw_open": float(fill["raw_open"]),
                "execution_price": float(fill["price"]),
                "gross_amount": float(fill["qty"]) * float(fill["price"]),
                "fee": float(fill["fee"]),
                "slippage_rate": float(fill["slippage_rate"]),
                "estimated_slippage_cost": slippage_cost,
            }
        )

    equity_rows: list[dict] = []
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    daily_return = equity.pct_change(fill_method=None)
    for date, amount in equity.items():
        signal_row = weights.loc[date] if date in weights.index else pd.Series(dtype=float)
        active = signal_row[signal_row > 0.0]
        target = str(active.index[0]) if not active.empty else "现金"
        equity_rows.append(
            {
                "date": date,
                "equity": float(amount),
                "cumulative_return": float(amount / float(market["project"]["initial_capital"]) - 1.0),
                "drawdown": float(drawdown.at[date]),
                "daily_return": clean(daily_return.at[date]),
                "close_target": target,
            }
        )

    monthly = equity.resample("ME").last()
    monthly_rows = [
        {
            "date": date,
            "equity": float(amount),
            "cumulative_return": float(amount / float(market["project"]["initial_capital"]) - 1.0),
            "drawdown": float((equity / equity.cummax() - 1.0).loc[:date].iloc[-1]),
        }
        for date, amount in monthly.items()
    ]

    periods = {
        "2018—2020": ("2018-07-02", "2020-12-31"),
        "2021—2022": ("2021-01-01", "2022-12-31"),
        "2023—2024": ("2023-01-01", "2024-12-31"),
        "2025—2026": ("2025-01-01", "2026-07-17"),
    }
    period_rows = []
    for label, (start, end) in periods.items():
        metrics = period_metrics(
            equity, start, end, float(market["project"]["initial_capital"])
        )
        closed = rounds[
            rounds["exit_completed_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        ]
        period_rows.append(
            {
                "period": label,
                **metrics,
                "completed_trades": int(len(closed)),
                "winning_trades": int((closed["net_pnl"] > 0).sum()),
                "losing_trades": int((closed["net_pnl"] <= 0).sum()),
                "net_pnl": float(closed["net_pnl"].sum()),
            }
        )

    summary = json.loads((BEST / "summary.json").read_text(encoding="utf-8"))
    payload = {
        "meta": {
            "strategy": "ye 策略",
            "data_through": summary["generated_through"],
            "initial_capital": float(market["project"]["initial_capital"]),
            "fixed_cost": ye["execution"]["cost_rule"],
            "execution": "收盘后形成信号，下一交易日开盘执行",
            "pool_size": len(symbols),
            "live_use_allowed": True,
            "round_trip_count": len(audit_rows),
            "fill_count": len(fill_rows),
        },
        "summary": summary,
        "periods": period_rows,
        "annual": annual_rows(equity, rounds, float(market["project"]["initial_capital"])),
        "round_trips": audit_rows,
        "fills": fill_rows,
        "equity": equity_rows,
        "monthly_equity": monthly_rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(clean(payload), ensure_ascii=False), encoding="utf-8")
    print(f"wrote={OUTPUT}")
    print(f"round_trips={len(audit_rows)} fills={len(fill_rows)} equity_rows={len(equity_rows)}")


if __name__ == "__main__":
    main()
