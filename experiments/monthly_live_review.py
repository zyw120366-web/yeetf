from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ABLATION_EQUITY = ROOT / "results" / "research" / "strategy_ablation" / "equity.csv"
OUTPUT = ROOT / "results" / "research" / "monthly_review"


def period_return(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    series = series.dropna().sort_index()
    window = series[(series.index >= start) & (series.index <= end)]
    if window.empty:
        return None
    previous = series[series.index < window.index[0]]
    base = float(previous.iloc[-1]) if not previous.empty else float(window.iloc[0])
    return float(window.iloc[-1] / base - 1.0) if base else None


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a research-only monthly ye live review")
    parser.add_argument("--month", required=True, help="YYYY-MM")
    args = parser.parse_args()
    month = pd.Period(args.month, freq="M")
    start = month.start_time.normalize()
    end = month.end_time.normalize()

    cards = []
    for path in sorted((ROOT / "results" / "audit").glob(f"{args.month}-??_live_run_card.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        account = payload.get("account_state", {})
        if account.get("confirmation_status") != "confirmed":
            continue
        cards.append((path, payload))
    if not cards:
        raise RuntimeError(f"no confirmed live run cards found for {args.month}")

    latest_path, latest = cards[-1]
    live_start_value = latest.get("account_state", {}).get("performance", {}).get(
        "strategy_start_date"
    )
    if live_start_value:
        live_start = pd.Timestamp(live_start_value)
        cards = [
            (path, payload)
            for path, payload in cards
            if pd.Timestamp(payload["signal_date"]) >= live_start
        ]
        if not cards:
            raise RuntimeError(f"no confirmed live run cards on or after {live_start.date()}")
        latest_path, latest = cards[-1]
    account = latest["account_state"]
    contributed = float(account.get("performance", {}).get("net_contributed_capital") or 0.0)
    equity = float(account.get("total_equity") or 0.0)
    live_return = equity / contributed - 1.0 if contributed > 0 else None
    daily_equity = pd.Series(
        {
            pd.Timestamp(payload["signal_date"]): float(payload["account_state"]["total_equity"])
            for _, payload in cards
        }
    ).sort_index()
    live_drawdown = float((daily_equity / daily_equity.cummax() - 1.0).min())

    if not ABLATION_EQUITY.exists():
        raise RuntimeError("run experiments/strategy_ablation.py before the monthly review")
    shadow = pd.read_csv(ABLATION_EQUITY, parse_dates=["date"]).set_index("date")
    full_return = period_return(shadow["full_strategy"], start, end)
    price_return = period_return(shadow["price_core"], start, end)

    payload = {
        "status": "research_only",
        "month": args.month,
        "daily_execution_impact": "none",
        "confirmed_run_days": len(cards),
        "latest_run_card": latest_path.relative_to(ROOT).as_posix(),
        "strategy_ids_seen": sorted({p["strategy"]["id"] for _, p in cards}),
        "live": {
            "net_contributed_capital": contributed,
            "ending_equity": equity,
            "return_since_live_start": live_return,
            "observed_month_drawdown": live_drawdown,
            "ending_positions": account.get("positions", []),
        },
        "shadow_same_month": {
            "full_strategy_return": full_return,
            "price_core_return": price_return,
            "full_minus_price_core": (
                full_return - price_return
                if full_return is not None and price_return is not None
                else None
            ),
        },
        "discipline": {
            "ai_review_complete_days": sum(
                payload.get("sentiment_review", {}).get("coverage") == 1.0
                for _, payload in cards
            ),
            "ready_days": sum(
                payload.get("release", {}).get("readiness") == "READY"
                for _, payload in cards
            ),
            "reconciliation_statuses": {
                status: sum(
                    payload.get("execution_reconciliation", {}).get("status") == status
                    for _, payload in cards
                )
                for status in sorted({
                    payload.get("execution_reconciliation", {}).get("status", "missing")
                    for _, payload in cards
                })
            },
        },
        "interpretation": "月度报告只核对实盘、纪律和冻结基线差异，不据此自动调参。",
    }
    payload = clean(payload)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT / f"{args.month}.json"
    md_path = OUTPUT / f"{args.month}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(value: float | None) -> str:
        return "无数据" if value is None else f"{value:+.2%}"

    md = f"""# ye 策略月度复盘：{args.month}

本报告是独立研究产物，不参与每日信号，不自动修改参数。

## 实盘

- 已确认运行日：{payload['confirmed_run_days']}日
- 净投入资金：{contributed:,.2f}元
- 月末权益：{equity:,.2f}元
- 实盘开启以来收益：{pct(live_return)}
- 已观察实盘回撤：{pct(live_drawdown)}

## 同期冻结影子对照

- 当前完整策略：{pct(full_return)}
- 纯价格核心：{pct(price_return)}
- 完整策略相对纯价格核心：{pct(payload['shadow_same_month']['full_minus_price_core'])}

## 纪律

- AI审核完整：{payload['discipline']['ai_review_complete_days']}/{payload['confirmed_run_days']}日
- READY：{payload['discipline']['ready_days']}/{payload['confirmed_run_days']}日
- 成交对账状态：{json.dumps(payload['discipline']['reconciliation_statuses'], ensure_ascii=False)}

只在月度频率观察，不根据单月盈亏修改策略。
"""
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
