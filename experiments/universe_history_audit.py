from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.execution import entry_eligibility


OUTPUT = ROOT / "results" / "research" / "universe_audit"


def main() -> None:
    market = yaml.safe_load((ROOT / "config" / "market.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "config" / "ye_strategy.yaml").read_text(encoding="utf-8"))
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    names = {symbol_key(item): item["name"] for item in market["universe"]}
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    eligibility, listed_sessions, trailing_amount = entry_eligibility(
        panel, symbols, config["rules"]
    )
    data_end = pd.Timestamp(market["project"]["data_end"])
    rows = []
    for symbol in symbols:
        valid = panel["close"][symbol].dropna()
        eligible_dates = eligibility.index[eligibility[symbol]]
        first = pd.Timestamp(valid.index[0])
        last = pd.Timestamp(valid.index[-1])
        internal = panel["close"].loc[first:last, symbol]
        rows.append({
            "symbol": symbol,
            "name": names[symbol],
            "category": categories[symbol],
            "first_price_date": str(first.date()),
            "last_price_date": str(last.date()),
            "valid_sessions": int(valid.size),
            "internal_missing_sessions": int(internal.isna().sum()),
            "first_entry_eligible_date": (
                str(pd.Timestamp(eligible_dates[0]).date()) if len(eligible_dates) else None
            ),
            "listed_sessions_at_data_end": int(listed_sessions.at[data_end, symbol]),
            "trailing_amount_at_data_end": (
                float(trailing_amount.at[data_end, symbol])
                if pd.notna(trailing_amount.at[data_end, symbol]) else None
            ),
            "ends_before_data_end": bool(last < data_end),
        })

    audit = pd.DataFrame(rows).sort_values(["first_price_date", "symbol"])
    calendar = panel["close"].index
    active_counts = []
    for year in range(calendar[0].year, data_end.year + 1):
        date = calendar[calendar <= pd.Timestamp(f"{year}-12-31")][-1]
        active_counts.append({
            "year": year,
            "date": str(date.date()),
            "priced_symbols": int(panel["close"].loc[date, symbols].notna().sum()),
            "entry_eligible_symbols": int(eligibility.loc[date, symbols].sum()),
        })

    output = {
        "status": "completed_with_known_limitation",
        "as_of": str(data_end.date()),
        "configured_pool_size": len(symbols),
        "point_in_time_listing_gate": True,
        "listing_warmup_sessions": int(config["rules"]["listing_warmup_days"]),
        "point_in_time_liquidity_gate": True,
        "symbols_ending_before_data_end": int(audit["ends_before_data_end"].sum()),
        "symbols_with_internal_price_gaps": int(audit["internal_missing_sessions"].gt(0).sum()),
        "active_counts": active_counts,
        "finding": (
            "现有45只池内没有价格序列提前终止的产品；上市满120个交易日和20日成交额门槛均按历史日期执行。"
        ),
        "known_limitation": (
            "本仓库只有当前配置的45只产品，无法观察历史上已清盘、合并且未进入当前池的ETF；"
            "因此只能确认点时上市/流动性门槛有效，不能宣称已消除当前存续池选择偏差。"
        ),
        "daily_execution_impact": "none",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT / "symbols.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# ye ETF池历史审计",
        "",
        f"- 审计日期：{output['as_of']}",
        f"- 当前固定池：{output['configured_pool_size']}只",
        f"- 价格序列提前终止：{output['symbols_ending_before_data_end']}只",
        f"- 存在内部价格缺口：{output['symbols_with_internal_price_gaps']}只",
        f"- 结论：{output['finding']}",
        f"- 已知限制：{output['known_limitation']}",
        "",
        "本报告只做一次性研究审计，不接入每日执行。",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
