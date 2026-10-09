"""Capture unadjusted exchange OHLC for live valuation and order estimates."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from pathlib import Path

from etf_rotation.data import all_instruments, symbol_key
from etf_rotation.live import atomic_json, valid_number

ROOT = Path(__file__).resolve().parents[1]


def fetch(date: str) -> dict:
    from easy_tdx import Adjust, Market, Period, UnifiedTdxClient

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if date > now.date().isoformat() or (date == now.date().isoformat() and now.hour < 15):
        raise ValueError("未收盘，不能冻结实盘日线")
    config = yaml.safe_load((ROOT / "config/market.yaml").read_text(encoding="utf-8"))
    client = UnifiedTdxClient(timeout=20)
    quotes = {}
    try:
        for item in all_instruments(config):
            frame = client.get_stock_kline(Market.SH.value if item["market"] == "SH" else Market.SZ.value,
                                           item["code"], Period.DAILY, 0, 200, 2, Adjust.NONE)
            rows = frame.loc[pd.to_datetime(frame["datetime"]).eq(pd.Timestamp(date))]
            if len(rows) != 1:
                raise ValueError(f"{symbol_key(item)} 缺少{date}唯一日线")
            row = {key: float(rows.iloc[0][key]) for key in ("open", "high", "low", "close", "amount", "vol")}
            if not all(valid_number(row[key], positive=True) for key in ("open", "high", "low", "close")):
                raise ValueError(f"无效日线: {symbol_key(item)}")
            quotes[symbol_key(item)] = row
    finally:
        client.close()
    payload = {"date": date, "adjust": "NONE", "final": True,
               "captured_at": now.isoformat(), "source": "easy-tdx / TDX unadjusted daily bars", "quotes": quotes}
    target = ROOT / "market_data/live_quotes" / f"{date}.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("quotes") == quotes and existing.get("adjust") == "NONE":
            return existing
        raise ValueError("已冻结实盘行情与新下载不同；须核对来源后修订，禁止静默覆盖")
    atomic_json(target, payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    result = fetch(args.date)
    print(f"Unadjusted {args.date}: {len(result['quotes'])} instruments")
