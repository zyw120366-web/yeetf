from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


FIELDS = ("open", "high", "low", "close", "vol", "amount")


def symbol_key(item: dict) -> str:
    return f"{item['code']}.{item['market']}"


def all_instruments(config: dict) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in config["universe"] + config["market_proxies"] + [config["benchmark"]]:
        merged[symbol_key(item)] = item
    return list(merged.values())


def fetch_easy_tdx(config: dict, data_dir: Path, force: bool = False) -> dict:
    """Download QFQ daily bars through easy-tdx and cache one CSV per ETF."""
    from easy_tdx import Adjust, Market, Period, UnifiedTdxClient

    data_dir.mkdir(parents=True, exist_ok=True)
    count = int(config["project"].get("data_count", 800))
    manifest: dict[str, dict] = {}
    client = UnifiedTdxClient(timeout=20)
    try:
        for item in all_instruments(config):
            key = symbol_key(item)
            path = data_dir / f"{key}.csv"
            if path.exists() and not force:
                frame = pd.read_csv(path, parse_dates=["datetime"])
            else:
                market = Market.SH.value if item["market"] == "SH" else Market.SZ.value
                last_error: Exception | None = None
                for attempt in range(3):
                    try:
                        frame = client.get_stock_kline(
                            market,
                            item["code"],
                            Period.DAILY,
                            0,
                            count,
                            2,
                            Adjust.QFQ,
                        )
                        if frame.empty:
                            raise RuntimeError(f"empty bars for {key}")
                        frame = frame.sort_values("datetime").drop_duplicates("datetime")
                        frame.to_csv(path, index=False, encoding="utf-8-sig")
                        break
                    except Exception as exc:  # network endpoints can fail transiently
                        last_error = exc
                        time.sleep(1.5 * (attempt + 1))
                        client.close()
                        client = UnifiedTdxClient(timeout=25)
                else:
                    raise RuntimeError(f"failed to download {key}: {last_error}")
            manifest[key] = {
                "name": item.get("name", key),
                "rows": int(len(frame)),
                "start": str(pd.to_datetime(frame["datetime"]).min().date()),
                "end": str(pd.to_datetime(frame["datetime"]).max().date()),
                "provider": "easy-tdx 1.20.4 / TDX QFQ daily bars",
            }
    finally:
        client.close()

    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_panel(config: dict, data_dir: Path) -> dict[str, pd.DataFrame]:
    instruments = all_instruments(config)
    raw: dict[str, pd.DataFrame] = {}
    for item in instruments:
        key = symbol_key(item)
        frame = pd.read_csv(data_dir / f"{key}.csv", parse_dates=["datetime"])
        frame = frame.set_index("datetime").sort_index()
        raw[key] = frame

    bench_key = symbol_key(config["benchmark"])
    calendar = raw[bench_key].index
    start = pd.Timestamp(config["project"]["warmup_start"])
    end = pd.Timestamp(
        config["project"].get("data_end", config["project"].get("test_end"))
    )
    calendar = calendar[(calendar >= start) & (calendar <= end)]

    panel: dict[str, pd.DataFrame] = {}
    for field in FIELDS:
        panel[field] = pd.DataFrame(
            {key: frame[field].reindex(calendar) for key, frame in raw.items()}, index=calendar
        ).astype(float)
    return panel


def universe_keys(config: dict) -> list[str]:
    return [symbol_key(item) for item in config["universe"]]


def proxy_keys(config: dict) -> list[str]:
    return [symbol_key(item) for item in config["market_proxies"]]
