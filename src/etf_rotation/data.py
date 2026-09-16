from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Iterable

import pandas as pd


FIELDS = ("open", "high", "low", "close", "vol", "amount")
PRICE_FIELDS = ("open", "high", "low", "close")


def symbol_key(item: dict) -> str:
    return f"{item['code']}.{item['market']}"


def all_instruments(config: dict) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in config["universe"] + config["market_proxies"] + [config["benchmark"]]:
        merged[symbol_key(item)] = item
    return list(merged.values())


def merge_frozen_history(cached: pd.DataFrame, downloaded: pd.DataFrame, finalized_through: str | None = None) -> pd.DataFrame:
    """Keep audited bars immutable and append only genuinely new sessions.

    TDX recalculates the full QFQ history after corporate actions.  The latest
    overlap ratio converts new OHLC bars to the already-frozen price scale so a
    routine refresh cannot rewrite prior signals or create an artificial jump.
    """
    cached = cached.sort_values("datetime").drop_duplicates("datetime", keep="last")
    downloaded = downloaded.sort_values("datetime").drop_duplicates("datetime", keep="last")
    if finalized_through is not None:
        cached = cached.loc[pd.to_datetime(cached["datetime"]) <= pd.Timestamp(finalized_through)]
    if cached.empty:
        return downloaded

    last_frozen = pd.to_datetime(cached["datetime"]).max()
    new_rows = downloaded[pd.to_datetime(downloaded["datetime"]) > last_frozen].copy()
    if new_rows.empty:
        return cached

    cached_close = cached.loc[pd.to_datetime(cached["datetime"]) == last_frozen, "close"].iloc[-1]
    overlap = downloaded.loc[pd.to_datetime(downloaded["datetime"]) == last_frozen, "close"]
    if overlap.empty or float(overlap.iloc[-1]) <= 0:
        raise ValueError("新旧复权行情没有有效重叠日，禁止拼接")
    if not overlap.empty and float(overlap.iloc[-1]) != 0.0:
        scale = float(cached_close) / float(overlap.iloc[-1])
        for field in PRICE_FIELDS:
            if field in new_rows:
                new_rows[field] = new_rows[field].astype(float) * scale

    return pd.concat([cached, new_rows], ignore_index=True).sort_values("datetime")


def audited_price_cutoff(root: Path, complete_through: str) -> str:
    """Freeze published evidence, not just days on which trading was allowed.

    Cash reconciliation can block orders while a complete observation report
    still uses that day's bars. Later downloads must not rewrite those bars.
    Unfinished/partial reports do not freeze a new session.
    """
    for path in sorted((root / "results/audit").glob("*_live_run_card.json"), reverse=True):
        date = path.name[:10]
        if date > complete_through:
            continue
        card = json.loads(path.read_text(encoding="utf-8"))
        status = card.get("release", {}).get("readiness")
        if card.get("signal_date") != date or status not in {"READY", "SELL_ONLY", "BLOCKED"}:
            continue
        if status == "BLOCKED" and card.get("observation", {}).get("status") != "complete":
            continue
        manifest_path = root / "results/audit" / f"{date}_run_manifest.json"
        content = manifest_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != card.get("audit", {}).get("run_manifest_sha256"):
            raise ValueError(f"{date} 运行清单绑定失效，停止行情刷新以保护已审计历史")
        manifest = json.loads(content)
        records = manifest.get("price_files", []) + manifest.get("critical_files", [])
        if manifest.get("signal_date") != date or not any(
            r.get("path", "").startswith("market_data/prices/") and r["path"].endswith(".csv")
            for r in records
        ):
            raise ValueError(f"{date} 运行清单缺少行情证据，不能确定冻结边界")
        return date
    return "1900-01-01"


def fetch_easy_tdx(config: dict, data_dir: Path, force: bool = False) -> dict:
    """Download QFQ bars while preserving every previously audited session."""
    from easy_tdx import Adjust, Market, Period, UnifiedTdxClient

    data_dir.mkdir(parents=True, exist_ok=True)
    count = int(config["project"].get("data_count", 800))
    manifest: dict[str, dict] = {}
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    complete_through = now.date() if now.hour >= 15 else now.date() - timedelta(days=1)
    frozen_through = audited_price_cutoff(data_dir.parents[1], complete_through.isoformat())
    client = UnifiedTdxClient(timeout=20)
    try:
        for item in all_instruments(config):
            key = symbol_key(item)
            path = data_dir / f"{key}.csv"
            cached = pd.read_csv(path, parse_dates=["datetime"]) if path.exists() else None
            if path.exists() and not force:
                frame = cached
            else:
                market = Market.SH.value if item["market"] == "SH" else Market.SZ.value
                last_error: Exception | None = None
                for attempt in range(3):
                    try:
                        downloaded = client.get_stock_kline(
                            market,
                            item["code"],
                            Period.DAILY,
                            0,
                            count,
                            2,
                            Adjust.QFQ,
                        )
                        if downloaded.empty:
                            raise RuntimeError(f"empty bars for {key}")
                        downloaded = downloaded.sort_values("datetime").drop_duplicates("datetime")
                        downloaded = downloaded.loc[pd.to_datetime(downloaded["datetime"]) <= pd.Timestamp(complete_through)]
                        frame = (
                            merge_frozen_history(cached, downloaded, frozen_through)
                            if cached is not None
                            else downloaded
                        )
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
