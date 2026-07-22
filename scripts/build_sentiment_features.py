from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.data import symbol_key
from etf_rotation.sentiment_ai import (
    canonical_event_key,
    deduplicate_reviewed_rows,
    effective_symbol_mapping,
)


def keyword_match(text: str, keywords: list[str]) -> bool:
    folded = text.casefold()
    return any(str(keyword).casefold() in folded for keyword in keywords)


def load_rows() -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, list[dict]]]:
    raw_dir = ROOT / "market_data" / "sentiment" / "ths_hot_reason"
    by_date: dict[pd.Timestamp, list[dict]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("ok") or payload.get("rows") is None:
            continue
        by_date[pd.Timestamp(payload["date"])] = list(payload["rows"])
    # A complete AI review replaces the legacy keyword-only mapping for that
    # date. Incomplete files are ignored, so a partial review can never leak
    # into a live feature set.
    ai_dir = ROOT / "market_data" / "sentiment" / "ai_review"
    for path in sorted(ai_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("coverage") != 1.0:
            continue
        if payload.get("input_count") != payload.get("reviewed_count"):
            continue
        by_date[pd.Timestamp(payload["date"])] = [
            {**item, "_ai_reviewed": True} for item in payload["items"]
        ]
    return pd.DatetimeIndex(sorted(by_date)), by_date


def build() -> pd.DataFrame:
    config = yaml.safe_load((ROOT / "config" / "sentiment.yaml").read_text(encoding="utf-8"))
    market = yaml.safe_load((ROOT / "config" / "market.yaml").read_text(encoding="utf-8"))
    dates, rows_by_date = load_rows()
    symbol_keywords = {str(k): list(v) for k, v in config["symbol_keywords"].items()}
    categories = {symbol_key(item): str(item["category"]) for item in market["universe"]}
    category_keywords = {str(k): list(v) for k, v in config["category_keywords"].items()}

    records: list[dict] = []
    for day in dates:
        source_rows = rows_by_date[day]
        ai_day = bool(source_rows and source_rows[0].get("_ai_reviewed"))
        if ai_day:
            normalized_rows = []
            for row in source_rows:
                effective, rejected = effective_symbol_mapping(
                    row, row["ai"], symbol_keywords
                )
                normalization = {
                    **row.get("normalization", {}),
                    "event_key": row.get("normalization", {}).get("event_key")
                    or canonical_event_key(row),
                    "effective_matched_symbols": effective,
                    "rejected_matched_symbols": rejected,
                }
                normalized_rows.append({**row, "normalization": normalization})
            source_rows = normalized_rows
        positive_market = [
            row for row in source_rows
            if not ai_day or (row["ai"]["relevant"] and int(row["ai"]["direction"]) > 0)
        ]
        if ai_day:
            positive_market = deduplicate_reviewed_rows(positive_market)
        market_count = len(positive_market)
        market_turnover = sum(float(row.get("turnover") or row.get("chengjiaoe") or 0.0) for row in positive_market)
        for symbol, category in categories.items():
            specific = symbol_keywords.get(symbol, [])
            broad = category_keywords.get(category, [])
            keywords = list(dict.fromkeys([*specific, *broad]))
            if ai_day:
                related = [
                    row for row in source_rows
                    if row["ai"]["relevant"]
                    and symbol in row["normalization"]["effective_matched_symbols"]
                ]
                related = deduplicate_reviewed_rows(related)
                matched = [row for row in related if int(row["ai"]["direction"]) > 0]
                negative = [row for row in related if int(row["ai"]["direction"]) < 0]
            else:
                matched = [
                    row for row in source_rows
                    if keywords and keyword_match(f"{row.get('name', '')}+{row.get('reason', '')}", keywords)
                ]
                negative = []
            count = len(matched)
            turnover = sum(float(row.get("turnover") or row.get("chengjiaoe") or 0.0) for row in matched)
            dde_values = [float(row.get("dde_net") or row.get("ddejingliang") or 0.0) for row in matched]
            ai_direction = [float(row["ai"]["direction"]) * float(row["ai"]["confidence"]) for row in related] if ai_day else []
            ai_negative_risk = [abs(float(row["ai"]["direction"])) / 2 * float(row["ai"]["confidence"]) for row in negative]
            records.append({
                "date": day,
                "symbol": symbol,
                "category": category,
                "matched_count": count,
                "matched_turnover": turnover,
                "positive_dde_share": np.mean(np.asarray(dde_values) > 0.0) if dde_values else np.nan,
                "market_hot_count": market_count,
                "market_hot_turnover": market_turnover,
                "matched_count_share": count / market_count if market_count else np.nan,
                "matched_turnover_share": turnover / market_turnover if market_turnover else np.nan,
                "ai_reviewed": ai_day,
                "ai_negative_count": len(negative),
                "ai_direction_score": np.mean(ai_direction) if ai_direction else np.nan,
                "ai_negative_risk": max(ai_negative_risk) if ai_negative_risk else 0.0,
            })
    frame = pd.DataFrame(records).sort_values(["symbol", "date"])
    grouped = frame.groupby("symbol", sort=False)
    frame["prior5_count_mean"] = grouped["matched_count"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=3).mean()
    )
    frame["count_acceleration"] = (
        (frame["matched_count"] + 1.0) / (frame["prior5_count_mean"] + 1.0) - 1.0
    )
    frame["count_history_percentile"] = grouped["matched_count_share"].transform(
        lambda values: values.rolling(60, min_periods=20).rank(pct=True)
    )
    frame["turnover_history_percentile"] = grouped["matched_turnover_share"].transform(
        lambda values: values.rolling(60, min_periods=20).rank(pct=True)
    )
    frame["cross_section_count_percentile"] = frame.groupby("date")["matched_count_share"].rank(pct=True)
    frame["hot_score"] = (
        0.45 * frame["count_history_percentile"].fillna(0.5)
        + 0.25 * frame["turnover_history_percentile"].fillna(0.5)
        + 0.20 * frame["cross_section_count_percentile"].fillna(0.5)
        + 0.10 * frame["positive_dde_share"].fillna(0.5)
    )
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def main() -> None:
    frame = build()
    output = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    summary = {
        "first_date": str(frame["date"].min().date()),
        "last_date": str(frame["date"].max().date()),
        "source_days": int(frame["date"].nunique()),
        "symbols": int(frame["symbol"].nunique()),
        "rows": int(len(frame)),
        "nonzero_symbol_days": int(frame["matched_count"].gt(0).sum()),
    }
    (output.parent / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
