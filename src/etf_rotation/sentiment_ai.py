from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROMPT_VERSION = "ye-codex-chat-review-2026-07-22"
REVIEW_POLICY_PATH = ROOT / "config" / "sentiment_review_policy.yaml"
SENTIMENT_CONFIG_PATH = ROOT / "config" / "sentiment.yaml"
REVIEW_SCHEMA_PATH = ROOT / "skills" / "ye-daily-execution" / "references" / "review-schema.md"
REQUIRED_REVIEW_METADATA = {
    "model_family",
    "model_snapshot",
    "surface",
    "reviewed_in_current_conversation",
}
REQUIRED_REVIEW_FIELDS = {
    "source_hash", "relevant", "matched_categories", "matched_symbols", "direction",
    "horizon", "confidence", "novelty", "summary", "evidence", "risk_flags",
}
VALID_HORIZONS = {"intraday", "1-3d", "1-4w", "structural", "unknown"}


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_protocol_fingerprints() -> dict:
    """Return immutable fingerprints for the human-readable review contract."""

    records = []
    for path in (REVIEW_POLICY_PATH, SENTIMENT_CONFIG_PATH, REVIEW_SCHEMA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"review protocol input is missing: {path}")
        records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(path),
        })
    return {"version": PROMPT_VERSION, "files": records}


def _fold_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def canonical_event_key(item: dict) -> str:
    """Build a stable same-day economic-event key across data vendors.

    Daily limit/hot lists commonly repeat the same company in several sources.
    Every source row remains reviewable, while downstream strength counts use
    one event per company and date.
    """

    date = str(item.get("published_at") or item.get("date") or "")[:10]
    name = _fold_text(item.get("name"))
    if name:
        identity = {"date": date, "company": name}
    else:
        identity = {
            "date": date,
            "text": _fold_text(f"{item.get('title', '')} {item.get('body', '')}"),
        }
    return canonical_hash(identity)


def effective_symbol_mapping(
    item: dict,
    review: dict,
    symbol_keywords: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Keep only symbol mappings directly supported by frozen row text.

    Category labels remain useful context, but they may not fan one generic
    category (for example 商品) out to every specialised ETF in that category.
    """

    text = " ".join(str(item.get(key) or "") for key in ("name", "title", "body"))
    folded = text.casefold()
    effective: list[str] = []
    rejected: list[str] = []
    for symbol in dict.fromkeys(str(value) for value in review.get("matched_symbols", [])):
        keywords = symbol_keywords.get(symbol, [])
        if keywords and any(str(keyword).casefold() in folded for keyword in keywords):
            effective.append(symbol)
        else:
            rejected.append(symbol)
    return effective, rejected


def deduplicate_reviewed_rows(rows: Iterable[dict]) -> list[dict]:
    """Collapse cross-source repeats conservatively for feature aggregation."""

    selected: dict[str, dict] = {}
    for row in rows:
        key = row.get("normalization", {}).get("event_key") or canonical_event_key(row)
        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue
        ai = row.get("ai", {})
        current_ai = current.get("ai", {})
        score = (
            abs(int(ai.get("direction", 0))),
            float(ai.get("confidence", 0.0)),
            -int(ai.get("direction", 0)),
        )
        current_score = (
            abs(int(current_ai.get("direction", 0))),
            float(current_ai.get("confidence", 0.0)),
            -int(current_ai.get("direction", 0)),
        )
        if score > current_score:
            selected[key] = row
    return list(selected.values())


def _number(row: dict, *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def normalize_snapshot(snapshot: dict) -> list[dict]:
    """Normalize every successfully collected source row; never keyword pre-filter."""
    normalized: list[dict] = []
    for source, block in (snapshot.get("sources") or {}).items():
        if not block.get("ok") or block.get("rows") is None:
            continue
        for row in block["rows"]:
            name = str(row.get("name") or row.get("n") or "").strip()
            reason = str(row.get("reason") or "").strip()
            title = str(row.get("title") or row.get("brief") or "").strip()
            content = str(row.get("content") or row.get("description") or "").strip()
            industry = str(row.get("hybk") or row.get("industry") or "").strip()
            if not title:
                title = " | ".join(part for part in (name, reason or industry) if part)
            body_parts = [part for part in (reason, content, industry) if part]
            payload = {
                "source": source,
                "source_id": str(row.get("id") or row.get("c") or row.get("code") or ""),
                "published_at": row.get("ctime") or row.get("time") or row.get("date"),
                "name": name,
                "title": title[:600],
                "body": " | ".join(body_parts)[:2400],
                "return_pct": _number(row, "zdp", "zhangfu"),
                "turnover": _number(row, "amount", "chengjiaoe"),
                "dde_net": _number(row, "ddejingliang"),
            }
            payload["source_hash"] = canonical_hash({"source": source, "row": row})
            normalized.append(payload)
    return list({item["source_hash"]: item for item in normalized}.values())


def validate_coverage(inputs: Iterable[dict], reviews: Iterable[dict]) -> None:
    expected = [item["source_hash"] for item in inputs]
    actual = [item.get("source_hash") for item in reviews]
    if len(actual) != len(set(actual)):
        raise ValueError("review contains duplicate source_hash values")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"review coverage mismatch: missing={len(missing)}, extra={len(extra)}")


def validate_review_items(reviews: Iterable[dict]) -> None:
    for index, item in enumerate(reviews, start=1):
        missing = REQUIRED_REVIEW_FIELDS - set(item)
        extra = set(item) - REQUIRED_REVIEW_FIELDS
        if missing or extra:
            raise ValueError(f"review item {index} schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        if not isinstance(item["relevant"], bool):
            raise ValueError(f"review item {index}: relevant must be boolean")
        if not isinstance(item["direction"], int) or not -2 <= item["direction"] <= 2:
            raise ValueError(f"review item {index}: direction must be an integer from -2 to 2")
        if item["horizon"] not in VALID_HORIZONS:
            raise ValueError(f"review item {index}: invalid horizon")
        for key in ("confidence", "novelty"):
            if not isinstance(item[key], (int, float)) or not 0 <= item[key] <= 1:
                raise ValueError(f"review item {index}: {key} must be from 0 to 1")
        for key in ("matched_categories", "matched_symbols", "evidence", "risk_flags"):
            if not isinstance(item[key], list) or not all(isinstance(value, str) for value in item[key]):
                raise ValueError(f"review item {index}: {key} must be a string array")
        if not isinstance(item["summary"], str):
            raise ValueError(f"review item {index}: summary must be a string")


def validate_review_metadata(metadata: dict) -> None:
    missing = REQUIRED_REVIEW_METADATA - set(metadata)
    if missing:
        raise ValueError(f"review metadata is incomplete: missing={sorted(missing)}")
    if metadata["reviewed_in_current_conversation"] is not True:
        raise ValueError("review metadata must confirm current-conversation review")


def review_snapshot(
    snapshot: dict,
    context: dict,
    review_batch: Callable[[list[dict], dict], list[dict]],
    *,
    model: str = "codex_chat",
    batch_size: int = 40,
    review_metadata: dict | None = None,
) -> dict:
    """Build a complete audit payload from a reviewer supplied by the current chat."""
    inputs = normalize_snapshot(snapshot)
    reviews: list[dict] = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        batch_reviews = review_batch(batch, context)
        validate_coverage(batch, batch_reviews)
        validate_review_items(batch_reviews)
        reviews.extend(batch_reviews)
    validate_coverage(inputs, reviews)
    by_hash = {item["source_hash"]: item for item in reviews}
    sentiment_config = yaml.safe_load(SENTIMENT_CONFIG_PATH.read_text(encoding="utf-8"))
    symbol_keywords = {
        str(symbol): list(keywords)
        for symbol, keywords in sentiment_config["symbol_keywords"].items()
    }
    first_by_event: dict[str, str] = {}
    reviewed_items = []
    for item in inputs:
        review = by_hash[item["source_hash"]]
        # Some vendor rows omit their own timestamp even though the frozen
        # snapshot has an unambiguous trading date. Use that date only for the
        # cross-source event key so same-company rows still deduplicate while
        # the immutable source row and source_hash remain untouched.
        event_key = canonical_event_key({
            **item,
            "date": item.get("date") or snapshot.get("date"),
        })
        duplicate_of = first_by_event.get(event_key)
        first_by_event.setdefault(event_key, item["source_hash"])
        effective, rejected = effective_symbol_mapping(item, review, symbol_keywords)
        reviewed_items.append({
            **item,
            "normalization": {
                "event_key": event_key,
                "duplicate_of": duplicate_of,
                "effective_matched_symbols": effective,
                "rejected_matched_symbols": rejected,
            },
            "ai": review,
        })
    metadata = review_metadata or {
        "model_family": model,
        "model_snapshot": "not_recorded",
        "surface": "unknown",
        "reviewed_in_current_conversation": True,
    }
    return {
        "date": snapshot.get("date"),
        "status": "complete",
        "policy": "every_collected_row_reviewed_in_codex_chat_fail_closed",
        "prompt_version": PROMPT_VERSION,
        "reviewer": model,
        "review_metadata": metadata,
        "review_protocol": review_protocol_fingerprints(),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_hash": canonical_hash(snapshot),
        "input_count": len(inputs),
        "reviewed_count": len(reviews),
        "relevant_count": sum(bool(item["ai"]["relevant"]) for item in reviewed_items),
        "coverage": 1.0,
        "items": reviewed_items,
    }


def write_manual_review(snapshot_path: Path, reviews_path: Path, output_dir: Path) -> Path:
    """Commit a Codex-chat review only when it covers the immutable snapshot exactly."""
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    failed_sources = [name for name, block in snapshot.get("sources", {}).items() if not block.get("ok")]
    if failed_sources:
        raise RuntimeError(f"source collection incomplete; fail closed: {failed_sources}")
    target = output_dir / f"{snapshot['date']}.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("snapshot_hash") != canonical_hash(snapshot):
            raise RuntimeError(f"immutable review exists for a different snapshot: {target}")
        return target
    raw_reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    reviews = raw_reviews["items"] if isinstance(raw_reviews, dict) else raw_reviews
    if not isinstance(reviews, list):
        raise ValueError("manual review input must be a list or an object containing items")
    if not isinstance(raw_reviews, dict):
        raise ValueError("manual review input must include review_metadata")
    metadata = raw_reviews.get("review_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("manual review input must include review_metadata")
    validate_review_metadata(metadata)
    payload = review_snapshot(
        snapshot,
        {},
        lambda _items, _context: reviews,
        batch_size=max(1, len(reviews)),
        review_metadata=metadata,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
