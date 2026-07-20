from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


PROMPT_VERSION = "ye-codex-chat-review-2026-07-18"
REQUIRED_REVIEW_FIELDS = {
    "source_hash", "relevant", "matched_categories", "matched_symbols", "direction",
    "horizon", "confidence", "novelty", "summary", "evidence", "risk_flags",
}
VALID_HORIZONS = {"intraday", "1-3d", "1-4w", "structural", "unknown"}


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def review_snapshot(
    snapshot: dict,
    context: dict,
    review_batch: Callable[[list[dict], dict], list[dict]],
    *,
    model: str = "codex_chat",
    batch_size: int = 40,
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
    reviewed_items = [{**item, "ai": by_hash[item["source_hash"]]} for item in inputs]
    return {
        "date": snapshot.get("date"),
        "status": "complete",
        "policy": "every_collected_row_reviewed_in_codex_chat_fail_closed",
        "prompt_version": PROMPT_VERSION,
        "reviewer": model,
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
    raw_reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    reviews = raw_reviews["items"] if isinstance(raw_reviews, dict) else raw_reviews
    if not isinstance(reviews, list):
        raise ValueError("manual review input must be a list or an object containing items")
    payload = review_snapshot(snapshot, {}, lambda _items, _context: reviews, batch_size=max(1, len(reviews)))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{snapshot['date']}.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("snapshot_hash") != payload["snapshot_hash"]:
            raise RuntimeError(f"immutable review exists for a different snapshot: {target}")
        return target
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
