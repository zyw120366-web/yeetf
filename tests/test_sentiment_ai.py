from __future__ import annotations

import pytest

from etf_rotation.sentiment_ai import (
    canonical_event_key,
    deduplicate_reviewed_rows,
    effective_symbol_mapping,
    normalize_snapshot,
    review_snapshot,
    validate_coverage,
    validate_review_metadata,
)


def sample_snapshot() -> dict:
    return {
        "date": "2026-07-17",
        "sources": {
            "ths_hot_reason": {"ok": True, "rows": [{"id": 1, "name": "甲公司", "reason": "算力订单"}]},
            "eastmoney_limit_down": {"ok": True, "rows": [{"c": "2", "n": "乙公司", "hybk": "医药", "zdp": -10}]},
        },
    }


def fake_review(items: list[dict], _: dict) -> list[dict]:
    return [
        {
            "source_hash": item["source_hash"], "relevant": True,
            "matched_categories": ["科技数字"], "matched_symbols": ["515050.SH"],
            "direction": 1, "horizon": "1-3d", "confidence": 0.8, "novelty": 0.7,
            "summary": "测试", "evidence": [item["title"]], "risk_flags": [],
        }
        for item in items
    ]


def test_every_source_row_is_reviewed() -> None:
    payload = review_snapshot(sample_snapshot(), {}, fake_review, model="test", batch_size=1)
    assert payload["status"] == "complete"
    assert payload["input_count"] == payload["reviewed_count"] == 2
    assert payload["coverage"] == 1.0
    assert payload["review_protocol"]["version"] == "ye-codex-chat-review-2026-07-22"
    assert all(len(record["sha256"]) == 64 for record in payload["review_protocol"]["files"])


def test_duplicate_or_missing_reviews_fail_closed() -> None:
    items = normalize_snapshot(sample_snapshot())
    with pytest.raises(ValueError):
        validate_coverage(items, [fake_review(items[:1], {})[0]])


def test_same_company_cross_source_rows_keep_audit_but_count_once() -> None:
    rows = [
        {
            "published_at": "2026-07-22", "name": "甲公司", "source": "vendor_a",
            "ai": {"direction": 1, "confidence": 0.68},
        },
        {
            "published_at": "2026-07-22", "name": "甲公司", "source": "vendor_b",
            "ai": {"direction": 1, "confidence": 0.78},
        },
    ]
    assert canonical_event_key(rows[0]) == canonical_event_key(rows[1])
    deduplicated = deduplicate_reviewed_rows(rows)
    assert len(deduplicated) == 1
    assert deduplicated[0]["source"] == "vendor_b"


def test_snapshot_date_deduplicates_vendor_row_without_timestamp() -> None:
    snapshot = {
        "date": "2026-07-27",
        "sources": {
            "vendor_a": {
                "ok": True,
                "rows": [{"id": 1, "date": "2026-07-27", "name": "甲公司", "reason": "算力"}],
            },
            "vendor_b": {
                "ok": True,
                "rows": [{"id": 2, "name": "甲公司", "industry": "通信"}],
            },
        },
    }
    payload = review_snapshot(snapshot, {}, fake_review, model="test")
    first, second = payload["items"]
    assert first["normalization"]["event_key"] == second["normalization"]["event_key"]
    assert second["normalization"]["duplicate_of"] == first["source_hash"]


def test_generic_commodity_text_cannot_map_to_soybean_etf() -> None:
    review = {"matched_symbols": ["518880.SH", "159985.SZ"]}
    item = {"name": "招金黄金", "title": "黄金+半年报预增", "body": "贵金属"}
    keywords = {
        "518880.SH": ["黄金", "金矿", "贵金属"],
        "159985.SZ": ["豆粕", "大豆", "养殖", "饲料"],
    }
    effective, rejected = effective_symbol_mapping(item, review, keywords)
    assert effective == ["518880.SH"]
    assert rejected == ["159985.SZ"]


def test_review_metadata_is_required_and_truthful() -> None:
    with pytest.raises(ValueError):
        validate_review_metadata({"model_family": "GPT-5"})
    with pytest.raises(ValueError):
        validate_review_metadata({
            "model_family": "GPT-5",
            "model_snapshot": "not_exposed_by_codex",
            "surface": "Codex desktop",
            "reviewed_in_current_conversation": False,
        })
