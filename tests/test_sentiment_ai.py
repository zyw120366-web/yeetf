from __future__ import annotations

import pytest

from etf_rotation.sentiment_ai import normalize_snapshot, review_snapshot, validate_coverage


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


def test_duplicate_or_missing_reviews_fail_closed() -> None:
    items = normalize_snapshot(sample_snapshot())
    with pytest.raises(ValueError):
        validate_coverage(items, [fake_review(items[:1], {})[0]])
