from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_formal_governance_keeps_rules_frozen() -> None:
    governance = yaml.safe_load((ROOT / "config" / "strategy_governance.yaml").read_text(encoding="utf-8"))
    assert governance["formal_strategy"]["id"] == "YE-FORMAL-2026-07-19"
    assert governance["formal_strategy"]["status"] == "frozen_for_live"
    assert governance["live_audit"]["plan_is_not_fill"] is True


def test_live_run_card_is_reproducible_for_current_snapshot() -> None:
    card_path = ROOT / "results" / "audit" / "2026-07-17_live_run_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["strategy"]["id"] == "YE-FORMAL-2026-07-19"
    assert card["sentiment_review"]["coverage"] == 1.0
    assert card["release"]["readiness"] == "READY"
    assert len(card["audit"]["run_manifest_sha256"]) == 64


def test_daily_entry_builds_run_card_and_keeps_research_outside_execution() -> None:
    entry = (ROOT / "scripts" / "run_after_close.py").read_text(encoding="utf-8")
    assert 'run("scripts/build_live_run_card.py", "--date", args.date)' in entry
    assert "research_" not in entry
