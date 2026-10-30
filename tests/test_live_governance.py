from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_formal_governance_keeps_rules_frozen() -> None:
    governance = yaml.safe_load((ROOT / "config" / "strategy_governance.yaml").read_text(encoding="utf-8"))
    assert governance["formal_strategy"]["id"] == "YE-FORMAL-2026-07-29C"
    assert governance["formal_strategy"]["status"] == "frozen_for_live"
    assert governance["formal_strategy"]["effective_for_signal_dates_from"] == "2026-07-29"
    assert governance["live_audit"]["plan_is_not_fill"] is True


def test_live_run_card_is_reproducible_for_current_snapshot() -> None:
    card_path = ROOT / "results" / "audit" / "2026-07-17_live_run_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["strategy"]["id"] == "YE-FORMAL-2026-07-19"
    assert card["sentiment_review"]["coverage"] == 1.0
    assert card["release"]["readiness"] == "READY"
    assert len(card["audit"]["run_manifest_sha256"]) == 64


def test_research_review_is_monthly_and_outside_daily_entry() -> None:
    governance = yaml.safe_load((ROOT / "config" / "strategy_governance.yaml").read_text(encoding="utf-8"))
    assert governance["research_review"]["cadence"] == "monthly_only"
    assert governance["research_review"]["daily_entry_integration"] == "forbidden"


def test_daily_entry_builds_run_card_and_keeps_research_outside_execution() -> None:
    entry = (ROOT / "scripts" / "run_after_close.py").read_text(encoding="utf-8")
    assert 'run("scripts/build_live_run_card.py", "--date", args.date)' in entry
    assert "research_" not in entry
    assert "experiments/" not in entry


def test_latest_run_uses_current_governance_and_documents_legacy_review() -> None:
    card = json.loads(
        (ROOT / "results" / "audit" / "2026-07-22_live_run_card.json").read_text(
            encoding="utf-8"
        )
    )
    assert card["strategy"]["id"] == "YE-FORMAL-2026-07-22"
    assert card["sentiment_review"]["protocol_status"] == "legacy_pre_policy"
