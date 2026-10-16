from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from etf_rotation import risk_release
from etf_rotation.data import load_panel
from etf_rotation.live import atomic_json, validate_authorized_plan, publish_blocked
from scripts import build_sentiment_features, run_after_close, validate_live_readiness
from experiments.strategy_ablation import component_effects
from experiments import rank_exit_review
from test_live_safety import authorized_fixture
from scripts import advance_authorized_live_account as advance_script

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-10"


@pytest.fixture
def risk_env(tmp_path, monkeypatch):
    for name in ("config", "scripts", "src"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    (tmp_path / "market_data/prices").mkdir(parents=True)
    for path in (ROOT / "market_data/prices").glob("*.csv"):
        target = tmp_path / "market_data/prices" / path.name
        if path.name in {"510300.SH.csv", "513200.SH.csv"}:
            shutil.copyfile(path, target)
        else:
            target.symlink_to(path)
    for name in ("live_quotes", "sentiment"):
        (tmp_path / "market_data" / name).symlink_to(ROOT / "market_data" / name, target_is_directory=True)
    # A fixed historical fixture, not the user's changing real account.
    state = json.loads((ROOT / "results/live/2026-09-10_order_plan.json").read_text())["account_state"]
    atomic_json(tmp_path / "results/live/account_state.json", state)
    market = yaml.safe_load((ROOT / "config/market.yaml").read_text())
    panel = load_panel(market, ROOT / "market_data/prices")
    monkeypatch.setattr(risk_release, "load_panel", lambda *_: panel)
    monkeypatch.setattr(validate_live_readiness, "previous_execution_is_reconciled", lambda *_: True)
    return tmp_path, panel, state


def hard_exit(panel):
    panel["close"].loc[pd.Timestamp(DATE), "513200.SH"] = .1


def test_hard_exit_does_not_need_news_but_is_sell_only(risk_env, monkeypatch):
    root, panel, state = risk_env
    hard_exit(panel)
    monkeypatch.setattr(risk_release, "validate_live_review", lambda *_: pytest.fail("硬退出不读取AI"))
    plan = risk_release.assess_sell_only(root, DATE)
    assert plan["target_symbol"] is None
    assert plan["review_dates_required"] == []
    assert plan["actions"][0]["side"] == "sell"
    assert plan["execution"]["orders"][0]["confirmed_quantity"] == state["positions"][0]["quantity"]


def test_soft_exit_missing_review_remains_blocked(risk_env, monkeypatch):
    root, _, _ = risk_env
    def missing(*_):
        raise ValueError("审核缺失")
    monkeypatch.setattr(risk_release, "validate_live_review", missing)
    with pytest.raises(ValueError, match="审核缺失"):
        risk_release.assess_sell_only(root, DATE)
    assert not risk_release.publish_sell_only(root, DATE, "missing review")


def test_soft_exit_cannot_bypass_hot_protection(risk_env, monkeypatch):
    root, panel, _ = risk_env
    dates = panel["close"].index[-2:]
    monkeypatch.setattr(build_sentiment_features, "build", lambda **_: pd.DataFrame([
        {"date": day, "symbol": "513200.SH", "hot_score": 1, "matched_count": 4, "positive_dde_share": 1}
        for day in dates]))
    with pytest.raises(ValueError, match="热点软退出保护"):
        risk_release.assess_sell_only(root, DATE)


@pytest.mark.parametrize("failure", ["unreconciled", "pending", "stale_calendar", "missing_bar", "bad_mark"])
def test_sell_only_rejects_untrusted_inputs(risk_env, monkeypatch, failure):
    root, panel, state = risk_env
    hard_exit(panel)
    if failure == "unreconciled":
        monkeypatch.setattr(validate_live_readiness, "previous_execution_is_reconciled", lambda *_: False)
    elif failure == "pending":
        state["pending_orders"] = [{"side": "sell"}]
    elif failure in {"stale_calendar", "missing_bar"}:
        key = "510300.SH" if failure == "stale_calendar" else "513200.SH"
        path = root / "market_data/prices" / f"{key}.csv"
        frame = pd.read_csv(path)
        frame.loc[~frame["datetime"].eq(DATE)].to_csv(path, index=False)
    else:
        state["positions"][0]["market_price"] += .1
        state["positions"][0]["market_value"] += 910
        state["total_equity"] += 910
    atomic_json(root / "results/live/account_state.json", state)
    with pytest.raises(ValueError):
        risk_release.assess_sell_only(root, DATE)


def test_sell_only_publication_has_authenticated_plan_and_no_actual_fills(risk_env):
    root, panel, state = risk_env
    hard_exit(panel)
    publish_blocked(root, DATE, "source failed")
    assert risk_release.publish_sell_only(root, DATE, "source failed")
    plan = json.loads((root / f"results/live/{DATE}_order_plan.json").read_text())
    validate_authorized_plan(root, plan, state, DATE)
    assert "仅卖出" in (root / "outputs/ETF轮动策略_今日日报.html").read_text()
    assert not (root / f"results/live/{DATE}_actual_fills.json").exists()
    assert json.loads((root / "results/live/account_state.json").read_text()) == state
    manifest = json.loads((root / f"results/audit/{DATE}_run_manifest.json").read_text())
    for item in manifest["critical_files"]:
        assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    plan["actions"].append({"side": "buy", "symbol": "159985.SZ"})
    with pytest.raises(ValueError, match="仅卖出放行"):
        validate_authorized_plan(root, plan, state, DATE)


def test_sell_only_authorized_advance_liquidates_without_new_position(tmp_path, monkeypatch):
    _, plan = authorized_fixture(tmp_path, monkeypatch)
    plan["target_symbol"] = None
    plan["actions"] = plan["actions"][:1]
    plan_path = tmp_path / f"results/live/{DATE}_order_plan.json"
    atomic_json(plan_path, plan)
    manifest_path = tmp_path / f"results/audit/{DATE}_run_manifest.json"
    atomic_json(manifest_path, {"critical_files": [{"path": str(plan_path.relative_to(tmp_path)),
                 "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()}]})
    atomic_json(tmp_path / f"results/audit/{DATE}_live_run_card.json", {
        "signal_date": DATE, "release": {"readiness": "SELL_ONLY"},
        "decision": {"target_symbol": None, "actions": plan["actions"]},
        "audit": {"run_manifest": str(manifest_path.relative_to(tmp_path)),
                  "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}})
    advance_script.advance("2026-09-11")
    after = json.loads(advance_script.ACCOUNT.read_text())
    assert after["positions"] == []
    assert after["available_cash"] > 9000
    assert after["last_exit_dates"]["513200.SH"] == "2026-09-11"
    advance_script.advance("2026-09-11")
    assert json.loads(advance_script.ACCOUNT.read_text()) == after


def test_failed_sell_publication_rolls_back_to_blocked(tmp_path, monkeypatch):
    atomic_json(tmp_path / "results/live/account_state.json", json.loads((ROOT / "results/live/account_state.json").read_text()))
    monkeypatch.setattr(run_after_close, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["run", "--date", DATE])
    def fail(*_):
        raise RuntimeError("write failure")
    monkeypatch.setattr(run_after_close, "execute", fail)
    monkeypatch.setattr(risk_release, "publish_sell_only", fail)
    with pytest.raises(SystemExit):
        run_after_close.main()
    assert json.loads((tmp_path / "results/live/readiness_report.json").read_text())["status"] == "BLOCKED"


def test_provenance_distinguishes_three_actual_computation_paths():
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2026-07-16", "2026-07-17"])
    assert list(build_sentiment_features.source_regimes(dates)["regime"]) == [
        "price_fallback", "keyword_proxy", "keyword_proxy", "ai_review"]
    _, rows = build_sentiment_features.load_rows(use_ai_reviews=False)
    assert not any(row.get("_ai_reviewed") for day in rows.values() for row in day)


def test_component_interpretation_uses_actual_sign():
    rows = [{"label": "A", "return_since_2024": 2, "max_drawdown_since_2024": -.2},
            {"label": "B", "return_since_2024": 1, "max_drawdown_since_2024": -.3},
            {"label": "C", "return_since_2024": 3, "max_drawdown_since_2024": -.1}]
    first, second = component_effects(rows)
    assert "-100.00个百分点" in first and "-10.00个百分点" in first
    assert "+200.00个百分点" in second and "+20.00个百分点" in second


def test_candidate_changes_only_dual_rank_exit_and_starts_in_cash(monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=65)
    frame = pd.DataFrame(1., index=dates, columns=["x"])
    context = {"entry_gate": frame.astype(bool), "dual_rank_decline": frame.astype(bool),
               "entry_rank": frame.copy(), "entry_score": frame, "soft_exit_confirmation": frame.astype(bool),
               "core_symbols": ["x"]}
    context["entry_rank"].iloc[-1] = 6
    captured = []
    def fake(*_, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(weights=frame), None
    monkeypatch.setattr(rank_exit_review, "etfwin_signals", fake)
    config = yaml.safe_load((ROOT / "config/ye_strategy.yaml").read_text())
    for candidate in (False, True):
        rank_exit_review.signals({"close": frame}, ["x"], config, context, frame.astype(bool), str(dates[10].date()), candidate)
    assert not captured[0]["entry_gate"].iloc[:10].any().any()
    assert captured[0]["dual_rank_decline_override"].all().all()
    assert captured[1]["dual_rank_decline_override"].sum().sum() == 1
    for key in ("entry_gate", "soft_exit_confirmation", "entry_ranking_score_override"):
        pd.testing.assert_frame_equal(captured[0][key], captured[1][key])
