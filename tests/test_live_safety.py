from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from etf_rotation.live import (apply_live_cooldown, atomic_json, fingerprint, raw_quote,
                               validate_account, validate_authorized_plan, validate_fills)
from etf_rotation.data import merge_frozen_history
from etf_rotation.sentiment_ai import review_snapshot, validate_live_review
from scripts import run_after_close, validate_live_readiness
from scripts import advance_authorized_live_account as advance_script

ROOT = Path(__file__).resolve().parents[1]


def authorized_fixture(tmp_path, monkeypatch):
    state = account()
    state["as_of"] = "2026-09-10_close"
    atomic_json(tmp_path / "results/live/account_state.json", state)
    plan = {"signal_date": "2026-09-10", "target_symbol": "159985.SZ",
            "account_state": state, "actions": [
                {"side": "sell", "symbol": "513200.SH"},
                {"side": "buy", "symbol": "159985.SZ"}]}
    plan_path = tmp_path / "results/live/2026-09-10_order_plan.json"
    atomic_json(plan_path, plan)
    manifest_path = tmp_path / "results/audit/2026-09-10_run_manifest.json"
    atomic_json(manifest_path, {"critical_files": [{"path": str(plan_path.relative_to(tmp_path)),
                                                  "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()}]})
    atomic_json(tmp_path / "results/audit/2026-09-10_live_run_card.json", {
        "signal_date": "2026-09-10", "release": {"readiness": "READY"},
        "decision": {"target_symbol": "159985.SZ"},
        "audit": {"run_manifest": str(manifest_path.relative_to(tmp_path)),
                  "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}})
    for name in ("market.yaml", "strategy_governance.yaml"):
        path = tmp_path / "config" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT / "config" / name).read_bytes())
    monkeypatch.setattr(advance_script, "ROOT", tmp_path)
    monkeypatch.setattr(advance_script, "ACCOUNT", tmp_path / "results/live/account_state.json")
    monkeypatch.setattr(advance_script, "trading_dates", lambda: pd.bdate_range("2026-09-10", "2026-09-11"))
    monkeypatch.setattr(advance_script, "quote", lambda symbol, day: (1., 1.) if symbol == "513200.SH" else (2., 2.))
    return state, plan


def test_authorized_account_advance_is_idempotent_and_records_exit(tmp_path, monkeypatch):
    authorized_fixture(tmp_path, monkeypatch)
    advance_script.advance("2026-09-11")
    after = json.loads(advance_script.ACCOUNT.read_text())
    assert after["positions"][0]["symbol"] == "159985.SZ"
    assert after["last_exit_dates"]["513200.SH"] == "2026-09-11"
    receipt = (tmp_path / "results/audit/2026-09-10_execution_reconciliation.json").read_bytes()
    advance_script.advance("2026-09-11")
    assert json.loads(advance_script.ACCOUNT.read_text()) == after
    assert (tmp_path / "results/audit/2026-09-10_execution_reconciliation.json").read_bytes() == receipt


def test_existing_real_reconciliation_is_never_overwritten(tmp_path, monkeypatch):
    state, _ = authorized_fixture(tmp_path, monkeypatch)
    path = tmp_path / "results/audit/2026-09-10_execution_reconciliation.json"
    atomic_json(path, {"status": "confirmed", "actual_fills": []})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="不得用假定记录覆盖"):
        advance_script.advance("2026-09-11")
    assert path.read_bytes() == before
    assert json.loads(advance_script.ACCOUNT.read_text()) == state


def test_changed_released_plan_cannot_execute(tmp_path, monkeypatch):
    state, plan = authorized_fixture(tmp_path, monkeypatch)
    plan["actions"][1]["symbol"] = "159865.SZ"
    atomic_json(tmp_path / "results/live/2026-09-10_order_plan.json", plan)
    with pytest.raises(ValueError, match="放行后发生变化"):
        advance_script.advance("2026-09-11")
    assert json.loads(advance_script.ACCOUNT.read_text()) == state


def test_receipt_recovers_interrupted_account_write_without_second_trade(tmp_path, monkeypatch):
    state, _ = authorized_fixture(tmp_path, monkeypatch)
    advance_script.advance("2026-09-11")
    after = json.loads(advance_script.ACCOUNT.read_text())
    atomic_json(advance_script.ACCOUNT, state)
    monkeypatch.setattr(advance_script, "quote", lambda *_: pytest.fail("不得重复执行交易"))
    advance_script.advance("2026-09-11")
    assert json.loads(advance_script.ACCOUNT.read_text()) == after


def test_failed_news_still_reports_price_exit_conditions():
    from etf_rotation.live import price_risk_check
    assert any("ROC20转负" in x for x in price_risk_check(ROOT, "2026-09-10", account()))


def account():
    # These cases exercise the September 10 plan, not today's mutable account.
    plan = json.loads((ROOT / "results/live/2026-09-10_order_plan.json").read_text())
    return copy.deepcopy(plan["account_state"])


def test_unknown_cash_after_confirmed_trade_is_not_zero_or_executable():
    value = account()
    value.update(confirmation_status="pending", available_cash=None, total_equity=None)
    before = copy.deepcopy(value)
    with pytest.raises(ValueError, match="账户状态未确认"):
        validate_account(value)
    assert value == before


def test_blocked_plan_cannot_advance_even_if_it_contains_orders(tmp_path):
    atomic_json(tmp_path / "results/audit/2026-09-10_live_run_card.json",
                {"signal_date": "2026-09-10", "release": {"readiness": "BLOCKED"}})
    with pytest.raises(ValueError, match="未放行"):
        validate_authorized_plan(tmp_path, {"actions": [{"side": "buy"}]}, account(), "2026-09-10")


@pytest.mark.parametrize("mutation", ["zero", "negative", "nan", "date"])
def test_invalid_filled_records_are_not_confirmed(mutation):
    plan = {"signal_date": "2026-09-10", "actions": [{"side": "buy", "symbol": "159985.SZ"}]}
    fills = {"signal_date": "2026-09-10", "fills": [{"side": "buy", "symbol": "159985.SZ", "status": "filled", "quantity": 4000, "price": 2.3}]}
    if mutation == "date":
        fills["signal_date"] = "2026-09-09"
    else:
        fills["fills"][0]["quantity"] = {"zero": 0, "negative": -1, "nan": float("nan")}[mutation]
    with pytest.raises(ValueError):
        validate_fills(fills, plan, "2026-09-10")


def test_account_equity_must_reconcile_and_quantities_cannot_be_negative():
    value = account()
    value["total_equity"] += 100
    with pytest.raises(ValueError, match="权益"):
        validate_account(value)
    value = account()
    value["positions"][0]["quantity"] = -10
    with pytest.raises(ValueError):
        validate_account(value)


def test_cooldown_blocks_sold_core_and_allows_satellite_when_no_core_available():
    rows = pd.DataFrame([{"symbol": "core", "pool_role": "core", "technical_entry_pass": True},
                         {"symbol": "satellite", "pool_role": "challenger", "technical_entry_pass": True}])
    dates = pd.bdate_range("2026-09-01", "2026-09-15")
    blocked = apply_live_cooldown(rows, {"last_exit_dates": {"core": "2026-09-03"}}, "2026-09-09", dates, 5)
    assert list(blocked.loc[blocked["final_entry_pass"], "symbol"]) == ["satellite"]
    recovered = apply_live_cooldown(rows, {"last_exit_dates": {"core": "2026-09-03"}}, "2026-09-10", dates, 5)
    assert list(recovered.loc[recovered["final_entry_pass"], "symbol"]) == ["core"]


def test_unadjusted_quotes_never_fall_back_to_adjusted_prices(tmp_path):
    path = tmp_path / "market_data/live_quotes/2026-09-10.json"
    atomic_json(path, {"date": "2026-09-10", "adjust": "QFQ", "final": True, "quotes": {"x": {"open": 9, "close": 10}}})
    with pytest.raises(ValueError):
        raw_quote(tmp_path, "x", "2026-09-10")
    atomic_json(path, {"date": "2026-09-10", "adjust": "NONE", "final": True, "quotes": {"x": {"open": 1, "close": 2}}})
    assert raw_quote(tmp_path, "x", "2026-09-10") == (1, 2)


def test_unfinalized_daily_bar_can_be_refreshed_without_rewriting_audited_day():
    cached = pd.DataFrame({"datetime": pd.to_datetime(["2026-09-09", "2026-09-10"]), "close": [1., 1.1]})
    downloaded = cached.copy()
    downloaded.loc[1, "close"] = 1.2
    result = merge_frozen_history(cached, downloaded, "2026-09-09")
    assert list(result["close"]) == [1., 1.2]


def test_empty_review_cannot_claim_100_percent():
    with pytest.raises(ValueError, match="空资讯"):
        review_snapshot({"date": "2026-09-10", "sources": {}}, {}, lambda *_: [])


def test_review_rechecks_snapshot_and_actual_items(tmp_path):
    folder = ROOT / "market_data/sentiment"
    snapshot = json.loads((folder / "2026-09-10.json").read_text())
    review = json.loads((folder / "ai_review/2026-09-10.json").read_text())
    atomic_json(tmp_path / "market_data/sentiment/2026-09-10.json", snapshot)
    atomic_json(tmp_path / "market_data/sentiment/ai_review/2026-09-10.json", review)
    assert validate_live_review(tmp_path, "2026-09-10")["coverage"] == 1
    review["items"].pop()
    atomic_json(tmp_path / "market_data/sentiment/ai_review/2026-09-10.json", review)
    with pytest.raises(ValueError):
        validate_live_review(tmp_path, "2026-09-10")


def test_failed_pipeline_always_publishes_blocked_report_and_card(tmp_path, monkeypatch):
    atomic_json(tmp_path / "results/live/account_state.json", account())
    monkeypatch.setattr(run_after_close, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run", "--date", "2026-09-10", "--skip-collect"])
    def failure(_):
        raise RuntimeError("资讯缺失")
    monkeypatch.setattr(run_after_close, "run_pipeline", failure)
    with pytest.raises(SystemExit):
        run_after_close.main()
    card = json.loads((tmp_path / "results/audit/2026-09-10_live_run_card.json").read_text())
    assert card["release"]["readiness"] == "BLOCKED"
    assert not card["decision"]["actions"]
    assert "暂无可执行买入计划" in (tmp_path / "outputs/ETF轮动策略_今日日报.html").read_text()


def test_blocked_html_does_not_present_a_buy_order(monkeypatch):
    spec = importlib.util.spec_from_file_location("html_safety", ROOT / "dashboard/scripts/build_ye_strategy_html.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = Path.read_text
    def blocked(path, *args, **kwargs):
        if path.name == "readiness_report.json":
            return json.dumps({"status": "BLOCKED", "signal_date": "2026-09-10", "blocking_items": ["test"]})
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", blocked)
    page = module.build_daily_page()
    assert "暂无可执行买入计划" in page
    assert "收盘估算约" not in page
