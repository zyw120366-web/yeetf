import copy
import json

import pandas as pd
import pytest

from etf_rotation.funding import affordable_quantity, proceeds_buy_quantity
from etf_rotation.live import validate_fills, validate_authorized_plan, atomic_json
from etf_rotation.rotation_execution import publish_proceeds_rotation
from etf_rotation.backtest import run_backtest
from test_execution import panel, project


def plan():
    return {"signal_date": "2026-10-27", "actions": [{"side": "sell", "symbol": "old"}, {"side": "buy", "symbol": "new"}],
            "account_state": {"positions": [{"symbol": "old", "quantity": 6500}]},
            "execution": {"funding_scope": "sale_proceeds_only", "orders": [
                {"side": "sell", "symbol": "old", "confirmed_quantity": 6500},
                {"side": "buy", "symbol": "new", "cost_parameters": {
                    "commission_rate": .0003, "minimum_commission": 5, "lot_size": 100}}]}}


def fills():
    return {"signal_date": "2026-10-27", "broker_available_after_sale": 14067.5, "fills": [
        {"side": "sell", "symbol": "old", "status": "filled", "quantity": 6500, "price": 2.165, "fee": 5},
        {"side": "buy", "symbol": "new", "status": "filled", "quantity": 16000, "price": .876, "fee": 5}]}


def test_net_proceeds_budget_and_rounding():
    f = fills()
    assert proceeds_buy_quantity(plan(), f["fills"][0], .876, 999999) == 16000
    assert proceeds_buy_quantity(plan(), f["fills"][0], .9, 999999) == 15600
    assert proceeds_buy_quantity(plan(), f["fills"][0], .876, 1000) == 1100
    validate_fills(f, plan(), "2026-10-27")
    assert affordable_quantity(1005, 1, .0003, 5) == 1000  # don't double-reserve minimum fee


@pytest.mark.parametrize("fault", ["partial", "unfilled", "quantity", "fee", "cash", "overspend", "buyfee"])
def test_buy_requires_actual_full_sale_and_cash(fault):
    f = fills()
    if fault in {"partial", "unfilled"}:
        f["fills"][0].update(status=fault, quantity=100 if fault == "partial" else 0)
    elif fault == "quantity":
        f["fills"][0]["quantity"] = 6400
    elif fault == "fee":
        f["fills"][0]["fee"] = None
    elif fault == "cash":
        f["broker_available_after_sale"] = 0
    elif fault == "overspend":
        f["fills"][1]["quantity"] = 16100
    else:
        f["fills"][1]["fee"] = None
    with pytest.raises((ValueError, TypeError)):
        validate_fills(f, plan(), "2026-10-27")


def test_unfilled_both_are_not_fake_trades():
    f = fills()
    for row in f["fills"]:
        row.update(status="unfilled", quantity=0, price=0, fee=None)
    validate_fills(f, plan(), "2026-10-27")


def test_proceeds_plan_never_auto_assumed(tmp_path):
    with pytest.raises(ValueError, match="禁止假定执行"):
        validate_authorized_plan(tmp_path, plan(), {}, "2026-10-27")


def test_real_fills_update_shares_but_not_unknown_legacy_cash(tmp_path, monkeypatch):
    from scripts import reconcile_actual_fills as module
    p = plan()
    p["account_state"] = {"confirmation_status": "pending", "as_of": "2026-09-11_close",
        "cash_reconciliation": {"status": "pending"}, "available_cash": None, "total_equity": None,
        "positions": [{"symbol": "old", "quantity": 6500, "average_cost": 2.35, "opened_on": "2026-09-11"}],
        "pending_orders": []}
    atomic_json(tmp_path / "results/live/account_state.json", p["account_state"])
    atomic_json(tmp_path / "results/live/2026-10-27_order_plan.json", p)
    f = fills()
    f.update(confirmation_status="complete", execution_date="2026-10-28")
    atomic_json(tmp_path / "results/live/2026-10-27_actual_fills.json", f)
    atomic_json(tmp_path / "market_data/live_quotes/2026-10-28.json", {
        "date": "2026-10-28", "adjust": "NONE", "final": True,
        "quotes": {"new": {"open": .876, "close": .88}}})
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["reconcile", "--date", "2026-10-27"])
    for _ in range(2):
        with pytest.raises(SystemExit) as done:
            module.reconcile()
        assert done.value.code == 0
    account = json.loads((tmp_path / "results/live/account_state.json").read_text())
    assert account["positions"][0]["symbol"] == "new"
    assert account["positions"][0]["quantity"] == 16000
    assert account["available_cash"] is None and account["total_equity"] is None
    from scripts.validate_live_readiness import previous_execution_is_reconciled
    assert previous_execution_is_reconciled("2026-10-28", tmp_path)


@pytest.mark.parametrize("blocked_sale", [False, True])
def test_backtest_rotation_sells_then_buys_same_open_without_overlap(blocked_sale):
    days = pd.bdate_range("2026-01-01", periods=5)
    close = pd.DataFrame({"old": [2.] * 5, "new": [1.] * 5}, index=days)
    data = panel(close)
    if blocked_sale:
        data["open"].loc[days[2], "old"] = float("nan")
    weights = pd.DataFrame({"old": [1., 0, 0, 0, 0], "new": [0., 1, 1, 1, 1]}, index=days)
    result = run_backtest("rotation", data, weights, str(days[1].date()), str(days[-1].date()), project())
    sales = result.trades.loc[result.trades.side.eq("SELL")]
    purchases = result.trades.loc[result.trades.side.eq("BUY") & result.trades.symbol.eq("new")]
    assert sales.iloc[0].date == purchases.iloc[0].date == days[3 if blocked_sale else 2]
    assert not (result.actual_weights.gt(0).sum(axis=1) > 1).any()


@pytest.mark.parametrize("fault", [None, "news", "candidate", "no_exit", "pending_order"])
def test_release_checks_complete_formal_decision(tmp_path, monkeypatch, fault):
    from pathlib import Path
    from etf_rotation import rotation_execution, daily_report, run_record
    from scripts import validate_live_readiness
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config").symlink_to(root / "config", target_is_directory=True)
    (tmp_path / "market_data").symlink_to(root / "market_data", target_is_directory=True)
    account = json.loads((root / "results/live/account_state.json").read_text())
    if fault == "pending_order":
        account["pending_orders"] = [{"side": "sell"}]
    atomic_json(tmp_path / "results/live/account_state.json", account)
    analysis = {"status": "complete", "holding_trusted": True, "exit_reasons": ["跌破MA120"],
        "technical_target": "512800.SH", "opportunity_switch": {},
        "ranking": [{"symbol": "512800.SH", "final_entry_pass": True}]}
    if fault == "news":
        analysis["status"] = "partial"
    if fault == "candidate":
        analysis["ranking"][0]["final_entry_pass"] = False
    if fault == "no_exit":
        analysis["exit_reasons"] = []
    monkeypatch.setattr(daily_report, "analyze", lambda *_: analysis)
    monkeypatch.setattr(daily_report, "publish", lambda *_: None)
    monkeypatch.setattr(run_record, "write_run_record", lambda *_: None)
    monkeypatch.setattr(validate_live_readiness, "previous_execution_is_reconciled", lambda *_: True)
    if fault == "pending_order":
        with pytest.raises(ValueError):
            publish_proceeds_rotation(tmp_path, "2026-10-27")
    else:
        assert publish_proceeds_rotation(tmp_path, "2026-10-27") == (fault is None)
        if fault is None:
            p = json.loads((tmp_path / "results/live/2026-10-27_order_plan.json").read_text())
            assert p["execution"]["orders"][1]["buy_estimate"]["estimated_quantity_at_last_close"] == 16000
            assert p["account_state"]["available_cash"] is None
