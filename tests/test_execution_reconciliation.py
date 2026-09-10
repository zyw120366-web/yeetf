from __future__ import annotations

import pytest

from scripts.reconcile_actual_fills import accept_confirmed_account_baseline
from scripts.validate_live_readiness import RECONCILED_EXECUTION_STATUSES


def exception_report() -> dict:
    return {
        "signal_date": "2026-09-09",
        "status": "exception",
        "confirmation_status": "complete",
        "matched_plan": True,
        "all_filled": False,
        "actual_fills": [
            {"side": "sell", "symbol": "513200.SH", "status": "cancelled", "quantity": 0},
            {"side": "buy", "symbol": "159865.SZ", "status": "cancelled", "quantity": 0},
        ],
    }


def confirmed_account() -> dict:
    return {
        "as_of": "2026-09-10_close",
        "confirmation_status": "confirmed",
        "total_equity": 9418.2,
        "available_cash": 17.9,
        "positions": [{"symbol": "513200.SH", "quantity": 9100}],
        "pending_orders": [],
    }


def test_confirmed_account_can_become_new_baseline_after_cancelled_plan() -> None:
    report = accept_confirmed_account_baseline(
        exception_report(), confirmed_account(), "2026-09-10", "results/live/account_state.json"
    )

    assert report["status"] == "baseline_confirmed"
    assert report["actual_fills"][0]["status"] == "cancelled"
    assert report["baseline_acceptance"]["account_symbol"] == "513200.SH"
    assert "baseline_confirmed" in RECONCILED_EXECUTION_STATUSES


@pytest.mark.parametrize("status,quantity", [("filled", 9100), ("cancelled", 100)])
def test_baseline_reset_rejects_any_executed_quantity(status: str, quantity: int) -> None:
    report = exception_report()
    report["actual_fills"][0].update(status=status, quantity=quantity)

    with pytest.raises(ValueError):
        accept_confirmed_account_baseline(
            report, confirmed_account(), "2026-09-10", "results/live/account_state.json"
        )


def test_baseline_reset_requires_same_day_confirmed_account_without_pending_orders() -> None:
    account = confirmed_account()
    account["pending_orders"] = [{"side": "sell", "symbol": "513200.SH"}]

    with pytest.raises(ValueError):
        accept_confirmed_account_baseline(
            exception_report(), account, "2026-09-10", "results/live/account_state.json"
        )
