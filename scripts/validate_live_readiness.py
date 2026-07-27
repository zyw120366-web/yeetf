from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.data import load_panel
from etf_rotation.sentiment_ai import review_protocol_fingerprints

AUDIT = ROOT / "results" / "ye_strategy" / "trade_audit.json"


def previous_execution_is_reconciled(date: str) -> bool:
    plans = sorted((ROOT / "results" / "live").glob("*_order_plan.json"))
    previous = [path for path in plans if path.name[:10] < date]
    if not previous:
        return True
    plan_path = previous[-1]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    requires_confirmation = any(item["side"] in {"buy", "sell"} for item in plan["actions"])
    if not requires_confirmation:
        return True
    reconciliation = ROOT / "results" / "audit" / f"{plan_path.name[:10]}_execution_reconciliation.json"
    if not reconciliation.exists():
        return False
    return json.loads(reconciliation.read_text(encoding="utf-8")).get("status") == "confirmed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-07-17")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config" / "ye_strategy.yaml").read_text(encoding="utf-8"))
    market = yaml.safe_load((ROOT / "config" / "market.yaml").read_text(encoding="utf-8"))
    governance = yaml.safe_load((ROOT / "config" / "strategy_governance.yaml").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    account_path = ROOT / "results" / "live" / "account_state.json"
    plan_path = ROOT / "results" / "live" / f"{args.date}_order_plan.json"
    account = json.loads(account_path.read_text(encoding="utf-8")) if account_path.exists() else {}
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    confirmed_positions = [
        item for item in account.get("positions", []) if float(item.get("quantity", 0.0)) > 0
    ]
    confirmed_symbol = str(confirmed_positions[0]["symbol"]) if len(confirmed_positions) == 1 else None
    metrics = audit["summary"]["metrics"]
    architecture = config["enhanced_selection"]["universe_architecture"]
    universe_symbols = {
        f"{item['code']}.{item['market']}" for item in market["universe"]
    }
    challenger_symbols = set(architecture["challenger_symbols"])
    core_symbols = universe_symbols - challenger_symbols
    final_equity = float(audit["equity"][-1]["equity"])
    cash_management_net = float(metrics.get("cash_interest_income", 0.0)) - float(
        metrics.get("cash_management_fees", 0.0)
    )
    # Completed round trips intentionally omit the currently open position.
    # Rebuild terminal equity from every recorded fill plus the final close so
    # a legitimate end-of-period holding is not mistaken for an audit break.
    initial_capital = float(market["project"]["initial_capital"])
    terminal_cash = initial_capital
    terminal_shares: dict[str, float] = {}
    for fill in audit["fills"]:
        symbol = str(fill["symbol"])
        quantity = float(fill["quantity"])
        gross_amount = float(fill["gross_amount"])
        fee = float(fill["fee"])
        if fill["side"] == "买入":
            terminal_cash -= gross_amount + fee
            terminal_shares[symbol] = terminal_shares.get(symbol, 0.0) + quantity
        else:
            terminal_cash += gross_amount - fee
            terminal_shares[symbol] = terminal_shares.get(symbol, 0.0) - quantity
    panel = load_panel(market, ROOT / "market_data" / "prices")
    terminal_date = panel["close"].index[-1]
    terminal_mark = sum(
        quantity * float(panel["close"].at[terminal_date, symbol])
        for symbol, quantity in terminal_shares.items()
        if abs(quantity) > 1e-9
    )
    terminal_reconstructed_equity = terminal_cash + terminal_mark + cash_management_net
    realized_round_trip_pnl = sum(float(row["net_pnl"]) for row in audit["round_trips"])
    open_position_pnl = terminal_reconstructed_equity - initial_capital - realized_round_trip_pnl - cash_management_net
    checks = {
        "single_strategy_name": config["name"] == "ye 策略" and config["role"] == "唯一正式策略",
        "pool_architecture_reconciles": (
            architecture["mode"] == "core_champion_cash_gap"
            and len(universe_symbols) == len(market["universe"])
            and len(universe_symbols) == config["enhanced_selection"]["fixed_pool_size"] == 51
            and len(core_symbols) == architecture["core_pool_size"] == 45
            and len(challenger_symbols) == 6
            and challenger_symbols <= universe_symbols
        ),
        "ordinary_cost_reconciles": math.isclose(
            market["execution"]["fixed_default"]["commission_rate"] + market["execution"]["fixed_default"]["slippage_rate"],
            config["execution"]["ordinary_etf_one_way_cost"], abs_tol=1e-12,
        ),
        "premium_cost_reconciles": math.isclose(
            market["execution"]["fixed_premium_sensitive"]["commission_rate"] + market["execution"]["fixed_premium_sensitive"]["slippage_rate"],
            config["execution"]["qdii_or_premium_sensitive_one_way_cost"], abs_tol=1e-12,
        ),
        "equity_return_reconciles": math.isclose(final_equity, 100000 * (1 + metrics["total_return"]), abs_tol=1e-6),
        "round_trip_pnl_reconciles": math.isclose(
            realized_round_trip_pnl + open_position_pnl + cash_management_net,
            final_equity - initial_capital,
            abs_tol=1e-6,
        ) and math.isclose(
            terminal_reconstructed_equity,
            final_equity,
            abs_tol=1e-6,
        ),
        "fees_reconcile": math.isclose(sum(row["fee"] for row in audit["fills"]), metrics["total_fees"], abs_tol=1e-6),
        "slippage_reconciles": math.isclose(sum(row["estimated_slippage_cost"] for row in audit["fills"]), metrics["slippage_cost_estimate"], abs_tol=1e-6),
        "html_exists": (ROOT / "dashboard" / "public" / "ye-strategy.html").exists(),
        "governance_frozen": (
            governance["formal_strategy"]["name"] == "ye 策略"
            and governance["formal_strategy"]["status"] == "frozen_for_live"
            and governance["live_audit"]["require_run_card"] is True
        ),
        "previous_execution_reconciled": previous_execution_is_reconciled(args.date),
        "live_account_state_confirmed": (
            account.get("confirmation_status") == "confirmed"
            and str(account.get("as_of", "")).startswith(args.date)
            and len(confirmed_positions) <= 1
            and not account.get("pending_orders")
            and float(account.get("total_equity", 0.0)) > 0
            and float(account.get("available_cash", -1.0)) >= 0
        ),
        "live_plan_uses_account_truth": (
            bool(plan)
            and plan.get("current_symbol") == confirmed_symbol
            and plan.get("account_state", {}).get("confirmation_status") == "confirmed"
            and float(plan.get("account_state", {}).get("total_equity", -1.0))
            == float(account.get("total_equity", -2.0))
        ),
    }
    review_path = ROOT / "market_data" / "sentiment" / "ai_review" / f"{args.date}.json"
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        checks["ai_review_complete"] = (
            review.get("status") == "complete" and review.get("coverage") == 1.0
            and review.get("input_count") == review.get("reviewed_count")
        )
        protocol_required = args.date >= "2026-07-23"
        protocol_fingerprinted = (
            review.get("review_protocol") == review_protocol_fingerprints()
            and isinstance(review.get("review_metadata"), dict)
            and review["review_metadata"].get("reviewed_in_current_conversation") is True
        )
        checks["ai_review_protocol_requirement_satisfied"] = (
            not protocol_required or protocol_fingerprinted
        )
    else:
        checks["ai_review_complete"] = False
        protocol_required = args.date >= "2026-07-23"
        protocol_fingerprinted = False
        checks["ai_review_protocol_requirement_satisfied"] = False
    core = [
        key for key in checks
        if key not in {"ai_review_complete", "ai_review_protocol_requirement_satisfied"}
    ]
    report = {
        "status": "READY" if all(checks.values()) else "BLOCKED",
        "core_backtest_and_site": "PASS" if all(checks[key] for key in core) else "FAIL",
        "checks": checks,
        "blocking_items": [key for key, passed in checks.items() if not passed],
        "review_protocol": {
            "required_for_signal_date": protocol_required,
            "fingerprinted": protocol_fingerprinted,
            "status": (
                "fingerprinted" if protocol_fingerprinted
                else "legacy_exempted" if not protocol_required
                else "missing"
            ),
        },
        "reconciliation": {
            "realized_round_trip_pnl": realized_round_trip_pnl,
            "open_position_pnl": open_position_pnl,
            "cash_management_net": cash_management_net,
            "terminal_reconstructed_equity": terminal_reconstructed_equity,
        },
    }
    output = ROOT / "results" / "live" / "readiness_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "READY" else 2)


if __name__ == "__main__":
    main()
