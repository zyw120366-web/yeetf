"""Fixed joint-input contrasts; isolated research, never a live execution path."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from experiments.rank_exit_review import ROOT, STARTS, PERIODS, digest, signals
from experiments import price_basis_audit_20260911 as price
from experiments.structural_audit_20260911 import specific_frame, remove_dde, observed_breadth, held
from etf_rotation import ye
from etf_rotation.data import load_panel, universe_keys, symbol_key
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.sentiment import sentiment_matrices_from_frame

OUT = ROOT / "results/research/joint_realism_20260914"
END = "2026-09-10"
COMPONENTS = ("weak_edge_filter", "emerging_trend", "quality_extension", "hot_exit_protection")
VARIANTS = ("rich", "no_dde", "specific", "joint", "joint_valid", *["without_" + x for x in COMPONENTS], "without_all")


def load_price_inputs(market):
    frozen = load_panel(market, ROOT / "market_data/prices")
    calendar = frozen["close"].index
    raws, events, action_map = {}, {}, {}
    for symbol in universe_keys(market):
        raw = pd.read_csv(price.OUT / f"{symbol}_raw.csv", parse_dates=["datetime"]).set_index("datetime").sort_index()
        raw = raw.loc[:END]
        actions = pd.read_csv(price.OUT / f"{symbol}_actions.csv")
        event, unsupported = price.events_from(actions, raw)
        assert not unsupported, (symbol, unsupported)
        raws[symbol], events[symbol] = raw, event
        for day, values in event.items():
            action_map.setdefault(day, {})[symbol] = values
    raw_panel = {key: pd.DataFrame({s: f[key].reindex(calendar) for s, f in raws.items()})
                 for key in ("open", "high", "low", "close", "amount", "vol")}
    economic = dict(raw_panel)
    economic["close"] = pd.DataFrame({s: price.economic_close(f, events[s]).reindex(calendar) for s, f in raws.items()})
    return raw_panel, economic, action_map


def spec(variant):
    narrow = variant not in ("rich", "no_dde")
    no_dde = variant not in ("rich", "specific")
    valid = variant == "joint_valid" or variant.startswith("without_")
    components = None
    if variant == "without_all":
        components = {c: False for c in COMPONENTS}
    elif variant.startswith("without_"):
        components = {variant.removeprefix("without_"): False}
    return narrow, no_dde, valid, components


def checks_for(metrics, phases, variant):
    def row(v, start=STARTS[0], cost=1, capital=100000):
        return metrics.query("variant == @v and start == @start and cost == @cost and capital == @capital").iloc[0]
    base, alt = row("joint_valid"), row(variant)
    phase = phases.pivot(index="period", columns="variant", values="total_return")
    checks = {
        "return_and_sharpe": bool(alt.total_return > base.total_return and alt.sharpe > base.sharpe),
        "drawdown_within_2pp": bool(alt.max_drawdown >= base.max_drawdown - .02),
        "starts_4_of_5": bool(sum(row(variant, s).total_return > row("joint_valid", s).total_return + 1e-10 for s in STARTS) >= 4),
        "periods_3_of_4": int(phase[variant].gt(phase.joint_valid + 1e-10).sum()) >= 3,
        "double_cost": bool(row(variant, cost=2).total_return > row("joint_valid", cost=2).total_return),
        "small_capital": bool(row(variant, capital=15275).total_return > row("joint_valid", capital=15275).total_return),
    }
    return {"checks": checks, "passed": all(checks.values()),
            "starts_won": int(sum(row(variant, s).total_return > row("joint_valid", s).total_return + 1e-10 for s in STARTS)),
            "periods_won": int(phase[variant].gt(phase.joint_valid + 1e-10).sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    market = yaml.safe_load((ROOT / "config/market.yaml").read_text())
    config = yaml.safe_load((ROOT / "config/ye_strategy.yaml").read_text())
    assert str(market["project"]["data_end"]) == END, "frozen research endpoint changed"
    folders = ("config", "src", "scripts", "market_data", "results/live", "results/ye_strategy", "outputs", "dashboard")
    protected = sorted(p for folder in folders for p in (ROOT / folder).rglob("*")
                       if p.is_file() and "__pycache__" not in str(p) and "node_modules" not in str(p) and ".git" not in p.parts)
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    symbols = universe_keys(market)
    categories = {symbol_key(x): x["category"] for x in market["universe"]}
    raw_panel, economic, action_map = load_price_inputs(market)
    calendar = economic["close"].index
    original = pd.read_csv(ROOT / "market_data/sentiment/features/symbol_daily.csv")
    narrow = specific_frame()
    original.to_csv(OUT / "original_features.csv", index=False)
    narrow.to_csv(OUT / "specific_features.csv", index=False)
    cash = config["cash_management"]
    cm = {"annual_rate": cash["historical_backtest_annual_rate"], **{k: cash[k] for k in ("fee_rate", "minimum_order", "order_lot")}}
    premium = [s for s in symbols if s.startswith("513") or s == "159941.SZ"]
    # Adaptation source is emitted only into this new research directory.
    with patch.object(price, "OUT", OUT):
        raw_run = price.raw_execution_function()
    rows, phases, differences, gates, curves, full_positions = [], [], [], [], {}, {}
    for variant in VARIANTS:
        is_narrow, no_dde, valid, components = spec(variant)
        frame = narrow if is_narrow else original
        if no_dde:
            frame = remove_dde(frame)
        sentiment, available = sentiment_matrices_from_frame(frame, calendar, symbols)
        breadth_fn = observed_breadth if valid else ye.anchor_category_breadth
        with patch.object(ye, "anchor_category_breadth", breadth_fn):
            bundle, _, eligibility, _, _, context = ye.build_ye_signals(
                economic, symbols, categories, config, sentiment, available, components=components)
        for start in STARTS:
            weights = signals(economic, symbols, config, context, eligibility, start, False)
            if start == STARTS[0]:
                assert np.allclose(weights.loc[start:], bundle.weights.loc[start:])
            scenarios = [(1, 100000)] + ([(2, 100000), (1, 15275)] if start == STARTS[0] else [])
            for cost, capital in scenarios:
                project = execution_project(market, premium, eligibility.shift(1, fill_value=False))
                project.update(initial_capital=capital, corporate_actions=action_map)
                for costs in [project, *project["symbol_costs"].values()]:
                    for key in ("commission_rate", "slippage_rate", "minimum_commission"):
                        costs[key] *= cost
                result = raw_run(variant, raw_panel, weights, start, END, project, cash_management=cm)
                rows.append({"variant": variant, "start": start, "cost": cost, "capital": capital, **result.metrics})
                if start == STARTS[0] and cost == 1 and capital == 100000:
                    curves[variant] = result.equity
                    full_positions[variant] = held(result.actual_weights)
                    result.trades.to_csv(OUT / f"{variant}_fills.csv", index=False)
                    for label, first, last in PERIODS:
                        phases.append({"variant": variant, "period": label, **period_metrics(result.equity, first, last, capital)})
                    if variant == "rich":
                        prior = pd.read_csv(price.OUT / "metrics.csv")
                        old = prior.query("variant == 'economic_signal_raw_execution' and start == @start and cost == 1 and capital == 100000").iloc[0]
                        assert abs(old.total_return - result.metrics["total_return"]) < 1e-9, "prior price result not reproduced"
            print(f"{variant} {start}: complete", flush=True)
        for key in ("current_normal", "emerging", "quality_extension", "hot_exit_protection", "entry_gate"):
            mask = (context[key] & eligibility).loc[STARTS[0]:END]
            gates.append({"variant": variant, "gate": key, "symbol_days": int(mask.sum().sum()), "days": int(mask.any(axis=1).sum())})
        pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    positions = pd.DataFrame(full_positions)
    for variant in VARIANTS:
        baseline = "joint_valid" if variant.startswith("without_") else "rich"
        for day in positions.index[positions[variant].ne(positions[baseline])]:
            differences.append({"variant": variant, "date": day, "baseline": baseline,
                                "base_position": positions.at[day, baseline], "position": positions.at[day, variant]})
    pd.DataFrame(curves).to_csv(OUT / "equity.csv")
    positions.to_csv(OUT / "positions.csv")
    pd.DataFrame(phases).to_csv(OUT / "periods.csv", index=False)
    pd.DataFrame(differences).to_csv(OUT / "position_differences.csv", index=False)
    pd.DataFrame(gates).to_csv(OUT / "gate_counts.csv", index=False)
    checks = {v: checks_for(pd.DataFrame(rows), pd.DataFrame(phases), v) for v in VARIANTS if v.startswith("without_")}
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in protected}, "production file changed"
    inputs = [Path(__file__), OUT / "PROTOCOL.md", ROOT / "experiments/rank_exit_review.py",
              ROOT / "experiments/structural_audit_20260911.py", ROOT / "experiments/price_basis_audit_20260911.py",
              *sorted(price.OUT.glob("*_raw.csv")), *sorted(price.OUT.glob("*_actions.csv"))]
    audit = {"end": END, "run_count": len(rows), "checks": checks, "formal_unchanged": before,
             "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
             "status": "research_only_not_promoted", "limitations": ["retrospective_not_OOS", "same_vendor_price_actions", "dividends_credited_ex_date", "specific_keywords_not_historical_AI", "no_DDE_is_stress_not_replay", "survivor_pool", "daily_fill_proxy"]}
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
