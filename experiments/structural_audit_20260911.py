"""Frozen research contrasts; no production/account/report writes."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from experiments.rank_exit_review import (
    ROOT, STARTS, PERIODS, digest, signals, remove_best_round,
)
from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_panel, universe_keys, symbol_key
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.evaluation import realized_round_trips
from etf_rotation.sentiment import sentiment_matrices_from_frame
from etf_rotation import ye
from scripts.build_sentiment_features import build, load_rows, source_regimes

OUT = ROOT / "results/research/structural_audit_20260911"
PROTOCOL = ROOT / "docs/2026-09-11_策略结构与数据可迁移性研究计划.md"
END = "2026-09-10"
VARIANTS = ("baseline", "observed_breadth", "no_dde",
            "specific_mapping", "live_schema", "simple_price")


def observed_breadth(roc20, categories, eligibility, core_symbols, challenger_symbols):
    """Unknown history is excluded, not negative; only frozen core peers vote."""
    out = pd.DataFrame(np.nan, index=roc20.index, columns=roc20.columns)
    for symbol in roc20:
        members = [s for s in core_symbols if categories[s] == categories[symbol]]
        if members:
            values = roc20[members]
            denominator = values.notna().sum(axis=1).replace(0, np.nan)
            out[symbol] = values.gt(0).sum(axis=1) / denominator
    return out


def remove_dde(frame):
    frame = frame.copy()
    before = frame["positive_dde_share"].fillna(.5)
    frame["positive_dde_share"] = np.where(frame.matched_count.gt(0), 0., np.nan)
    frame["hot_score"] += .1 * (frame["positive_dde_share"].fillna(.5) - before)
    return frame


def specific_frame():
    # build() is read-only; only generated research configuration lives in temp.
    with tempfile.TemporaryDirectory(prefix="ye-specific-") as tmp:
        root = Path(tmp)
        (root / "config").mkdir()
        (root / "market_data").mkdir()
        (root / "config/market.yaml").symlink_to(ROOT / "config/market.yaml")
        conf = yaml.safe_load((ROOT / "config/sentiment.yaml").read_text())
        conf["category_keywords"] = {}
        (root / "config/sentiment.yaml").write_text(yaml.safe_dump(conf, allow_unicode=True))
        (root / "market_data/sentiment").symlink_to(ROOT / "market_data/sentiment")
        return build(root=root)


def held(weights):
    return weights.apply(lambda row: "|".join(row.index[row.gt(1e-8)]) or "cash", axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    market = yaml.safe_load((ROOT / "config/market.yaml").read_text())
    config = yaml.safe_load((ROOT / "config/ye_strategy.yaml").read_text())
    assert str(market["project"]["data_end"]) == END
    guarded = [*sorted((ROOT / "config").glob("*")),
               *sorted((ROOT / "results/live").glob("*")),
               *sorted((ROOT / "results/ye_strategy").glob("*")),
               *sorted((ROOT / "outputs").glob("*.html"))]
    guarded = [p for p in guarded if p.is_file()]
    before = {str(p.relative_to(ROOT)): digest(p) for p in guarded}
    panel = load_panel(market, ROOT / "market_data/prices")
    symbols = universe_keys(market)
    categories = {symbol_key(x): x["category"] for x in market["universe"]}
    calendar = panel["close"].index
    raw = pd.read_csv(ROOT / "market_data/sentiment/features/symbol_daily.csv")
    narrow = specific_frame()
    premium = [s for s in symbols if s.startswith("513") or s == "159941.SZ"]
    cash = config["cash_management"]
    cash_model = {"annual_rate": cash["historical_backtest_annual_rate"],
                  **{k: cash[k] for k in ("fee_rate", "minimum_order", "order_lot")}}
    records, phases, trip_rows, differences, gates = [], [], [], [], []
    full = {}
    contexts = {}
    for variant in VARIANTS:
        frame = narrow if variant in {"specific_mapping", "live_schema"} else raw
        if variant in {"no_dde", "live_schema"}:
            frame = remove_dde(frame)
        sentiment, available = sentiment_matrices_from_frame(frame, calendar, symbols)
        components = None
        if variant == "simple_price":
            available[:] = True
            components = {key: False for key in (
                "weak_edge_filter", "emerging_trend", "quality_extension", "hot_exit_protection")}
        breadth_fn = observed_breadth if variant == "observed_breadth" else ye.anchor_category_breadth
        with patch.object(ye, "anchor_category_breadth", breadth_fn):
            bundle, features, eligibility, _, _, context = ye.build_ye_signals(
                panel, symbols, categories, config, sentiment, available, components=components)
        contexts[variant] = context
        for start in STARTS:
            weights = signals(panel, symbols, config, context, eligibility, start, False)
            if start == STARTS[0]:
                assert np.allclose(weights.loc[start:], bundle.weights.loc[start:])
            scenarios = [(1, 100000)]
            if start == STARTS[0]:
                scenarios += [(2, 100000), (1, 9825)]
            for multiplier, capital in scenarios:
                project = execution_project(market, premium, eligibility.shift(1, fill_value=False))
                project["initial_capital"] = capital
                for costs in [project, *project["symbol_costs"].values()]:
                    for key in ("commission_rate", "slippage_rate", "minimum_commission"):
                        costs[key] *= multiplier
                result = run_backtest(variant, panel, weights, start, END, project, cash_management=cash_model)
                removed, best = remove_best_round(result, capital)
                records.append({"variant": variant, "start": start, "cost": multiplier,
                                "capital": capital, **result.metrics,
                                "without_best_round_return": removed})
                if start == STARTS[0] and multiplier == 1 and capital == 100000:
                    full[variant] = result
                    for label, first, last in PERIODS:
                        phases.append({"variant": variant, "period": label,
                                       **period_metrics(result.equity, first, last, capital)})
                    for row in realized_round_trips(result.trades, result.equity.index).to_dict("records"):
                        trip_rows.append({"variant": variant, **row})
                    if variant == "baseline":
                        formal = json.loads((ROOT / "results/ye_strategy/summary.json").read_text())
                        assert abs(result.metrics["total_return"] - formal["metrics"]["total_return"]) < 1e-9
            print(f"{variant} {start} complete", flush=True)
        for key in ("current_normal", "emerging", "quality_extension", "hot_exit_protection", "entry_gate"):
            mask = context[key] & eligibility
            for label, first, last in PERIODS:
                part = mask.loc[first:last]
                gates.append({"variant": variant, "gate": key, "period": label,
                              "symbol_days": int(part.sum().sum()), "days": int(part.any(axis=1).sum())})
        position = held(full[variant].actual_weights)
        baseline_position = held(full["baseline"].actual_weights)
        for day in position.index[position.ne(baseline_position)]:
            differences.append({"variant": variant, "date": day,
                                "baseline": baseline_position.at[day], "alternative": position.at[day]})

    # Current data contract audit: source-level completeness, not keyword guess.
    dates, by_date = load_rows()
    source_rows = []
    for day, rows in by_date.items():
        if day > pd.Timestamp(END):
            continue
        regime = "ai_review" if rows and rows[0].get("_ai_reviewed") else "keyword_proxy"
        valid_dde = sum(pd.notna(row.get("dde_net") if row.get("dde_net") is not None
                               else row.get("ddejingliang")) for row in rows)
        source_rows.append({"date": day, "regime": regime, "rows": len(rows),
                            "dde_present": valid_dde})
    pd.DataFrame(source_rows).to_csv(OUT / "source_coverage.csv", index=False)
    regime = source_regimes(calendar).set_index("date")["regime"]
    f = raw.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f[f.date.le(END)]
    f["regime"] = f.date.map(regime)
    features_audit = f.groupby("regime").agg(
        days=("date", "nunique"), symbol_days=("symbol", "size"),
        nonzero_matches=("matched_count", lambda s: int(s.gt(0).sum())),
        dde_pass=("positive_dde_share", lambda s: int(s.ge(.5).sum())),
        negative_days=("ai_negative_count", lambda s: int(s.gt(0).sum()))).reset_index()

    b = contexts["baseline"]["category_breadth"]
    o = contexts["observed_breadth"]["category_breadth"]
    breadth_records = []
    for day in ("2018-07-02", "2019-06-03", "2021-06-01", "2023-06-01"):
        for symbol in ("518880.SH", "512010.SH", "512660.SH"):
            members = [s for s in contexts["baseline"]["core_symbols"]
                       if categories[s] == categories[symbol]]
            breadth_records.append({"date": day, "symbol": symbol, "category": categories[symbol],
                                    "core_peers": len(members),
                                    "observed_peers": int(features.roc_short.loc[day, members].notna().sum()),
                                    "baseline": b.at[pd.Timestamp(day), symbol],
                                    "observed": o.at[pd.Timestamp(day), symbol]})

    # Execution fragility, same signal, no signal optimization.
    base_w = signals(panel, symbols, config, contexts["baseline"], eligibility, STARTS[0], False)
    stresses = []
    for name in ("one_day_late", "scale_x10"):
        p = execution_project(market, premium, eligibility.shift(1, fill_value=False))
        test_panel = panel
        w = base_w
        if name == "one_day_late":
            w = w.shift(1, fill_value=0)
        else:
            test_panel = {k: v * 10 if k in ("open", "high", "low", "close") else v
                          for k, v in panel.items()}
        result = run_backtest(name, test_panel, w, STARTS[0], END, p, cash_management=cash_model)
        stresses.append({"case": name, **result.metrics})
        print(f"stress {name} complete", flush=True)

    # Price-data missingness and large-return anomalies; do not correct silently.
    anomalies = []
    for s in symbols:
        px = panel["close"][s]
        active = px.loc[px.first_valid_index():]
        r = px.pct_change(fill_method=None)
        for day in r.index[r.abs().gt(.25)]:
            anomalies.append({"symbol": s, "date": day, "return": r.at[day],
                              "close": px.at[day], "kind": "abs_return_gt_25pct"})
        if active.isna().any():
            anomalies.append({"symbol": s, "kind": "missing_after_first",
                              "count": int(active.isna().sum())})

    metrics = pd.DataFrame(records)
    checks = {}
    for variant in VARIANTS[1:]:
        def row(v, start=STARTS[0], cost=1, capital=100000):
            return metrics.query("variant == @v and start == @start and cost == @cost and capital == @capital").iloc[0]
        a, c = row("baseline"), row(variant)
        phase = pd.DataFrame(phases).pivot(index="period", columns="variant", values="total_return")
        checks[variant] = {
            "return_and_sharpe": bool(c.total_return > a.total_return and c.sharpe > a.sharpe),
            "drawdown_within_2pp": bool(c.max_drawdown >= a.max_drawdown - .02),
            "starts_won": sum(row(variant, s).total_return > row("baseline", s).total_return for s in STARTS),
            "periods_won": int(phase[variant].gt(phase.baseline).sum()),
            "double_cost": bool(row(variant, cost=2).total_return > row("baseline", cost=2).total_return),
            "small_capital": bool(row(variant, capital=9825).total_return > row("baseline", capital=9825).total_return),
        }
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in guarded}
    for name, rows in (("metrics", records), ("periods", phases), ("trades", trip_rows),
                       ("position_differences", differences), ("gate_counts", gates),
                       ("breadth_examples", breadth_records), ("execution_stress", stresses),
                       ("price_anomalies", anomalies)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    features_audit.to_csv(OUT / "feature_regimes.csv", index=False)
    pd.DataFrame({k: v.equity for k, v in full.items()}).to_csv(OUT / "equity.csv")
    inputs = [PROTOCOL, Path(__file__), ROOT / "experiments/rank_exit_review.py",
              *sorted((ROOT / "src/etf_rotation").glob("*.py")),
              ROOT / "scripts/build_sentiment_features.py",
              ROOT / "market_data/sentiment/features/symbol_daily.csv",
              *sorted((ROOT / "market_data/prices").glob("*.csv"))]
    payload = {"end": END, "variants": list(VARIANTS), "run_count": len(records) + len(stresses),
               "checks": checks, "formal_unchanged_sha256": before,
               "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
               "source_tree_sha256": {folder: digest_tree(ROOT / folder) for folder in (
                   "market_data/sentiment/ths_hot_reason", "market_data/sentiment/ai_review")},
               "warning": "retrospective sensitivity, not independent/OOS performance or investable live plan"}
    (OUT / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(checks, ensure_ascii=False, default=str), flush=True)


def digest_tree(folder):
    import hashlib
    return hashlib.sha256("\n".join(f"{p.name}:{digest(p)}" for p in sorted(folder.glob("*.json"))).encode()).hexdigest()


if __name__ == "__main__":
    main()
