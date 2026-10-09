"""Follow an observed price-basis anomaly; no signal-parameter optimization."""
from __future__ import annotations
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from easy_tdx import UnifiedTdxClient, TdxClient, Market, Period, Adjust
from experiments.rank_exit_review import ROOT, STARTS, PERIODS, digest, signals
from etf_rotation import backtest as engine
from etf_rotation.data import load_panel, universe_keys, symbol_key
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals
from etf_rotation.evaluation import realized_round_trips

OUT = ROOT / "results/research/structural_audit_20260911/price_basis"
END = "2026-09-10"


def number(row, key, default=0.):
    value = row.get(key)
    return float(value) if value is not None and pd.notna(value) else default


def events_from(frame, raw):
    events = {}
    unsupported = []
    for row in frame.to_dict("records"):
        day = pd.Timestamp(row["date"])
        if day > pd.Timestamp(END) or day < raw.index[0]:
            continue
        category = int(row["category"])
        if category == 1:
            ratio = 1 + number(row, "songzhuangu") + number(row, "peigu")
            dividend = number(row, "fenhong") - number(row, "peigu") * number(row, "peigujia")
        elif category in (11, 12):
            ratio, dividend = number(row, "suogu", 1.), 0.
        else:
            # Capital amount changes are issuance/redemption, not per-share return.
            if category in (13, 14):
                unsupported.append(row)
            continue
        if day not in raw.index:
            future = raw.index[raw.index > day]
            if not len(future):
                unsupported.append(row)
                continue
            day = future[0]
        if ratio <= 0 or day in events:
            raise ValueError("invalid or duplicate share action")
        events[day] = {"ratio": ratio, "dividend": dividend}
    return events, unsupported


def economic_close(raw, events):
    px = raw.close
    factors = px / px.shift(1)
    for day, event in events.items():
        previous = px.loc[px.index < day]
        if len(previous):
            factors.at[day] = (px.at[day] * event["ratio"] + event["dividend"]) / previous.iloc[-1]
    factors.iloc[0] = 1.
    if factors.isna().any() or not factors.gt(0).all():
        raise ValueError("invalid total-return factor")
    return factors.cumprod() * px.iloc[0]


def raw_execution_function():
    """Reuse exact production engine, inserting explicit corporate cash/shares.

    No production file changes; sentinel assertions fail if upstream source moves.
    The emitted adapted source is retained for inspection.
    """
    source = inspect.getsource(engine.run_backtest)
    sentinel = "        loc = calendar.get_loc(date)"
    assert source.count(sentinel) == 1
    source = source.replace(sentinel, '''
        # Research-only corporate action accounting, credited at ex-date.
        day_actions = project.get("corporate_actions", {}).get(date, {})
        for s, action in day_actions.items():
            old_quantity = float(shares[s])
            cash += old_quantity * action["dividend"]
            shares[s] = old_quantity * action["ratio"]
            last_close[s] = (last_close[s] - action["dividend"]) / action["ratio"]
        loc = calendar.get_loc(date)''')
    sentinel = "            previous_close = close.iloc[loc - 1]"
    assert source.count(sentinel) == 1
    source = source.replace(sentinel, '''
            previous_close = close.iloc[loc - 1].copy()
            for s, action in day_actions.items():
                previous_close[s] = (previous_close[s] - action["dividend"]) / action["ratio"]''')
    (OUT / "adapted_execution.py").write_text(source)
    namespace = {"np": np, "pd": pd, "BacktestResult": engine.BacktestResult, "_metrics": engine._metrics}
    exec(compile(source, str(OUT / "adapted_execution.py"), "exec"), namespace)
    return namespace["run_backtest"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    market = yaml.safe_load((ROOT / "config/market.yaml").read_text())
    config = yaml.safe_load((ROOT / "config/ye_strategy.yaml").read_text())
    assert str(market["project"]["data_end"]) == END
    symbols = universe_keys(market)
    categories = {symbol_key(x): x["category"] for x in market["universe"]}
    guarded = [*sorted((ROOT / "market_data/prices").glob("*.csv")),
               ROOT / "config/ye_strategy.yaml", ROOT / "results/live/account_state.json",
               ROOT / "results/ye_strategy/summary.json", ROOT / "results/ye_strategy/signal_weights.csv"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in guarded}
    raws, corp, unsupported = {}, {}, []
    # Cache inputs for deterministic reruns; never overwrite research snapshots.
    with UnifiedTdxClient(timeout=15) as c, TdxClient.from_best_host(timeout=15) as tc:
        for symbol in symbols:
            code, exchange = symbol.split(".")
            m = Market.SH if exchange == "SH" else Market.SZ
            raw_path, action_path = OUT / f"{symbol}_raw.csv", OUT / f"{symbol}_actions.csv"
            if not raw_path.exists():
                f = c.get_stock_kline(m.value, code, Period.DAILY, 0, 3000, 2, Adjust.NONE)
                f = f[f.datetime.le(pd.Timestamp(END))]
                if f.empty:
                    raise ValueError(f"empty raw prices {symbol}")
                f.to_csv(raw_path, index=False)
            if not action_path.exists():
                action = tc.get_xdxr_info(m, code)
                if action is None:
                    raise ValueError(f"missing action response {symbol}")
                if action.empty and not len(action.columns):
                    action = pd.DataFrame(columns=["date", "category"])
                action.to_csv(action_path, index=False)
            raw = pd.read_csv(raw_path, parse_dates=["datetime"]).set_index("datetime").sort_index()
            action = pd.read_csv(action_path)
            ev, unsup = events_from(action, raw)
            raws[symbol], corp[symbol] = raw, ev
            unsupported += [{"symbol": symbol, **x} for x in unsup]
            print(f"raw/actions {symbol}: {len(raw)} bars, {len(ev)} actions", flush=True)
    if unsupported:
        (OUT / "unsupported.json").write_text(json.dumps(unsupported, default=str))
        raise ValueError("unsupported corporate actions; no silent approximation")
    frozen = load_panel(market, ROOT / "market_data/prices")
    calendar = frozen["close"].index
    raw_panel = {key: pd.DataFrame({s: f[key].reindex(calendar) for s, f in raws.items()})
                 for key in ("open", "high", "low", "close", "amount", "vol")}
    economic_panel = dict(raw_panel)
    economic_panel["close"] = pd.DataFrame({
        s: economic_close(f, corp[s]).reindex(calendar) for s, f in raws.items()})
    sentiment, available = load_sentiment_matrices(
        ROOT / "market_data/sentiment/features/symbol_daily.csv", calendar, symbols)
    contexts = {}
    eligibilities = {}
    for name, panel in (("frozen", frozen), ("economic", economic_panel)):
        _, _, eligibilities[name], _, _, contexts[name] = build_ye_signals(
            panel, symbols, categories, config, sentiment, available)
    premium = [s for s in symbols if s.startswith("513") or s == "159941.SZ"]
    action_map = {}
    for s, events in corp.items():
        for day, event in events.items():
            action_map.setdefault(day, {})[s] = event
    cash = config["cash_management"]
    cm = {"annual_rate": cash["historical_backtest_annual_rate"],
          **{key: cash[key] for key in ("fee_rate", "minimum_order", "order_lot")}}
    raw_run = raw_execution_function()
    metrics, phases, curves, trades = [], [], {}, []
    for variant in ("frozen_baseline", "old_signal_raw_execution", "economic_signal_raw_execution"):
        basis = "economic" if variant.startswith("economic") else "frozen"
        for start in STARTS:
            weights = signals(economic_panel if basis == "economic" else frozen,
                              symbols, config, contexts[basis], eligibilities[basis], start, False)
            scenarios = [(1, 100000)]
            if start == STARTS[0]:
                scenarios += [(2, 100000), (1, 9825)]
            for multiplier, capital in scenarios:
                project = execution_project(market, premium, eligibilities[basis].shift(1, fill_value=False))
                project["initial_capital"] = capital
                project["corporate_actions"] = action_map
                for costs in [project, *project["symbol_costs"].values()]:
                    for key in ("commission_rate", "slippage_rate", "minimum_commission"):
                        costs[key] *= multiplier
                run = engine.run_backtest if variant == "frozen_baseline" else raw_run
                result = run(variant, frozen if variant == "frozen_baseline" else raw_panel,
                             weights, start, END, project, cash_management=cm)
                metrics.append({"variant": variant, "start": start, "cost": multiplier,
                                "capital": capital, **result.metrics})
                if start == STARTS[0] and multiplier == 1 and capital == 100000:
                    curves[variant] = result.equity
                    result.trades.to_csv(OUT / f"{variant}_fills.csv", index=False)
                    if variant == "frozen_baseline":
                        formal = json.loads((ROOT / "results/ye_strategy/summary.json").read_text())
                        assert abs(formal["metrics"]["total_return"] - result.metrics["total_return"]) < 1e-9
                    for label, first, last in PERIODS:
                        phases.append({"variant": variant, "period": label,
                                       **period_metrics(result.equity, first, last, capital)})
            print(f"{variant} {start} complete", flush=True)
    examples = []
    for symbol, start, end in (("512690.SH", "2020-01-23", "2020-02-03"),
                               ("515050.SH", "2025-07-04", "2025-11-03"),
                               ("515050.SH", "2026-04-08", "2026-06-30")):
        row = {"symbol": symbol, "start": start, "end": end}
        for label, panel in (("frozen", frozen), ("raw", raw_panel), ("economic", economic_panel)):
            values = panel["close"][symbol]
            row[label + "_return"] = values.at[pd.Timestamp(end)] / values.at[pd.Timestamp(start)] - 1
        row["actions_in_window"] = sum(pd.Timestamp(start) < d <= pd.Timestamp(end) for d in corp[symbol])
        examples.append(row)
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(phases).to_csv(OUT / "periods.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "equity.csv")
    pd.DataFrame(examples).to_csv(OUT / "return_examples.csv", index=False)
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in guarded}
    (OUT / "audit.json").write_text(json.dumps({
        "status": "research_proxy_not_promoted", "run_count": len(metrics), "end": END,
        "method": "causal per-date total-return signal; raw-price fills, ex-date cash dividend and share adjustments",
        "limitations": ["same vendor raw and action data, not independent feed", "dividend payment dates unavailable; credited ex-date",
                       "current surviving ETF universe unchanged", "no live account or production files changed"],
        "formal_unchanged": before, "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
            [Path(__file__), *sorted(OUT.glob("*_raw.csv")), *sorted(OUT.glob("*_actions.csv"))]}
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
