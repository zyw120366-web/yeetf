"""Read-only source-contract diagnosis and post-run attribution; no new rules."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from unittest.mock import patch
from experiments.rank_exit_review import ROOT, digest
from experiments.joint_realism_20260914 import OUT, END
from scripts.build_sentiment_features import load_rows
from scripts import build_sentiment_features as feature_builder
from etf_rotation.sentiment_ai import deduplicate_reviewed_rows, canonical_event_key
from etf_rotation.sentiment import sentiment_matrices_from_frame
from etf_rotation.data import load_panel, universe_keys, symbol_key
from etf_rotation.ye import build_ye_signals


def present(row, field):
    value = row.get(field)
    return value is not None and pd.notna(value)


def main():
    _, days = load_rows()
    coverage, lost = [], []
    for day, rows in days.items():
        if day > pd.Timestamp(END) or not rows or not rows[0].get("_ai_reviewed"):
            continue
        positive = [r for r in rows if r["ai"]["relevant"] and r["ai"]["direction"] > 0]
        groups = defaultdict(list)
        for row in positive:
            groups[row.get("normalization", {}).get("event_key") or canonical_event_key(row)].append(row)
        picked = deduplicate_reviewed_rows(positive)
        reverse = deduplicate_reviewed_rows(reversed(positive))
        reverse_map = {r.get("normalization", {}).get("event_key") or canonical_event_key(r): r for r in reverse}
        for row in picked:
            key = row.get("normalization", {}).get("event_key") or canonical_event_key(row)
            group = groups[key]
            for field in ("dde_net", "turnover"):
                alternatives = [r for r in group if present(r, field)]
                if not present(row, field) and alternatives:
                    lost.append({"date": str(day.date()), "company": row.get("name"), "field": field,
                                 "selected_source_hash": row["source_hash"], "selected_source": row["source"],
                                 "alternative_source_hash": alternatives[0]["source_hash"],
                                 "alternative_source": alternatives[0]["source"],
                                 "alternative_value": alternatives[0][field],
                                 "reverse_order_recovers_field": present(reverse_map[key], field)})
        coverage.append({"date": str(day.date()), "rows": len(rows), "positive_rows": len(positive), "dedup_positive_events": len(picked),
                         "dde_present": sum(present(r, "dde_net") for r in rows),
                         "turnover_present": sum(present(r, "turnover") for r in rows),
                         "selected_dde_present": sum(present(r, "dde_net") for r in picked),
                         "selected_turnover_present": sum(present(r, "turnover") for r in picked),
                         "order_changes_selected_hash": sum(row["source_hash"] != reverse_map[row.get("normalization", {}).get("event_key") or canonical_event_key(row)]["source_hash"] for row in picked)})
    pd.DataFrame(coverage).to_csv(OUT / "live_field_coverage.csv", index=False)
    pd.DataFrame(lost).to_csv(OUT / "dedup_field_losses.csv", index=False)
    evidence = {"audit_scope": "positive-market aggregate before symbol-specific filtering; not a return counterfactual",
                "days": len(coverage), "rows": sum(x["rows"] for x in coverage),
                "no_dde_days": sum(x["dde_present"] == 0 for x in coverage),
                "field_losses": len(lost), "order_recoverable_field_losses": sum(x["reverse_order_recovers_field"] for x in lost),
                "losses_by_field": dict(pd.Series([x["field"] for x in lost]).value_counts().astype(int)),
                "source_code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), ROOT / "src/etf_rotation/sentiment_ai.py", ROOT / "scripts/build_sentiment_features.py")}}
    # Permutation only: same frozen rows, no new fields, labels or prices.
    dates, unchanged = load_rows()
    reversed_rows = {day: list(reversed(rows)) if rows and rows[0].get("_ai_reviewed") else rows
                     for day, rows in unchanged.items()}
    normal = feature_builder.build().set_index(["date", "symbol"]).sort_index()
    with patch.object(feature_builder, "load_rows", return_value=(dates, reversed_rows)):
        reordered = feature_builder.build().set_index(["date", "symbol"]).sort_index()
    cutoff = normal.index.get_level_values("date") <= pd.Timestamp(END)
    normal, reordered = normal.loc[cutoff], reordered.loc[cutoff]
    unequal = ~np.isclose(normal.hot_score, reordered.hot_score, equal_nan=True)
    changed = pd.DataFrame({"original_hot_score": normal.hot_score[unequal], "reversed_hot_score": reordered.hot_score[unequal],
                            "original_turnover_share": normal.matched_turnover_share[unequal],
                            "reversed_turnover_share": reordered.matched_turnover_share[unequal]})
    changed.to_csv(OUT / "source_order_feature_changes.csv")
    evidence["permutation_hot_score_changed_symbol_days"] = len(changed)
    evidence["permutation_hot_score_changed_days"] = changed.index.get_level_values("date").nunique()
    evidence["permutation_max_abs_hot_score_difference"] = float((changed.original_hot_score - changed.reversed_hot_score).abs().max())
    market = yaml.safe_load((ROOT / "config/market.yaml").read_text())
    config = yaml.safe_load((ROOT / "config/ye_strategy.yaml").read_text())
    assert str(market["project"]["data_end"]) == END
    panel = load_panel(market, ROOT / "market_data/prices")
    symbols = universe_keys(market)
    categories = {symbol_key(x): x["category"] for x in market["universe"]}
    bundles, contexts = [], []
    for frame in (normal, reordered):
        sentiment, available = sentiment_matrices_from_frame(frame.reset_index(), panel["close"].index, symbols)
        bundle, _, _, _, _, context = build_ye_signals(panel, symbols, categories, config, sentiment, available)
        bundles.append(bundle)
        contexts.append(context)
    gate_changes = []
    for key in ("entry_gate", "quality_extension", "hot_exit_protection", "emerging"):
        different = contexts[0][key].ne(contexts[1][key])
        for day, symbol in different.stack().loc[lambda s: s].index:
            gate_changes.append({"date": str(day.date()), "symbol": symbol, "gate": key,
                                 "original": bool(contexts[0][key].at[day, symbol]),
                                 "reversed": bool(contexts[1][key].at[day, symbol])})
    pd.DataFrame(gate_changes).to_csv(OUT / "source_order_gate_changes.csv", index=False)
    evidence["permutation_gate_changes"] = len(gate_changes)
    evidence["permutation_signal_weight_changed_days"] = int(bundles[0].weights.ne(bundles[1].weights).any(axis=1).sum())
    evidence["permutation_signal_basis"] = "unchanged formal frozen prices and config; recomputed features, order-only perturbation; no account or execution changes"
    if (OUT / "positions.csv").exists():
        positions = pd.read_csv(OUT / "positions.csv", index_col=0, parse_dates=True)
        equity = pd.read_csv(OUT / "equity.csv", index_col=0, parse_dates=True)
        windows = []
        for variant in positions.columns:
            base = "joint_valid" if variant.startswith("without_") else "rich"
            different = positions[variant].ne(positions[base])
            run_id = different.ne(different.shift(1)).cumsum()
            for _, mask in different.groupby(run_id):
                if not mask.iloc[0]:
                    continue
                first, last = mask.index[0], mask.index[-1]
                idx = equity.index.get_loc(first)
                bases = equity.iloc[idx - 1] if idx else pd.Series(100000., index=equity.columns)
                left = equity.at[last, base] / bases[base] - 1
                right = equity.at[last, variant] / bases[variant] - 1
                windows.append({"variant": variant, "baseline": base, "start": str(first.date()), "end": str(last.date()),
                                "sessions": len(mask), "base_positions": "|".join(positions.loc[first:last, base].unique()),
                                "alternative_positions": "|".join(positions.loc[first:last, variant].unique()),
                                "base_window_return": left, "alternative_window_return": right,
                                "difference_pp": (right-left)*100})
        pd.DataFrame(windows).to_csv(OUT / "divergence_windows.csv", index=False)
        evidence["window_caveat"] = "consecutive differing holdings; not matched independent trades, not additive contributions; endpoints include execution days"
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=int))
    print(json.dumps(evidence, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
