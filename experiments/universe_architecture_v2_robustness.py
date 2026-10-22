"""ETF 扩池结构 v2 强制稳健性检查（任务书 §10）。

针对当前唯一满足路径隔离、且是候选上线架构的 champion_cash_gap，检查：

结构不变量：
  I1 诱饵ETF：加入一只永不满足上市/流动性的假ETF，逐日信号与净值必须完全不变。
  I2 顺序无关：打乱挑战者在配置中的顺序，逐日权重必须完全一致。
  I3 核心排名/退出不变：核心45只的排名与排名退出与纯45只版完全一致。
  I4 占位=0：挑战者占仓且核心已有合格候选的天数必须为0。

增量ETF稳定性：
  加一（逐只挑战者单独加入）、留一（六选五）、多顺序，报告各自终值与挑战者成交。

输出：results/research/universe_architecture_v2/robustness.csv 与 path_differences.csv
"""

from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.execution import execution_project
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals

FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
OUTPUT = ROOT / "results" / "research" / "universe_architecture_v2"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def premium_sensitive(symbols):
    return [s for s in symbols if s.split(".")[0].startswith("513") or s == "159941.SZ"]


def cash_management(config):
    c = config["cash_management"]
    return {"annual_rate": float(c["historical_backtest_annual_rate"]),
            "fee_rate": float(c["fee_rate"]), "minimum_order": float(c["minimum_order"]),
            "order_lot": float(c["order_lot"])}


def run(market, formal, panel, symbols, categories, sentiment_all, available, mode, challengers):
    config = copy.deepcopy(formal)
    arch = config["enhanced_selection"]["universe_architecture"]
    arch["mode"] = mode
    arch["core_pool_size"] = len([s for s in symbols if s not in set(challengers)])
    arch["challenger_symbols"] = list(challengers)
    cats = {s: categories[s] for s in symbols}
    sent = {n: f[symbols] for n, f in sentiment_all.items()}
    bundle, _, elig, _, _, decision = build_ye_signals(panel, symbols, cats, config, sent, available)
    project = execution_project(market, premium_sensitive(symbols),
                                elig.shift(1, fill_value=False).astype(bool))
    result = run_backtest("v", panel, bundle.weights, str(market["project"]["backtest_start"]),
                          str(market["project"]["data_end"]), project,
                          cash_management=cash_management(config))
    return bundle, decision, result


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    formal = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    all_symbols = universe_keys(market)
    core_size = int(formal["enhanced_selection"]["universe_architecture"]["core_pool_size"])
    core_symbols = all_symbols[:core_size]
    challengers = [str(s) for s in formal["enhanced_selection"]["universe_architecture"]["challenger_symbols"]]
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment_all, available = load_sentiment_matrices(FEATURES, panel["close"].index, all_symbols)

    results = {}
    rows = []

    # 基准：核心45 与 当前 champion_cash_gap 51
    b_core, d_core, r_core = run(market, formal, panel, core_symbols, categories,
                                 sentiment_all, available, "core_anchor_challenger", [])
    b_champ, d_champ, r_champ = run(market, formal, panel, all_symbols, categories,
                                    sentiment_all, available, "core_champion_cash_gap", challengers)
    results["core_45"] = (b_core, r_core)
    results["champion_51"] = (b_champ, r_champ)

    # ---- I1 诱饵ETF：注入一只永不合格的假ETF（价格极低成交额为0）----
    decoy = "999999.SH"
    panel_decoy = {k: v.copy() for k, v in panel.items()}
    for field in panel_decoy:
        panel_decoy[field][decoy] = np.nan
    # 给一点点价格但成交额始终为0 → 永远无法通过流动性门槛
    panel_decoy["close"][decoy] = 1.0
    panel_decoy["open"][decoy] = 1.0
    panel_decoy["high"][decoy] = 1.0
    panel_decoy["low"][decoy] = 1.0
    panel_decoy["vol"][decoy] = 0.0
    panel_decoy["amount"][decoy] = 0.0
    cats_decoy = dict(categories)
    cats_decoy[decoy] = "宽基"
    symbols_decoy = all_symbols + [decoy]
    # 诱饵作为"新增但永不合格的挑战者"注入：核心仍45只，测试永不合格的新增ETF
    # 是否会改变任何信号与净值。
    b_decoy, d_decoy, r_decoy = run(market, formal, panel_decoy, symbols_decoy, cats_decoy,
                                    {n: f.reindex(columns=list(f.columns) + [decoy]) for n, f in sentiment_all.items()},
                                    available, "core_champion_cash_gap", challengers + [decoy])
    common = r_champ.equity.index.intersection(r_decoy.equity.index)
    i1_equity_identical = bool(np.allclose(r_champ.equity.loc[common].to_numpy(),
                                           r_decoy.equity.loc[common].to_numpy(), atol=1e-6))
    w1 = b_champ.weights
    w2 = b_decoy.weights.reindex(columns=list(w1.columns) + [decoy], fill_value=0.0)
    i1_weights_identical = bool(np.allclose(
        w1.fillna(0).to_numpy(), w2[w1.columns].fillna(0).to_numpy(), atol=1e-12))
    rows.append({"check": "I1_decoy_equity_unchanged", "pass": i1_equity_identical})
    rows.append({"check": "I1_decoy_weights_unchanged", "pass": i1_weights_identical})

    # ---- I2 顺序无关：打乱挑战者顺序 ----
    shuffled = list(reversed(challengers))
    b_shuf, d_shuf, r_shuf = run(market, formal, panel, all_symbols, categories,
                                 sentiment_all, available, "core_champion_cash_gap", shuffled)
    i2 = bool(np.allclose(b_champ.weights.fillna(0).to_numpy(),
                          b_shuf.weights[b_champ.weights.columns].fillna(0).to_numpy(), atol=1e-12))
    rows.append({"check": "I2_challenger_order_invariant", "pass": i2})

    # 另一种：整体 universe 顺序打乱（核心也换位），核心排名应仍一致
    perm = all_symbols[::-1]
    # 保持核心/挑战者集合不变，只换列顺序传入
    b_perm, d_perm, r_perm = run(market, formal, panel, perm, categories,
                                 sentiment_all, available, "core_champion_cash_gap", challengers)
    i2b = bool(np.allclose(b_champ.weights.fillna(0).to_numpy(),
                           b_perm.weights[b_champ.weights.columns].fillna(0).to_numpy(), atol=1e-12))
    rows.append({"check": "I2b_full_universe_order_invariant", "pass": i2b})

    # ---- I3 核心排名与排名退出不变 ----
    i3_rank = bool(d_core["entry_rank"].equals(d_champ["entry_rank"][core_symbols]))
    i3_exit = bool(d_core["dual_rank_decline"].equals(d_champ["dual_rank_decline"][core_symbols]))
    rows.append({"check": "I3_core_rank_unchanged", "pass": i3_rank})
    rows.append({"check": "I3_core_rank_exit_unchanged", "pass": i3_exit})

    # ---- I4 占位=0 ----
    core_avail = b_champ.diagnostics["priority_entry_available"].astype(bool)
    ch_held = b_champ.weights[challengers].sum(axis=1).gt(0.0)
    i4_days = int((ch_held & core_avail).sum())
    rows.append({"check": "I4_challenger_held_while_core_available_days", "pass": i4_days == 0, "value": i4_days})

    # ---- I5 同题材复制：复制传媒ETF一只价格完全相同的克隆，作为核心之外的挑战者，
    #        不应仅因产品数量增加而改变结果（补位方案下克隆与原挑战者竞争，权重不应翻倍）----
    clone = "512980C.SH"
    src = "512980.SH"
    panel_clone = {k: v.copy() for k, v in panel.items()}
    for field in panel_clone:
        panel_clone[field][clone] = panel_clone[field][src]
    cats_clone = dict(categories); cats_clone[clone] = categories[src]
    symbols_clone = all_symbols + [clone]
    sent_clone = {n: f.reindex(columns=list(f.columns) + [clone]) for n, f in sentiment_all.items()}
    for n in sent_clone:
        if src in sent_clone[n].columns:
            sent_clone[n][clone] = sent_clone[n][src]
    b_clone, d_clone, r_clone = run(market, formal, panel_clone, symbols_clone, cats_clone,
                                    sent_clone, available, "core_champion_cash_gap", challengers + [clone])
    max_exposure = float(b_clone.weights.sum(axis=1).max())
    i5 = bool(max_exposure <= 1.0 + 1e-9)
    rows.append({"check": "I5_theme_clone_no_double_weight", "pass": i5, "value": max_exposure})

    # ---- 增量稳定性：加一 / 留一 ----
    incr_rows = []
    for ch in challengers:
        b1, _, r1 = run(market, formal, panel, core_symbols + [ch], categories,
                        sentiment_all, available, "core_champion_cash_gap", [ch])
        tr = r1.trades
        n = int((tr["symbol"] == ch).sum()) if not tr.empty else 0
        incr_rows.append({"test": "add_one", "symbol": ch,
                          "total_return": float(r1.metrics["total_return"]),
                          "final_equity": float(r1.equity.iloc[-1]),
                          "challenger_trades": n,
                          "gap_vs_core45": float(r1.equity.iloc[-1] - r_core.equity.iloc[-1])})
    for drop in challengers:
        keep = [c for c in challengers if c != drop]
        syms = core_symbols + keep
        b2, _, r2 = run(market, formal, panel, syms, categories,
                        sentiment_all, available, "core_champion_cash_gap", keep)
        incr_rows.append({"test": "leave_one_out", "symbol": f"drop_{drop}",
                          "total_return": float(r2.metrics["total_return"]),
                          "final_equity": float(r2.equity.iloc[-1]),
                          "challenger_trades": np.nan,
                          "gap_vs_core45": float(r2.equity.iloc[-1] - r_core.equity.iloc[-1])})
    pd.DataFrame(incr_rows).to_csv(OUTPUT / "robustness_incremental.csv", index=False, encoding="utf-8-sig")

    # ---- 路径差异表（champion 相对 core45）----
    w_champ = b_champ.weights
    w_core = b_core.weights.reindex(columns=w_champ.columns, fill_value=0.0)
    common_idx = w_champ.index.intersection(w_core.index)
    diff_rows = []
    for date in common_idx:
        a = w_champ.loc[date].fillna(0)
        c = w_core.loc[date].fillna(0)
        if not np.allclose(a.to_numpy(), c.reindex(a.index).fillna(0).to_numpy(), atol=1e-9):
            champ_hold = a[a > 0].index.tolist()
            core_hold = c[c > 0].index.tolist()
            diff_rows.append({
                "date": str(date.date()),
                "champion_hold": "|".join(champ_hold),
                "core45_hold": "|".join(core_hold),
            })
    pd.DataFrame(diff_rows).to_csv(OUTPUT / "path_differences.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "robustness.csv", index=False, encoding="utf-8-sig")

    summary = {
        "target_variant": "champion_cash_gap_51",
        "invariants": {r["check"]: bool(r["pass"]) for r in rows},
        "invariant_values": {r["check"]: r.get("value") for r in rows if "value" in r},
        "champion_vs_core45_signal_diff_days": len(diff_rows),
        "all_invariants_pass": bool(all(r["pass"] for r in rows)),
    }
    (OUTPUT / "robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n增量稳定性（加一/留一）：")
    print(pd.DataFrame(incr_rows).to_string(index=False))
    print(f"\nchampion 相对 core45 的信号差异天数：{len(diff_rows)}")
    print(pd.DataFrame(diff_rows).to_string(index=False) if diff_rows else "（无差异）")


if __name__ == "__main__":
    main()
