"""ETF 扩池结构问题 v2 研究脚本。

目标：在不改动动量评分、MA120、买卖阈值、仓位、成本与执行口径的前提下，
用同一价格/资讯/成本/次日开盘执行，比较不同"可投资宇宙"架构：

  baseline_core_45          原45只核心池（对照基线，不含挑战者）
  global_rank_51            51只全局混排（历史已拒绝）
  anchor_equal_51           45核心+6挑战者平权竞争（历史已拒绝）
  champion_cash_gap_51      45核心冠军 + 挑战者空档补位（当前生产安全层 F）
  qualified_pool_51         方案A：全池资格前置 + 合格池排名（对称，无核心先验）
  theme_champion_51         方案B：点时合格 + 题材冠军两阶段竞赛（对称，无核心先验）
  challenger_dominance_51   方案E：全合格 + 挑战者需多维显著支配核心第一名才可替代

方案 C/D/F 属治理/版本机制，不产生与上面不同的逐日信号，在 summary 中单独论证，
不在此重复回测（F 已由 champion_cash_gap_51 代表）。

所有变体只改变"哪些 ETF 能进入当日买入候选/以什么标尺排名"，不改变：
  - ROC20/ROC60 与评分 ROC20 + 1.5×ROC60
  - MA120、9% 常规乖离、9%-12% 质量延伸
  - 新趋势/弱边缘/资讯审核规则与其阈值
  - 单只ETF或现金的仓位约束、T+1开盘执行、固定成本、收盘宝口径
  - 120日上市期、20日成交额中位数门槛

用法：
  PYTHONPATH=src python3 experiments/universe_architecture_v2_study.py
"""

from __future__ import annotations

import copy
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
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.sentiment import load_sentiment_matrices
from etf_rotation.ye import build_ye_signals


FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
OUTPUT = ROOT / "results" / "research" / "universe_architecture_v2"

PERIODS = {
    "2018—2020": ("2018-07-02", "2020-12-31"),
    "2021—2022": ("2021-01-01", "2022-12-31"),
    "2023—2024": ("2023-01-01", "2024-12-31"),
    "2025—2026": ("2025-01-01", "2026-07-27"),
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def premium_sensitive(symbols: list[str]) -> list[str]:
    return [
        symbol for symbol in symbols
        if symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"
    ]


def cash_management(config: dict) -> dict:
    cash = config["cash_management"]
    return {
        "annual_rate": float(cash["historical_backtest_annual_rate"]),
        "fee_rate": float(cash["fee_rate"]),
        "minimum_order": float(cash["minimum_order"]),
        "order_lot": float(cash["order_lot"]),
    }


def qualified_first_gate(entry_gate, decision, config, eligibility) -> pd.DataFrame:
    """方案A：资格前置 + 合格池排名。

    候选集 = 满足除"全池前5"外全部技术/资格/AI 条件的标的（pre_rank_gate）。
    在该合格集内部按选择分重新排名，取前 entry_rank_limit 只。
    不合格标的不占据排名名额，因此不能把合格标的挤出候选。
    """

    k = int(config["rules"]["entry_rank_limit"])
    pre = decision.get("pre_rank_gate", entry_gate).astype(bool)
    score = decision["entry_score"]
    qualified_scores = score.where(pre)
    qualified_rank = qualified_scores.rank(axis=1, ascending=False, method="min")
    return (pre & qualified_rank.le(k)).astype(bool)


def theme_champion_gate(
    entry_gate: pd.DataFrame,
    entry_score: pd.DataFrame,
    categories: dict[str, str],
) -> pd.DataFrame:
    """方案B：每个冻结题材内部只保留选择分最高的合格标的。

    只在 entry_gate 已经为 True 的候选中，按题材分组，每组仅留分数最高的一只。
    不改变任何 ETF 的资格判定、分数或排名标尺，只做"同题材去拥挤"。
    """

    groups: dict[str, list[str]] = {}
    for symbol, category in categories.items():
        groups.setdefault(category, []).append(symbol)
    champion = pd.DataFrame(False, index=entry_gate.index, columns=entry_gate.columns)
    for members in groups.values():
        if len(members) == 1:
            champion[members[0]] = entry_gate[members[0]]
            continue
        gated = entry_gate[members]
        scores = entry_score[members].where(gated)
        # 每日题材内最高分标的（无合格标的则该题材当日无冠军）
        valid = scores.notna().any(axis=1)
        best = scores.dropna(how="all").idxmax(axis=1).reindex(scores.index)
        for member in members:
            champion[member] = gated[member] & valid & (best == member)
    return champion.astype(bool)


def run_variant(
    key: str,
    market: dict,
    formal: dict,
    panel: dict,
    all_symbols: list[str],
    core_symbols: list[str],
    categories_all: dict[str, str],
    sentiment_all: dict,
    available: pd.Series,
    *,
    symbols: list[str],
    mode: str,
    challengers: list[str],
    theme_champion: bool = False,
    challenger_dominance: bool = False,
    qualified_first: bool = False,
) -> dict:
    config = copy.deepcopy(formal)
    architecture = config["enhanced_selection"]["universe_architecture"]
    architecture["mode"] = mode
    architecture["core_pool_size"] = len(core_symbols)
    architecture["challenger_symbols"] = list(challengers)
    categories = {symbol: categories_all[symbol] for symbol in symbols}
    sentiment = {name: frame[symbols] for name, frame in sentiment_all.items()}

    bundle, features, eligibility, _, _, decision = build_ye_signals(
        panel, symbols, categories, config, sentiment, available
    )

    # 方案A/B 需要"无 rank 名额限制"的资格门，用于在合格池内重新排名。
    if qualified_first or theme_champion:
        wide = copy.deepcopy(config)
        wide["rules"] = dict(wide["rules"])
        wide["rules"]["entry_rank_limit"] = 10_000
        arch = wide["enhanced_selection"]["universe_architecture"]
        arch["mode"] = "fixed_pool"
        arch["challenger_symbols"] = []
        _, _, _, _, _, wide_decision = build_ye_signals(
            panel, symbols, categories, wide, sentiment, available
        )
        decision["pre_rank_gate"] = wide_decision["entry_gate"]

    # 方案A/B/E：在既有 entry_gate 上做架构级改造，然后用原选择分重跑一次信号。
    if theme_champion or challenger_dominance or qualified_first:
        from etf_rotation.etfwin import etfwin_signals
        from etf_rotation.ye import _rules

        rules = _rules(config["rules"])
        entry_gate = decision["entry_gate"]
        entry_score = decision["entry_score"]
        dual_rank = decision["dual_rank_decline"]
        soft_exit = decision["soft_exit_confirmation"]

        if qualified_first or theme_champion:
            # 方案A：资格前置。把 entry_gate 里隐含的"全池 rank≤k"约束，
            # 替换为"仅在当日技术合格标的之间排名 rank≤k"。不合格标的不占名额。
            new_gate = qualified_first_gate(
                entry_gate, decision, config, eligibility
            )
            if theme_champion:
                # 方案B：在资格前置基础上再做题材冠军去拥挤。
                new_gate = theme_champion_gate(new_gate, entry_score, categories)
        else:  # challenger_dominance 方案E
            new_gate = challenger_dominance_gate(
                entry_gate, decision, core_symbols, challengers
            )
        bundle, _ = etfwin_signals(
            panel["close"][symbols],
            symbols,
            rules,
            entry_eligibility=eligibility,
            entry_gate=new_gate,
            entry_ranking_score_override=entry_score,
            soft_exit_confirmation=soft_exit,
            dual_rank_decline_override=dual_rank,
            reentry_cooldown_days=int(config["enhanced_selection"]["reentry_cooldown_days"]),
        )
        decision["entry_gate_effective"] = new_gate

    project = execution_project(
        market,
        premium_sensitive(symbols),
        eligibility.shift(1, fill_value=False).astype(bool),
    )
    result = run_backtest(
        key, panel, bundle.weights, str(market["project"]["backtest_start"]),
        str(market["project"]["data_end"]), project,
        cash_management=cash_management(config),
    )
    capital = float(market["project"]["initial_capital"])
    period_returns = {
        name: period_metrics(result.equity, s, e, capital)
        for name, (s, e) in PERIODS.items()
    }
    # 合格池规模与低分入场诊断
    eff_gate = decision.get("entry_gate_effective", decision["entry_gate"])
    qualified_count = eff_gate.sum(axis=1)
    return {
        "key": key,
        "metrics": result.metrics,
        "period_returns": period_returns,
        "equity": result.equity,
        "trades": result.trades,
        "weights": bundle.weights,
        "diagnostics": bundle.diagnostics,
        "decision": decision,
        "eligibility": eligibility,
        "qualified_pool_size_mean": float(qualified_count[qualified_count > 0].mean())
        if (qualified_count > 0).any() else 0.0,
        "qualified_pool_size_max": float(qualified_count.max()),
        "symbols": symbols,
    }


def challenger_dominance_gate(entry_gate, decision, core_symbols, challengers):
    """方案E：挑战者只有在多维显著优于核心第一名时才允许进入候选。

    维度：动量选择分、R²20、效率。要求三者均不弱于核心第一名，
    且选择分与至少一项质量维度明显更强（分别高出 10% 与 5%）。
    核心标的的 gate 保持不变；仅收紧挑战者的进入条件。
    """

    entry_score = decision["entry_score"]
    r2 = decision["r2_20"]
    eff = decision["efficiency20"]
    core = [s for s in core_symbols]
    ch = [s for s in challengers]
    new_gate = entry_gate.copy()
    if not ch:
        return new_gate.astype(bool)
    core_gate = entry_gate[core]
    core_score = entry_score[core].where(core_gate)
    # 核心第一名的分数/质量
    best_core_score = core_score.max(axis=1)
    _valid_core = core_score.notna().any(axis=1)
    best_core_symbol = core_score.dropna(how="all").idxmax(axis=1).reindex(core_score.index)
    best_r2 = pd.Series(np.nan, index=entry_gate.index)
    best_eff = pd.Series(np.nan, index=entry_gate.index)
    for date in entry_gate.index:
        sym = best_core_symbol.get(date)
        if isinstance(sym, str):
            best_r2[date] = r2.loc[date, sym] if sym in r2.columns else np.nan
            best_eff[date] = eff.loc[date, sym] if sym in eff.columns else np.nan
    has_core_candidate = core_gate.any(axis=1)
    for symbol in ch:
        cg = entry_gate[symbol]
        s = entry_score[symbol]
        rr = r2[symbol]
        ee = eff[symbol]
        dominate = (
            (s >= best_core_score * 1.10)
            & (rr >= best_r2)
            & (ee >= best_eff)
            & ((rr >= best_r2 * 1.05) | (ee >= best_eff * 1.05))
        ).fillna(False)
        # 无核心候选时，挑战者按原资格进入（补位）；有核心候选时须显著支配
        new_gate[symbol] = (cg & ~has_core_candidate) | (cg & has_core_candidate & dominate)
    return new_gate.astype(bool)


def analyse_incremental_trades(variant: dict, core_variant: dict, challengers: list[str]) -> dict:
    """量化扩池增量：挑战者交易、单笔最大贡献、相对核心版的终值差归因。"""

    trades = variant["trades"]
    challenger_trades = trades[trades["symbol"].isin(challengers)] if not trades.empty else trades
    n_ch_trades = int(len(challenger_trades))
    ch_symbols_traded = sorted(challenger_trades["symbol"].unique().tolist()) if n_ch_trades else []
    final_gap = float(variant["equity"].iloc[-1] - core_variant["equity"].iloc[-1])
    # 逐日权重差异天数
    w_v = variant["weights"]
    w_c = core_variant["weights"].reindex(columns=w_v.columns, fill_value=0.0)
    common = w_v.index.intersection(w_c.index)
    diff_days = int((~np.isclose(
        w_v.loc[common].fillna(0).to_numpy(),
        w_c.loc[common].fillna(0).to_numpy(), atol=1e-9
    ).all(axis=1)).sum())
    return {
        "challenger_trade_count": n_ch_trades,
        "challenger_symbols_traded": ch_symbols_traded,
        "final_equity_gap_vs_core45": final_gap,
        "signal_diff_days_vs_core45": diff_days,
    }


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    formal = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    all_symbols = universe_keys(market)
    core_size = int(formal["enhanced_selection"]["universe_architecture"]["core_pool_size"])
    core_symbols = all_symbols[:core_size]
    challengers = [str(s) for s in formal["enhanced_selection"]["universe_architecture"]["challenger_symbols"]]
    categories_all = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment_all, available = load_sentiment_matrices(FEATURES, panel["close"].index, all_symbols)

    def build(key, symbols, mode, chs, **kw):
        return run_variant(
            key, market, formal, panel, all_symbols, core_symbols,
            categories_all, sentiment_all, available,
            symbols=symbols, mode=mode, challengers=chs, **kw,
        )

    variants = {}
    variants["baseline_core_45"] = build("baseline_core_45", core_symbols, "core_anchor_challenger", [])
    variants["global_rank_51"] = build("global_rank_51", all_symbols, "fixed_pool", [])
    variants["anchor_equal_51"] = build("anchor_equal_51", all_symbols, "core_anchor_challenger", challengers)
    variants["champion_cash_gap_51"] = build("champion_cash_gap_51", all_symbols, "core_champion_cash_gap", challengers)
    variants["qualified_pool_51"] = build("qualified_pool_51", all_symbols, "fixed_pool", [], qualified_first=True)
    variants["theme_champion_51"] = build("theme_champion_51", all_symbols, "fixed_pool", [], theme_champion=True)
    variants["challenger_dominance_51"] = build(
        "challenger_dominance_51", all_symbols, "core_anchor_challenger", challengers,
        challenger_dominance=True,
    )

    core = variants["baseline_core_45"]

    rows = []
    incremental = {}
    for key, v in variants.items():
        m = v["metrics"]
        row = {"variant": key, **m}
        for name, pr in v["period_returns"].items():
            row[f"ret_{name}"] = pr["total_return"]
            row[f"mdd_{name}"] = pr["max_drawdown"]
        row["qualified_pool_size_mean"] = v["qualified_pool_size_mean"]
        row["qualified_pool_size_max"] = v["qualified_pool_size_max"]
        rows.append(row)
        if key != "baseline_core_45":
            incremental[key] = analyse_incremental_trades(v, core, challengers)

    metrics = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")

    incr_rows = []
    for key, info in incremental.items():
        incr_rows.append({"variant": key, **{k: v for k, v in info.items() if k != "challenger_symbols_traded"},
                          "challenger_symbols_traded": "|".join(info["challenger_symbols_traded"])})
    pd.DataFrame(incr_rows).to_csv(OUTPUT / "incremental_symbols.csv", index=False, encoding="utf-8-sig")

    payload = {
        "status": "research_evidence_v2",
        "generated_through": str(market["project"]["data_end"]),
        "same_data_cost_and_execution": True,
        "challengers": challengers,
        "core_pool_size": core_size,
        "variants": clean(metrics.to_dict(orient="records")),
        "incremental": clean(incremental),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 简短控制台摘要
    print(f"{'variant':<26}{'total':>9}{'cagr':>8}{'mdd':>8}{'trades':>8}{'ret2025':>9}")
    for r in rows:
        print(f"{r['variant']:<26}{r['total_return']:>9.3f}{r['cagr']:>8.3f}"
              f"{r['max_drawdown']:>8.3f}{r['trade_count']:>8.0f}{r.get('ret_2025—2026', float('nan')):>9.3f}")
    print("\nincremental vs baseline_core_45:")
    for key, info in incremental.items():
        print(f"  {key:<26} ch_trades={info['challenger_trade_count']} "
              f"gap={info['final_equity_gap_vs_core45']:.0f} "
              f"diff_days={info['signal_diff_days_vs_core45']} "
              f"traded={info['challenger_symbols_traded']}")

    # 返回给稳健性脚本复用
    return variants


if __name__ == "__main__":
    main()
