"""Audit whether the frozen 45-ETF universe is membership-sensitive.

This is a research-only diagnostic.  It never mutates the formal strategy,
account state, daily report, orders, or production configuration.

The study deliberately does not choose a pool by backtest return.  It asks:

1. Does the original 45 sit unusually high among equally sized 45-of-51 pools?
2. How much does removing, adding, or same-category swapping one member change
   the path and terminal result?
3. Does a pre-declared cardinality-scaled rank limit (5/45 selection intensity)
   reduce the mechanical damage from genuine breadth additions?

Run with:
  PYTHONPATH=.:src:experiments python3 experiments/universe_membership_overfit_audit.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.etfwin import EtfwinFeatures, etfwin_signals
from etf_rotation.execution import entry_eligibility, execution_project, period_metrics
from etf_rotation.sentiment import broadcast, load_sentiment_matrices
from etf_rotation.ye import _rules, category_breadth
from universe_architecture_v2_study import FEATURES, cash_management, premium_sensitive
from universe_architecture_v3_conviction import build_components


OUTPUT = ROOT / "results" / "research" / "universe_membership_overfit_v1"
START = "2018-07-02"
END = "2026-07-28"
RANDOM_SEED = 20260728
RANDOM_POOL_COUNT = 96
BASE_SIZE = 45
BASE_LIMITS = {"normal": 5, "fallback": 3, "emerging": 15}
PERIODS = {
    "2018_2020": (START, "2020-12-31"),
    "2021_2022": ("2021-01-01", "2022-12-31"),
    "2023_2024": ("2023-01-01", "2024-12-31"),
    "2024_2026": ("2024-01-01", END),
    "2025_2026": ("2025-01-01", END),
}


CTX: dict[str, object] = {}


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


def scaled_limits(size: int) -> dict[str, int]:
    """Preserve the original rank-selection intensity without fitting returns."""

    return {
        key: int(math.ceil(size * value / BASE_SIZE))
        for key, value in BASE_LIMITS.items()
    }


def slice_features(features: EtfwinFeatures, symbols: list[str]) -> EtfwinFeatures:
    return EtfwinFeatures(
        **{
            field: getattr(features, field)[symbols]
            for field in features.__dataclass_fields__
        }
    )


def build_paths(
    components: dict,
    eligibility: pd.DataFrame,
    formal: dict,
    sentiment: dict[str, pd.DataFrame],
    available_series: pd.Series,
    categories: dict[str, str],
    limits: dict[str, int],
) -> dict[str, pd.DataFrame]:
    """Rebuild the formal fixed-pool paths using cached price features."""

    features = components["features"]
    r2 = components["r2"]
    efficiency = components["efficiency"]
    roc5 = components["roc5"]
    symbols = list(eligibility.columns)
    available = broadcast(available_series, symbols)
    values = formal["rules"]
    enhanced = formal["enhanced_selection"]
    live = enhanced["sentiment_available"]
    fallback_cfg = enhanced["price_only_for_historical_dates_without_sentiment"]
    rank = features.ranking_score.rank(axis=1, ascending=False, method="min")
    breadth = category_breadth(features.roc_short, categories)

    normal_price = (
        eligibility
        & features.roc_short.gt(0.0)
        & features.roc_medium.gt(0.0)
        & features.above_ma
        & features.ma_bias.le(float(values["max_entry_ma_bias"]))
        & rank.le(int(limits["normal"]))
    )
    fallback = (
        normal_price
        & rank.le(int(limits["fallback"]))
        & breadth.ge(float(fallback_cfg["category_roc20_positive_breadth_min"]))
    )
    weak_edge = rank.gt(int(limits["fallback"])) & features.roc_short.lt(0.02)
    edge_confirm = (
        sentiment["matched_count"].ge(3)
        & sentiment["count_acceleration"].ge(0.0)
        & sentiment["positive_dde_share"].ge(0.50)
    )
    current_normal = normal_price & (~weak_edge | edge_confirm)

    emerging_cfg = live["emerging_trend"]
    emerging_trigger = (
        eligibility
        & rank.le(int(limits["emerging"]))
        & features.roc_short.ge(float(emerging_cfg["roc20_min"]))
        & features.roc_medium.ge(float(emerging_cfg["roc60_range"][0]))
        & features.roc_medium.le(float(emerging_cfg["roc60_range"][1]))
        & features.ma_bias.ge(float(emerging_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(emerging_cfg["ma120_bias_range"][1]))
        & r2.ge(float(emerging_cfg["r2_20_min"]))
        & efficiency.ge(float(emerging_cfg["efficiency20_min"]))
        & sentiment["matched_count"].ge(float(emerging_cfg["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(emerging_cfg["hot_score_min"]))
        & sentiment["count_acceleration"].ge(float(emerging_cfg["count_acceleration_min"]))
        & sentiment["positive_dde_share"].ge(float(emerging_cfg["positive_dde_share_min"]))
    )
    emerging = emerging_trigger.rolling(
        int(emerging_cfg["memory_days"]), min_periods=1
    ).max().fillna(False).astype(bool)
    emerging &= (
        features.roc_short.gt(0.0)
        & features.roc_medium.ge(float(emerging_cfg["roc60_range"][0]))
        & features.ma_bias.ge(float(emerging_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(live["quality_extension"]["ma120_bias_range"][1]))
    )

    extension_cfg = live["quality_extension"]
    extension = (
        eligibility
        & rank.le(int(limits["normal"]))
        & features.roc_short.gt(0.0)
        & features.roc_medium.gt(0.0)
        & features.above_ma
        & features.ma_bias.gt(float(extension_cfg["ma120_bias_range"][0]))
        & features.ma_bias.le(float(extension_cfg["ma120_bias_range"][1]))
        & r2.ge(float(extension_cfg["r2_20_min"]))
        & efficiency.ge(float(extension_cfg["efficiency20_min"]))
        & roc5.ge(float(extension_cfg["roc5_min"]))
        & sentiment["matched_count"].ge(float(extension_cfg["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(extension_cfg["hot_score_min"]))
        & sentiment["count_acceleration"].ge(float(extension_cfg["count_acceleration_min"]))
    )
    gate = (~available & fallback) | (available & (current_normal | emerging | extension))
    entry_score = features.ranking_score.where(
        ~emerging, features.roc_short + 0.05 * r2.fillna(0.0)
    )

    missing_exit = fallback_cfg["soft_exit_protection"]
    missing_strong = (
        ~available
        & rank.le(int(limits["fallback"]))
        & features.roc_medium.ge(float(missing_exit["roc60_min"]))
        & features.above_ma
    )
    hot = live["hot_exit_protection"]
    hot_trigger = (
        available
        & sentiment["matched_count"].ge(float(hot["matched_hot_stocks_min"]))
        & sentiment["hot_score"].ge(float(hot["hot_score_min"]))
        & sentiment["positive_dde_share"].ge(float(hot["positive_dde_share_min"]))
    )
    hot_memory = hot_trigger.rolling(
        int(hot["memory_days"]), min_periods=1
    ).max().fillna(False).astype(bool)
    soft = (~missing_strong & ~hot_memory).astype(bool)
    decline = (
        rank.gt(rank.shift(int(values["rank_change_short_days"])))
        & rank.gt(rank.shift(int(values["rank_change_long_days"])))
    ).fillna(False)
    return {
        "gate": gate.astype(bool),
        "entry_score": entry_score,
        "soft": soft,
        "decline": decline,
        "rank": rank,
    }


def initialise_worker(root_text: str) -> None:
    root = Path(root_text)
    market = load_yaml(root / "config" / "market.yaml")
    formal = load_yaml(root / "config" / "ye_strategy.yaml")
    panel = load_panel(market, root / "market_data" / "prices")
    symbols = universe_keys(market)
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment, available = load_sentiment_matrices(
        root / "market_data" / "sentiment" / "features" / "symbol_daily.csv",
        panel["close"].index,
        symbols,
    )
    components = build_components(panel, symbols, formal)
    eligibility, _, _ = entry_eligibility(panel, symbols, formal["rules"])
    CTX.update(
        root=root,
        market=market,
        formal=formal,
        panel=panel,
        all_symbols=symbols,
        categories=categories,
        sentiment=sentiment,
        available=available,
        components=components,
        eligibility=eligibility,
    )
    reference = compute_variant(
        {
            "variant": "worker_reference_core45",
            "family": "reference",
            "symbols": symbols[:BASE_SIZE],
            "limits": BASE_LIMITS,
        },
        compare_reference=False,
    )
    CTX["reference_active"] = reference.pop("_active")


def compute_variant(task: dict, *, compare_reference: bool = True) -> dict:
    market = CTX["market"]
    formal = CTX["formal"]
    panel = CTX["panel"]
    categories_all = CTX["categories"]
    sentiment_all = CTX["sentiment"]
    available = CTX["available"]
    components_all = CTX["components"]
    eligibility_all = CTX["eligibility"]
    symbols = list(task["symbols"])
    limits = {key: int(value) for key, value in task["limits"].items()}
    features = slice_features(components_all["features"], symbols)
    components = {
        "features": features,
        "r2": components_all["r2"][symbols],
        "efficiency": components_all["efficiency"][symbols],
        "roc5": components_all["roc5"][symbols],
    }
    eligibility = eligibility_all[symbols]
    sentiment = {name: frame[symbols] for name, frame in sentiment_all.items()}
    categories = {symbol: categories_all[symbol] for symbol in symbols}
    paths = build_paths(
        components, eligibility, formal, sentiment, available, categories, limits
    )
    bundle, _ = etfwin_signals(
        panel["close"][symbols],
        symbols,
        _rules(formal["rules"]),
        entry_eligibility=eligibility,
        entry_gate=paths["gate"],
        entry_ranking_score_override=paths["entry_score"],
        soft_exit_confirmation=paths["soft"],
        dual_rank_decline_override=paths["decline"],
        reentry_cooldown_days=int(formal["enhanced_selection"]["reentry_cooldown_days"]),
    )
    project = execution_project(
        market,
        premium_sensitive(symbols),
        eligibility.shift(1, fill_value=False).astype(bool),
    )
    result = run_backtest(
        str(task["variant"]),
        panel,
        bundle.weights,
        START,
        END,
        project,
        cash_management=cash_management(formal),
    )
    capital = float(market["project"]["initial_capital"])
    active = bundle.weights.loc[START:].idxmax(axis=1).where(
        bundle.weights.loc[START:].max(axis=1).gt(0.0), "CASH"
    )
    output = {
        "variant": str(task["variant"]),
        "family": str(task["family"]),
        "pool_size": len(symbols),
        "normal_limit": limits["normal"],
        "fallback_limit": limits["fallback"],
        "emerging_limit": limits["emerging"],
        "symbols_hash": hashlib.sha256("|".join(symbols).encode()).hexdigest(),
        "symbols": "|".join(symbols),
        **result.metrics,
        "candidate_day_rate": float(paths["gate"].loc[START:].any(axis=1).mean()),
    }
    for name, (start, end) in PERIODS.items():
        values = period_metrics(result.equity, start, end, capital)
        output[f"return_{name}"] = values["total_return"]
        output[f"mdd_{name}"] = values["max_drawdown"]
    if compare_reference:
        reference = CTX["reference_active"]
        common = active.index.intersection(reference.index)
        output["path_diff_days_vs_core45"] = int(
            (active.loc[common] != reference.loc[common]).sum()
        )
        output["target_agreement_vs_core45"] = float(
            (active.loc[common] == reference.loc[common]).mean()
        )
    output["_active"] = active
    return clean(output)


def run_task(task: dict) -> dict:
    output = compute_variant(task)
    output.pop("_active", None)
    return output


def make_tasks(all_symbols: list[str], categories: dict[str, str]) -> list[dict]:
    core = all_symbols[:BASE_SIZE]
    challengers = all_symbols[BASE_SIZE:]
    tasks: list[dict] = [
        {
            "variant": "core45_fixed_limits",
            "family": "benchmark",
            "symbols": core,
            "limits": BASE_LIMITS,
        },
        {
            "variant": "global51_fixed_limits",
            "family": "benchmark",
            "symbols": all_symbols,
            "limits": BASE_LIMITS,
        },
        {
            "variant": "global51_scaled_limits",
            "family": "candidate",
            "symbols": all_symbols,
            "limits": scaled_limits(len(all_symbols)),
        },
    ]
    governed = [symbol for symbol in all_symbols if symbol not in {"561380.SH", "561360.SH"}]
    tasks.append(
        {
            "variant": "governed49_scaled_limits_diagnostic",
            "family": "candidate",
            "symbols": governed,
            "limits": scaled_limits(len(governed)),
        }
    )
    for symbol in core:
        members = [item for item in core if item != symbol]
        tasks.append(
            {
                "variant": f"leave_one__{symbol}",
                "family": "leave_one_core",
                "symbols": members,
                "limits": BASE_LIMITS,
            }
        )
    for challenger in challengers:
        members = core + [challenger]
        tasks.extend(
            [
                {
                    "variant": f"add_one_fixed__{challenger}",
                    "family": "add_one_fixed",
                    "symbols": members,
                    "limits": BASE_LIMITS,
                },
                {
                    "variant": f"add_one_scaled__{challenger}",
                    "family": "add_one_scaled",
                    "symbols": members,
                    "limits": scaled_limits(len(members)),
                },
            ]
        )
        peers = [symbol for symbol in core if categories[symbol] == categories[challenger]]
        if not peers:
            # A genuinely new category is swapped against one representative
            # from every existing category, never chosen by return.
            peers = []
            seen_categories: set[str] = set()
            for symbol in core:
                category = categories[symbol]
                if category not in seen_categories:
                    peers.append(symbol)
                    seen_categories.add(category)
        for removed in peers:
            members = [symbol for symbol in core if symbol != removed] + [challenger]
            members.sort(key=all_symbols.index)
            tasks.append(
                {
                    "variant": f"swap__{removed}__for__{challenger}",
                    "family": "same_exposure_or_category_swap",
                    "symbols": members,
                    "limits": BASE_LIMITS,
                }
            )

    rng = random.Random(RANDOM_SEED)
    seen: set[tuple[str, ...]] = {tuple(core)}
    while len([task for task in tasks if task["family"] == "random_45_of_51"]) < RANDOM_POOL_COUNT:
        chosen = tuple(sorted(rng.sample(all_symbols, BASE_SIZE), key=all_symbols.index))
        if chosen in seen:
            continue
        seen.add(chosen)
        tasks.append(
            {
                "variant": f"random45_{len(seen)-1:03d}",
                "family": "random_45_of_51",
                "symbols": list(chosen),
                "limits": BASE_LIMITS,
            }
        )
    return tasks


def quantiles(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def exposure_overlap_audit() -> pd.DataFrame:
    """Measure whether a challenger is a new exposure or a near duplicate.

    Correlation is diagnostic only.  The final exposure map must also use the
    tracked index and constituent overlap; return correlation alone may not
    define product equivalence.
    """

    market = load_yaml(ROOT / "config" / "market.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    core = symbols[:BASE_SIZE]
    challengers = symbols[BASE_SIZE:]
    names = {symbol_key(item): item["name"] for item in market["universe"]}
    returns = panel["close"][symbols].pct_change(fill_method=None)
    rows = []
    for window in (63, 252, 756):
        corr = returns.tail(window).corr()
        for symbol in challengers:
            values = corr.loc[symbol, core].dropna().sort_values(ascending=False)
            peer = str(values.index[0])
            rows.append(
                {
                    "window_days": window,
                    "symbol": symbol,
                    "name": names[symbol],
                    "nearest_core_symbol": peer,
                    "nearest_core_name": names[peer],
                    "nearest_correlation": float(values.iloc[0]),
                    "median_core_correlation": float(values.median()),
                    "core_peers_corr_ge_0_80": int(values.ge(0.80).sum()),
                    "core_peers_corr_ge_0_60": int(values.ge(0.60).sum()),
                }
            )
    return pd.DataFrame(rows)


def write_report(frame: pd.DataFrame, overlap: pd.DataFrame) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "all_variants.csv", index=False)
    benchmarks = frame[frame["family"].isin(["benchmark", "candidate"])].copy()
    random_rows = frame[frame["family"] == "random_45_of_51"].copy()
    core = benchmarks.loc[benchmarks["variant"] == "core45_fixed_limits"].iloc[0]
    random_percentile = float((random_rows["total_return"] <= core["total_return"]).mean())
    random_exceed_count = int(random_rows["total_return"].gt(core["total_return"]).sum())
    zero_exceedance_upper_95 = (
        float(1.0 - 0.05 ** (1.0 / len(random_rows)))
        if random_exceed_count == 0 else None
    )
    family_rows = []
    for family, group in frame.groupby("family", sort=False):
        family_rows.append(
            {
                "family": family,
                "count": int(len(group)),
                "return_median": float(group["total_return"].median()),
                "return_min": float(group["total_return"].min()),
                "return_max": float(group["total_return"].max()),
                "mdd_median": float(group["max_drawdown"].median()),
                "path_diff_median": float(group["path_diff_days_vs_core45"].median()),
            }
        )
    family_frame = pd.DataFrame(family_rows)
    family_frame.to_csv(OUTPUT / "family_summary.csv", index=False)
    overlap.to_csv(OUTPUT / "exposure_overlap.csv", index=False)
    summary = {
        "status": "research_only_no_formal_change",
        "generated_through": END,
        "random_seed": RANDOM_SEED,
        "random_pool_count": RANDOM_POOL_COUNT,
        "pool_provenance": (
            "The 45 ETFs were built in 2026 from the user mother pool plus coverage, "
            "liquidity and correlation audit, not by deleting individual historical losers. "
            "However, the entire 2018-2026 history was already visible and strategy parameters "
            "were subsequently evaluated on this membership."
        ),
        "core45_total_return": float(core["total_return"]),
        "core45_random_return_percentile": random_percentile,
        "random_pools_exceeding_core45": random_exceed_count,
        "zero_exceedance_probability_upper_95": zero_exceedance_upper_95,
        "random_45_return_distribution": quantiles(random_rows["total_return"]),
        "random_45_max_drawdown_distribution": quantiles(random_rows["max_drawdown"]),
        "benchmarks": benchmarks.to_dict(orient="records"),
        "families": family_rows,
        "exposure_overlap_252d": overlap.loc[
            overlap["window_days"].eq(252)
        ].to_dict(orient="records"),
        "interpretation_contract": [
            "Random pools diagnose membership sensitivity; they are never candidates selected by return.",
            "A high percentile is evidence of pool-specific backtest dependence, not proof of intentional cherry-picking.",
            "The governed49 diagnostic uses current operational audit information and is not a point-in-time historical pool.",
            "No current-survivor-only experiment can remove delisted/merged ETF survivorship bias.",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def pct(value: float) -> str:
        return f"{value*100:+.2f}%"

    bench_lines = []
    for _, row in benchmarks.iterrows():
        bench_lines.append(
            f"| {row['variant']} | {int(row['pool_size'])} | {int(row['normal_limit'])} | "
            f"{pct(row['total_return'])} | {pct(row['cagr'])} | {pct(row['max_drawdown'])} | "
            f"{int(row['path_diff_days_vs_core45'])} |"
        )
    random_q = summary["random_45_return_distribution"]
    leave = frame[frame["family"] == "leave_one_core"]
    add_fixed = frame[frame["family"] == "add_one_fixed"]
    add_scaled = frame[frame["family"] == "add_one_scaled"]
    overlap_lines = []
    for _, row in overlap.loc[overlap["window_days"].eq(252)].iterrows():
        overlap_lines.append(
            f"| {row['name']}（{row['symbol']}） | {row['nearest_core_name']}（{row['nearest_core_symbol']}） | "
            f"{row['nearest_correlation']:.3f} | {row['median_core_correlation']:.3f} |"
        )
    report = f"""# 45只ETF池成员过拟合与广度稳定性审计

研究截止 {END}。本研究只诊断池成员依赖，不修改正式策略、账户、订单或每日入口。

## 结论先行

1. 原45只不是按单只历史收益删选出来的，因此不能直接指控为“故意挑赢家”；但它在全部历史已经可见后构建，随后参数又在这套池上反复验证，**严格意义上仍属于研究内样本**。
2. 45只池对成员极其敏感。随机抽取 {RANDOM_POOL_COUNT} 组同样大小的45/51池后，原45只累计收益位于样本第 {random_percentile*100:.1f} 百分位，0组超过原45；随机池中位数为 {pct(random_q['median'])}，5%—95%区间为 {pct(random_q['p05'])} 至 {pct(random_q['p95'])}。在独立同分布近似下，“随机池超过原45”的概率95%单侧上界约为 {zero_exceedance_upper_95*100:.1f}%。这说明漂亮回测明显依赖特定成员组合，但不证明研究者主观挑选了它。
3. “增加广度后收益下降”不能再被自动解释为新ETF不好。它同时可能表示：旧池恰好保留了有利复利路径，而更广的池暴露了原策略对成员和单仓路径的脆弱性。
4. 解决办法不能是继续按历史收益挑核心名单。应把广度单位从“ETF数量”改为“独立经济暴露”，先用非收益规则选每个暴露的代表产品，再按 `5/45` 比例机械缩放排名名额；新产品、重复产品和新暴露分别处理。

## 核心对照

| 方案 | 池规模 | 常规名额 | 累计收益 | 年化 | 最大回撤 | 与原45路径差异日 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(bench_lines)}

`global51_scaled_limits` 的阈值完全由原 `5/45、3/45、15/45` 比例计算，不使用收益调参；51只对应常规前6、历史回退前4、新趋势前17。`governed49` 仅演示把当前数据质量尚不足的电网和石油排除后的结构，不可当作无偏历史池。

## 成员扰动

- 留一核心（45种）：累计收益中位数 {pct(float(leave['total_return'].median()))}，范围 {pct(float(leave['total_return'].min()))} 至 {pct(float(leave['total_return'].max()))}；路径差异日中位数 {int(leave['path_diff_days_vs_core45'].median())}。
- 单加挑战者、仍固定前5（6种）：累计收益中位数 {pct(float(add_fixed['total_return'].median()))}。
- 单加挑战者、按规模改为前6（6种）：累计收益中位数 {pct(float(add_scaled['total_return'].median()))}。

这些数字只说明路径敏感度；不得从最高的一行反向挑池。

## 近252日暴露重叠诊断

| 挑战者 | 最相近核心ETF | 相关系数 | 对全部核心相关性中位数 |
|---|---|---:|---:|
{chr(10).join(overlap_lines)}

传媒ETF与游戏ETF近252日相关系数高达 {float(overlap.loc[(overlap['window_days'].eq(252)) & (overlap['symbol'].eq('512980.SH')), 'nearest_correlation'].iloc[0]):.3f}，不应未经指数与成分重叠核实就机械视作一个全新广度单位。相关性只用于发现疑似重复，最终仍需跟踪指数、成分重叠与费率元数据确认。

## 更合理的长期结构

### 1. 广度单位改为“独立经济暴露”

- 同一指数或高度重叠暴露的多只ETF只算一个名额；代表产品按费率、跟踪误差、流动性、数据完整性决定，不看回测收益。
- 新增同暴露产品只可能替换代表产品，不扩大排名尺，也不能挤压别的题材。
- 真正新增的独立题材才增加暴露数，并按原选择强度机械增加名额：`K = ceil(暴露数 × 5 / 45)`。

### 2. 排名仍在完整暴露池进行

不先用双ROC筛到只剩十来只再取前5；那会把“最强11%”变成“合格池前45%”。先在所有已准入暴露代表中排名，再套双ROC、MA120、乖离和AI路径，保留横截面强度语义。

### 3. 池版本只在半年审计日变化

准入只看上市长度、流动性稳定、缺失率、费率、跟踪误差与暴露缺口；不看加入后的收益。新版本冻结半年，日常运行不增加人工步骤。

### 4. 不再用旧回测收益作为必须守住的目标

如果一个按事前规则构建的更广池历史收益低于旧45，只能如实接受。要求“扩池后收益不能下降”本身会迫使研究回到挑名单和调阈值，形成新的过拟合。

### 5. 前瞻双轨，而不是立即替换

正式账户暂时维持已冻结路径；同时从下一信号日起记录“经济暴露池＋比例名额”的影子计划，每月比较，至少积累252个新交易日。影子层完全自动，不增加每日人工资讯审核。

## 仍无法由本研究消除的偏差

- 仓库只有当前存续产品，没有历史清盘、合并ETF，仍有幸存者偏差。
- 2018—2023与2024以后使用不同资讯制度。
- 2018—2026已经被项目反复观察，任何历史结果都不能再称真正样本外。
"""
    (OUTPUT / "summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    formal = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    all_symbols = universe_keys(market)
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    tasks = make_tasks(all_symbols, categories)
    workers = min(4, max(1, mp.cpu_count() - 1))
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=initialise_worker,
        initargs=(str(ROOT),),
    ) as pool:
        rows = list(pool.imap_unordered(run_task, tasks, chunksize=1))
    frame = pd.DataFrame(rows).sort_values(["family", "variant"]).reset_index(drop=True)
    overlap = exposure_overlap_audit()
    write_report(frame, overlap)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "variants": len(frame),
                "families": frame["family"].value_counts().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
