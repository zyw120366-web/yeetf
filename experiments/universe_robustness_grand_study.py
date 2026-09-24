"""ETF 池鲁棒性终审：单赢家与主策略/扩展策略资金架构。

这不是参数寻优。动量、MA120、AI 确认、退出、成本与 T+1 执行全部冻结。
唯一新增的结构候选预先固定为：

    80% 原45只主策略资金 + 20% 锚定平权广池扩展策略资金

两部分各自只持有一只 ETF 或现金；当两者选中同一 ETF 时合并为一个仓位。
20% 是风险预算，不按历史收益选择。10%/30% 只作邻近压力测试，不用于选优。

为回答“扩大 ETF 池为何会严重改变收益”，脚本遍历六只挑战者的全部 64 种
成员组合，并分别计算：

1. 100% 平权广池的结果；
2. 80/20 两部分资金方案的结果；
3. 8 段 CSCV/PBO 诊断，衡量事后挑池的过拟合风险。

用法：
    PYTHONPATH=src python3 experiments/universe_robustness_grand_study.py
"""

from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from universe_architecture_v2_study import (  # noqa: E402
    cash_management,
    clean,
    load_yaml,
    premium_sensitive,
    run_variant,
)
from etf_rotation.backtest import run_backtest  # noqa: E402
from etf_rotation.data import load_panel, symbol_key, universe_keys  # noqa: E402
from etf_rotation.execution import execution_project, period_metrics  # noqa: E402
from etf_rotation.sentiment import load_sentiment_matrices  # noqa: E402


FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
OUTPUT = ROOT / "results" / "research" / "universe_robustness_grand_study"
CORE_SHARE = 0.80
CSCV_BLOCKS = 8
PERIODS = {
    "2018_2020": ("2018-07-02", "2020-12-31"),
    "2021_2022": ("2021-01-01", "2022-12-31"),
    "2023_2024": ("2023-01-01", "2024-12-31"),
    "2025_2026": ("2025-01-01", "2026-07-29"),
}


def align_weights(weights: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    return weights.reindex(columns=symbols, fill_value=0.0).fillna(0.0)


def blend_weights(
    core_weights: pd.DataFrame,
    broad_weights: pd.DataFrame,
    symbols: list[str],
    core_share: float,
) -> pd.DataFrame:
    if not 0.0 <= core_share <= 1.0:
        raise ValueError("core_share must be between zero and one")
    blended = (
        align_weights(core_weights, symbols) * core_share
        + align_weights(broad_weights, symbols) * (1.0 - core_share)
    )
    if float(blended.sum(axis=1).max()) > 1.0 + 1e-10:
        raise AssertionError("blended exposure exceeds 100%")
    return blended


def metric_row(name: str, result, initial_capital: float) -> dict:
    row = {"variant": name, **result.metrics}
    for period, (start, end) in PERIODS.items():
        values = period_metrics(result.equity, start, end, initial_capital)
        row[f"return_{period}"] = values["total_return"]
        row[f"mdd_{period}"] = values["max_drawdown"]
    return row


def sharpe(values: pd.Series) -> float:
    values = values.dropna()
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean() / std * np.sqrt(252)) if std > 0 else -np.inf


def cscv_pbo(return_matrix: pd.DataFrame, blocks: int = CSCV_BLOCKS) -> dict:
    """Small, transparent CSCV/PBO diagnostic for a family of equity curves.

    Rows are dates and columns are membership variants. The history is split into
    contiguous blocks. For every half/half split, the in-sample Sharpe champion is
    located in the out-of-sample ranking. PBO is the share of splits in which that
    champion falls below the OOS median.
    """

    if blocks % 2:
        raise ValueError("CSCV block count must be even")
    ordered = return_matrix.dropna(how="all").fillna(0.0)
    locations = np.array_split(np.arange(len(ordered)), blocks)
    records = []
    for chosen in itertools.combinations(range(blocks), blocks // 2):
        in_rows = np.concatenate([locations[i] for i in chosen])
        out_rows = np.concatenate([locations[i] for i in range(blocks) if i not in chosen])
        in_scores = ordered.iloc[in_rows].apply(sharpe)
        winner = str(in_scores.idxmax())
        out_scores = ordered.iloc[out_rows].apply(sharpe)
        percentile = float(out_scores.rank(method="average", pct=True)[winner])
        records.append(
            {
                "is_blocks": "|".join(map(str, chosen)),
                "is_winner": winner,
                "is_sharpe": float(in_scores[winner]),
                "oos_sharpe": float(out_scores[winner]),
                "oos_percentile": percentile,
                "oos_below_median": percentile <= 0.50,
            }
        )
    details = pd.DataFrame(records)
    return {
        "pbo": float(details["oos_below_median"].mean()),
        "median_oos_percentile": float(details["oos_percentile"].median()),
        "split_count": int(len(details)),
        "details": details,
    }


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    formal = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    all_symbols = universe_keys(market)
    core_size = int(
        formal["enhanced_selection"]["universe_architecture"]["core_pool_size"]
    )
    core_symbols = all_symbols[:core_size]
    challengers = [
        str(symbol)
        for symbol in formal["enhanced_selection"]["universe_architecture"][
            "challenger_symbols"
        ]
    ]
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment, available = load_sentiment_matrices(
        FEATURES, panel["close"].index, all_symbols
    )
    start = str(market["project"]["backtest_start"])
    end = str(market["project"]["data_end"])
    capital = float(market["project"]["initial_capital"])

    def build(key: str, subset: tuple[str, ...] | list[str], mode: str):
        subset = list(subset)
        symbols = core_symbols + subset
        return run_variant(
            key,
            market,
            formal,
            panel,
            all_symbols,
            core_symbols,
            categories,
            sentiment,
            available,
            symbols=symbols,
            mode=mode,
            challengers=subset,
        )

    core = build("core45", [], "core_anchor_challenger")
    full_anchor = build("anchor_equal51", challengers, "core_anchor_challenger")
    formal_gap = build("formal_cash_gap51", challengers, "core_champion_cash_gap")
    full_eligibility = full_anchor["eligibility"].reindex(
        columns=all_symbols, fill_value=False
    )
    project = execution_project(
        market,
        premium_sensitive(all_symbols),
        full_eligibility.shift(1, fill_value=False).astype(bool),
    )

    benchmark_rows = []
    for key, variant in (
        ("core45", core),
        ("anchor_equal51", full_anchor),
        ("formal_cash_gap51", formal_gap),
    ):
        benchmark_rows.append(
            {
                "variant": key,
                **variant["metrics"],
                **{
                    f"return_{period}": period_metrics(
                        variant["equity"], pstart, pend, capital
                    )["total_return"]
                    for period, (pstart, pend) in PERIODS.items()
                },
            }
        )

    sensitivity_results = {}
    for broad_share in (0.10, 0.20, 0.30):
        weights = blend_weights(
            core["weights"], full_anchor["weights"], all_symbols, 1.0 - broad_share
        )
        result = run_backtest(
            f"main_expanded_{int(broad_share * 100)}",
            panel,
            weights,
            start,
            end,
            project,
            cash_management=cash_management(formal),
        )
        sensitivity_results[broad_share] = result
        benchmark_rows.append(
            metric_row(f"main_expanded_{int(broad_share * 100)}", result, capital)
        )

    # 更简洁的20%扩展资金：只在扩展策略选中新增ETF时启用；其余日期
    # 100%跟随原45只主策略，避免新增ETF退出后继续沿另一条核心路径分叉。
    full_expanded_is_new = full_anchor["weights"][challengers].sum(axis=1).gt(0.0)
    new_etf_only_full_weights = align_weights(core["weights"], all_symbols)
    new_etf_only_full_weights.loc[full_expanded_is_new] = (
        align_weights(core["weights"], all_symbols).loc[full_expanded_is_new]
        * CORE_SHARE
        + align_weights(full_anchor["weights"], all_symbols).loc[full_expanded_is_new]
        * (1.0 - CORE_SHARE)
    )
    new_etf_only_full = run_backtest(
        "new_etf_only_20",
        panel,
        new_etf_only_full_weights,
        start,
        end,
        project,
        cash_management=cash_management(formal),
    )
    benchmark_rows.append(
        metric_row("new_etf_only_20", new_etf_only_full, capital)
    )

    membership_rows = []
    single_returns = {}
    dual_returns = {}
    new_etf_only_returns = {}
    for count in range(len(challengers) + 1):
        for subset in itertools.combinations(challengers, count):
            label = "none" if not subset else "+".join(symbol.split(".")[0] for symbol in subset)
            broad = core if not subset else build(
                f"anchor_{label}", subset, "core_anchor_challenger"
            )
            dual_weights = blend_weights(
                core["weights"], broad["weights"], all_symbols, CORE_SHARE
            )
            dual = run_backtest(
                f"dual_{label}",
                panel,
                dual_weights,
                start,
                end,
                project,
                cash_management=cash_management(formal),
            )
            subset_new_selected = (
                broad["weights"][list(subset)].sum(axis=1).gt(0.0)
                if subset
                else pd.Series(False, index=broad["weights"].index)
            )
            new_etf_only_weights = align_weights(core["weights"], all_symbols)
            new_etf_only_weights.loc[subset_new_selected] = (
                align_weights(core["weights"], all_symbols).loc[subset_new_selected]
                * CORE_SHARE
                + align_weights(broad["weights"], all_symbols).loc[subset_new_selected]
                * (1.0 - CORE_SHARE)
            )
            new_etf_only = run_backtest(
                f"new_etf_only_{label}",
                panel,
                new_etf_only_weights,
                start,
                end,
                project,
                cash_management=cash_management(formal),
            )
            membership_rows.append(
                {
                    "membership": label,
                    "challenger_count": count,
                    "single_total_return": broad["metrics"]["total_return"],
                    "single_cagr": broad["metrics"]["cagr"],
                    "single_sharpe": broad["metrics"]["sharpe"],
                    "single_max_drawdown": broad["metrics"]["max_drawdown"],
                    "dual_total_return": dual.metrics["total_return"],
                    "dual_cagr": dual.metrics["cagr"],
                    "dual_sharpe": dual.metrics["sharpe"],
                    "dual_max_drawdown": dual.metrics["max_drawdown"],
                    "dual_trade_count": dual.metrics["trade_count"],
                    "new_etf_only_total_return": new_etf_only.metrics["total_return"],
                    "new_etf_only_cagr": new_etf_only.metrics["cagr"],
                    "new_etf_only_sharpe": new_etf_only.metrics["sharpe"],
                    "new_etf_only_max_drawdown": new_etf_only.metrics["max_drawdown"],
                    "new_etf_only_trade_count": new_etf_only.metrics["trade_count"],
                }
            )
            single_returns[label] = broad["equity"].pct_change().fillna(0.0)
            dual_returns[label] = dual.equity.pct_change().fillna(0.0)
            new_etf_only_returns[label] = new_etf_only.equity.pct_change().fillna(0.0)

    membership = pd.DataFrame(membership_rows)
    single_matrix = pd.DataFrame(single_returns)
    dual_matrix = pd.DataFrame(dual_returns)
    new_etf_only_matrix = pd.DataFrame(new_etf_only_returns)
    single_pbo = cscv_pbo(single_matrix)
    dual_pbo = cscv_pbo(dual_matrix)
    new_etf_only_pbo = cscv_pbo(new_etf_only_matrix)

    # 小资金可执行性：沿用相同信号，仅把初始资金改为当前实盘量级。
    small_project = copy.deepcopy(project)
    small_project["initial_capital"] = 12_000.0
    small_rows = []
    small_variants = {
        "core45": align_weights(core["weights"], all_symbols),
        "anchor_equal51": align_weights(full_anchor["weights"], all_symbols),
        "formal_cash_gap51": align_weights(formal_gap["weights"], all_symbols),
        "main_expanded_20": blend_weights(
            core["weights"], full_anchor["weights"], all_symbols, CORE_SHARE
        ),
        "new_etf_only_20": new_etf_only_full_weights,
    }
    for key, weights in small_variants.items():
        result = run_backtest(
            f"small_{key}",
            panel,
            weights,
            start,
            end,
            small_project,
            cash_management=cash_management(formal),
        )
        small_rows.append({"variant": key, **result.metrics})

    def dispersion(prefix: str) -> dict:
        total = membership[f"{prefix}_total_return"]
        mdd = membership[f"{prefix}_max_drawdown"]
        return {
            "return_min": float(total.min()),
            "return_median": float(total.median()),
            "return_max": float(total.max()),
            "return_range": float(total.max() - total.min()),
            "return_std": float(total.std(ddof=1)),
            "mdd_worst": float(mdd.min()),
            "mdd_median": float(mdd.median()),
        }

    summary = {
        "status": "research_only_not_formal_strategy",
        "generated_through": end,
        "pre_registered_candidate": "80%原45只主策略资金 + 20%锚定平权广池扩展策略资金",
        "unchanged_logic": [
            "ROC20 + 1.5*ROC60",
            "MA120 and bias gates",
            "AI/news confirmation",
            "exit rules",
            "T+1 open execution and costs",
        ],
        "membership_variants": int(len(membership)),
        "single_pool_dispersion": dispersion("single"),
        "main_expanded_dispersion": dispersion("dual"),
        "new_etf_only_dispersion": dispersion("new_etf_only"),
        "single_pool_pbo": {k: v for k, v in single_pbo.items() if k != "details"},
        "main_expanded_pbo": {k: v for k, v in dual_pbo.items() if k != "details"},
        "new_etf_only_pbo": {
            k: v for k, v in new_etf_only_pbo.items() if k != "details"
        },
        "max_new_universe_target_weight": 1.0 - CORE_SHARE,
        "decision_rule": (
            "Do not select the historically best membership or broad-share value. "
            "Judge whether the pre-registered 80/20 structure materially compresses "
            "membership sensitivity while preserving acceptable risk/return."
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(benchmark_rows).to_csv(
        OUTPUT / "benchmarks.csv", index=False, encoding="utf-8-sig"
    )
    membership.to_csv(
        OUTPUT / "membership_64.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(small_rows).to_csv(
        OUTPUT / "small_capital_12000.csv", index=False, encoding="utf-8-sig"
    )
    single_pbo["details"].to_csv(
        OUTPUT / "cscv_single.csv", index=False, encoding="utf-8-sig"
    )
    dual_pbo["details"].to_csv(
        OUTPUT / "cscv_dual.csv", index=False, encoding="utf-8-sig"
    )
    new_etf_only_pbo["details"].to_csv(
        OUTPUT / "cscv_new_etf_only.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
    print("\nBenchmarks")
    print(
        pd.DataFrame(benchmark_rows)[
            ["variant", "total_return", "cagr", "max_drawdown", "sharpe"]
        ].to_string(index=False)
    )
    print("\n12,000 capital")
    print(
        pd.DataFrame(small_rows)[
            ["variant", "total_return", "cagr", "max_drawdown", "sharpe"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
