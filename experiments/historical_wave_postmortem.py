from __future__ import annotations

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
from etf_rotation.etfwin import etfwin_signals
from etf_rotation.execution import execution_project, period_metrics
from etf_rotation.sentiment import broadcast, load_sentiment_matrices
from etf_rotation.ye import _rules, build_ye_signals


OUTPUT = ROOT / "results" / "research" / "historical_wave_postmortem"
FEATURES = ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv"
WAVES = {
    "2021新能源": {
        "start": "2021-03-01",
        "end": "2021-12-31",
        "proxies": ["515790.SH", "159755.SZ"],
        "finding": (
            "光伏在2021-06-21至22日满足前5和9%乖离，但只排第5/4，未满足历史回退的前3；"
            "次日升至第3时乖离已达9.8%。电池ETF直到2021-12-17才满120个交易日，已晚于11月高点。"
        ),
    },
    "2021元宇宙": {
        "start": "2021-09-01",
        "end": "2021-12-31",
        "proxies": ["159869.SZ", "159819.SZ", "515050.SH"],
        "finding": (
            "游戏ETF在最强阶段排名第1，但MA120乖离约12%至17%，超过9%；11月26日回落后正式通过，"
            "组合已于11月24日选中酒ETF，单仓且旧仓有效不换仓。"
        ),
    },
    "2022末疫后复苏": {
        "start": "2022-10-01",
        "end": "2023-02-28",
        "proxies": ["159766.SZ", "510630.SH", "512690.SH"],
        "finding": (
            "旅游ETF从2022-11-03起已多日满足常规价格条件，但消费农业大类宽度多数只有20%至40%，"
            "低于历史回退要求的75%；唯一正式通过日是2023-02-27，已在主升浪之后且当时持有酒ETF。"
        ),
    },
    "2023 AI算力+CPO": {
        "start": "2023-01-01",
        "end": "2023-06-30",
        "proxies": ["159819.SZ", "515050.SH", "512760.SH"],
        "finding": (
            "人工智能和通信ETF在主升阶段多数时间MA120乖离超过9%；少数回落到9%以内的日期通常只排第4/5，"
            "未满足历史回退的前3。策略在2023-03-09至04-27基本空仓，因此核心原因不是仓位占用。"
        ),
    },
    "2023中特估": {
        "start": "2022-11-01",
        "end": "2023-05-31",
        "proxies": ["516970.SH", "510880.SH", "512040.SH", "159611.SZ"],
        "finding": (
            "基建ETF在2022-12-05正式通过，但当时持有港股医药且未触发退出；红利ETF到2023-04-28才升至前3并通过，"
            "五一假期后开盘成交时已接近5月8日阶段高点，最终该笔约亏3.25%。"
        ),
    },
}


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def first_true(mask: pd.Series) -> str | None:
    dates = mask.index[mask.fillna(False)]
    return str(pd.Timestamp(dates[0]).date()) if len(dates) else None


def main() -> None:
    market = yaml.safe_load((ROOT / "config" / "market.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "config" / "ye_strategy.yaml").read_text(encoding="utf-8"))
    governance = yaml.safe_load(
        (ROOT / "config" / "strategy_governance.yaml").read_text(encoding="utf-8")
    )
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    names = {symbol_key(item): item["name"] for item in market["universe"]}
    categories = {symbol_key(item): item["category"] for item in market["universe"]}
    sentiment, sentiment_available = load_sentiment_matrices(
        FEATURES, panel["close"].index, symbols
    )
    formal, features, eligibility, _, trailing_amount, decision = build_ye_signals(
        panel, symbols, categories, config, sentiment, sentiment_available
    )
    liquidity_only_eligibility = trailing_amount.ge(
        float(config["rules"]["minimum_entry_amount"])
    ).fillna(False)

    wave_rows = []
    for title, definition in WAVES.items():
        index = panel["close"].loc[definition["start"] : definition["end"]].index
        selected = formal.diagnostics.loc[index, "selected"].fillna("")
        changes = selected[selected.ne(selected.shift())]
        proxies = []
        for symbol in definition["proxies"]:
            prices = panel["close"].loc[index, symbol].dropna()
            proxy_index = prices.index
            normal = decision["normal"].loc[proxy_index, symbol] & eligibility.loc[proxy_index, symbol]
            fallback = decision["fallback"].loc[proxy_index, symbol] & eligibility.loc[proxy_index, symbol]
            focus = eligibility.loc[proxy_index, symbol] & features.roc_short.loc[proxy_index, symbol].gt(0)
            proxies.append({
                "symbol": symbol,
                "name": names[symbol],
                "category": categories[symbol],
                "window_return": float(prices.iloc[-1] / prices.iloc[0] - 1.0),
                "start_to_peak_return": float(prices.max() / prices.iloc[0] - 1.0),
                "peak_date": prices.idxmax(),
                "first_entry_eligible_date": first_true(eligibility.loc[proxy_index, symbol]),
                "best_rank": float(features.rank.loc[proxy_index, symbol].min()),
                "normal_days": int(normal.sum()),
                "formal_fallback_days": int(fallback.sum()),
                "first_normal_date": first_true(normal),
                "first_formal_fallback_date": first_true(fallback),
                "held_days": int(formal.weights.loc[proxy_index, symbol].gt(0).sum()),
                "blocker_day_counts": {
                    "roc60_nonpositive": int((focus & features.roc_medium.loc[proxy_index, symbol].le(0)).sum()),
                    "below_ma120": int((focus & ~features.above_ma.loc[proxy_index, symbol]).sum()),
                    "ma120_bias_over_9pct": int((focus & features.ma_bias.loc[proxy_index, symbol].gt(0.09)).sum()),
                    "rank_worse_than_3": int((focus & features.rank.loc[proxy_index, symbol].gt(3)).sum()),
                    "category_breadth_below_75pct": int((focus & decision["category_breadth"].loc[proxy_index, symbol].lt(0.75)).sum()),
                },
            })
        wave_rows.append({
            "wave": title,
            "start": definition["start"],
            "end": definition["end"],
            "finding": definition["finding"],
            "portfolio_changes": [
                {"date": str(pd.Timestamp(date).date()), "symbol": symbol or None, "name": names.get(symbol)}
                for date, symbol in changes.items()
            ],
            "proxies": proxies,
        })

    available = broadcast(sentiment_available, symbols)
    live_gate = decision["current_normal"] | decision["emerging"] | decision["quality_extension"]
    base = (
        features.roc_short.gt(0)
        & features.roc_medium.gt(0)
        & features.above_ma
        & features.ma_bias.le(0.09)
    )
    base_12 = (
        features.roc_short.gt(0)
        & features.roc_medium.gt(0)
        & features.above_ma
        & features.ma_bias.le(0.12)
    )
    variants = {
        "formal": {
            "fallback": decision["fallback"],
            "eligibility": eligibility,
        },
        "remove_breadth_keep_top3": {
            "fallback": base & features.rank.le(3),
            "eligibility": eligibility,
        },
        "remove_breadth_use_top5": {
            "fallback": base & features.rank.le(5),
            "eligibility": eligibility,
        },
        "remove_breadth_top5_bias12": {
            "fallback": base_12 & features.rank.le(5),
            "eligibility": eligibility,
        },
        "remove_listing_and_breadth_top3": {
            "fallback": base & features.rank.le(3),
            "eligibility": liquidity_only_eligibility,
        },
        "remove_listing_and_breadth_top5": {
            "fallback": base & features.rank.le(5),
            "eligibility": liquidity_only_eligibility,
        },
    }
    premium_sensitive = [
        symbol for symbol in symbols
        if symbol.split(".")[0].startswith("513") or symbol == "159941.SZ"
    ]
    cash = config["cash_management"]
    cash_management = {
        "annual_rate": float(cash["historical_backtest_annual_rate"]),
        "fee_rate": float(cash["fee_rate"]),
        "minimum_order": float(cash["minimum_order"]),
        "order_lot": float(cash["order_lot"]),
    }
    counterfactuals = []
    for variant, definition in variants.items():
        variant_eligibility = definition["eligibility"]
        project = execution_project(
            market,
            premium_sensitive,
            variant_eligibility.shift(1, fill_value=False).astype(bool),
        )
        if variant == "formal":
            bundle = formal
        else:
            entry_gate = (~available & definition["fallback"]) | (available & live_gate)
            bundle, _ = etfwin_signals(
                panel["close"][symbols],
                symbols,
                _rules(config["rules"]),
                entry_eligibility=variant_eligibility,
                entry_gate=entry_gate,
                entry_ranking_score_override=decision["entry_score"],
                soft_exit_confirmation=decision["soft_exit_confirmation"],
                reentry_cooldown_days=int(config["enhanced_selection"]["reentry_cooldown_days"]),
            )
        result = run_backtest(
            variant,
            panel,
            bundle.weights,
            str(market["project"]["backtest_start"]),
            str(market["project"]["data_end"]),
            project,
            cash_management=cash_management,
        )
        years = {
            str(year): period_metrics(
                result.equity, f"{year}-01-01", f"{year}-12-31", project["initial_capital"]
            )["total_return"]
            for year in (2021, 2022, 2023)
        }
        counterfactuals.append({
            "variant": variant,
            "full_total_return": result.metrics["total_return"],
            "full_cagr": result.metrics["cagr"],
            "full_max_drawdown": result.metrics["max_drawdown"],
            "return_2021_2023": period_metrics(
                result.equity, "2021-01-01", "2023-12-31", project["initial_capital"]
            )["total_return"],
            "annual_returns": years,
            "fill_count": int(len(result.trades)),
        })

    payload = clean({
        "status": "research_only",
        "formal_strategy_id": governance["formal_strategy"]["id"],
        "data_through": market["project"]["data_end"],
        "daily_execution_impact": "none",
        "historical_regime": (
            "2024年前没有情绪数据，使用排名前3、类别ROC20正向宽度至少75%的价格回退；"
            "当前前5、9%-12%质量延伸和新趋势例外并未追溯应用。"
        ),
        "waves": wave_rows,
        "counterfactuals": counterfactuals,
        "conclusion": (
            "五段主升浪主要被前3排名、75%类别宽度、9%乖离、MA120所需数据期和单仓不换仓的交集挡住。"
            "但单独删除上市满120日并不会提前入场，因为MA120本身仍需120根有效日线；"
            "简单放宽宽度、排名或乖离会增加大量错误交易并显著扩大回撤，不能因漏掉几段已知行情直接修改正式参数。"
        ),
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cf = {row["variant"]: row for row in payload["counterfactuals"]}
    lines = [
        "# 2021—2023五段主升浪信号尸检",
        "",
        f"- 正式策略：`{payload['formal_strategy_id']}`",
        f"- 数据截止：{payload['data_through']}",
        f"- 历史制度：{payload['historical_regime']}",
        "",
        "## 逐段结论",
        "",
    ]
    for row in payload["waves"]:
        lines.extend([f"### {row['wave']}", "", row["finding"], ""])
        lines.extend([
            "| 代理ETF | 区间涨跌 | 起点至峰值 | 最佳排名 | 常规合格日 | 历史回退合格日 | 实际持有日 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for proxy in row["proxies"]:
            lines.append(
                f"| {proxy['name']} | {proxy['window_return']:.2%} | {proxy['start_to_peak_return']:.2%} | "
                f"{proxy['best_rank']:.0f} | {proxy['normal_days']} | {proxy['formal_fallback_days']} | {proxy['held_days']} |"
            )
        lines.append("")
    lines.extend([
        "## 简单放宽的反事实",
        "",
        "| 版本 | 2021—2023收益 | 全区间年化 | 最大回撤 | 成交笔数 |",
        "|---|---:|---:|---:|---:|",
    ])
    labels = {
        "formal": "正式策略",
        "remove_breadth_keep_top3": "去宽度、保留前3",
        "remove_breadth_use_top5": "去宽度、放到前5",
        "remove_breadth_top5_bias12": "去宽度、前5、乖离12%",
        "remove_listing_and_breadth_top3": "去120日、去宽度、保留前3",
        "remove_listing_and_breadth_top5": "去120日、去宽度、放到前5",
    }
    for key in labels:
        row = cf[key]
        lines.append(
            f"| {labels[key]} | {row['return_2021_2023']:.2%} | {row['full_cagr']:.2%} | "
            f"{row['full_max_drawdown']:.2%} | {row['fill_count']} |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        payload["conclusion"],
        "",
        "本报告只做隔离研究，不修改正式信号、参数、日报或实盘订单。",
    ])
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
