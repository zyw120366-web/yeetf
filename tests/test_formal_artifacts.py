from __future__ import annotations

import html
import json
import importlib.util
from pathlib import Path
import re

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_formal_summary_matches_frozen_validation() -> None:
    config = yaml.safe_load((ROOT / "config" / "ye_strategy.yaml").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "results" / "ye_strategy" / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    validation = config["validation"]
    assert summary["generated_through"] >= validation["backtest_end"]
    if summary["generated_through"] == validation["backtest_end"]:
        assert metrics["total_return"] == pytest.approx(validation["total_return"], abs=1e-10)
        assert metrics["cagr"] == pytest.approx(validation["cagr"], abs=1e-10)
        assert summary["periods"]["2025—2026"]["total_return"] == pytest.approx(
            validation["return_2025_2026"], abs=1e-10
        )
    else:
        assert metrics["total_return"] > -1.0
        assert metrics["cagr"] > -1.0
    assert metrics["max_drawdown"] == pytest.approx(validation["max_drawdown"], abs=1e-10)
    assert summary["timing"]["failed_operation_rate"] == pytest.approx(
        validation["failed_operation_rate"], abs=1e-10
    )
    assert summary["periods"]["2021—2022"]["total_return"] == pytest.approx(
        validation["return_2021_2022"], abs=1e-10
    )


def test_daily_entry_does_not_call_research_scripts() -> None:
    entry = (ROOT / "scripts" / "run_after_close.py").read_text(encoding="utf-8")
    assert "research_" not in entry
    assert 'run("run_strategies.py")' in entry


def test_public_html_contains_only_current_strategy_context() -> None:
    strategy = (ROOT / "dashboard" / "public" / "ye-strategy.html").read_text(encoding="utf-8")
    for stale in ("best_cooldown5", "继续研究方向", "ye v2", "重成本"):
        assert stale not in strategy
    assert "ye 当前完整规则" in strategy
    assert "ye 与 etfwin 规则对照" in strategy


def test_backtest_dashboard_covers_full_pool_with_frozen_scores() -> None:
    script = ROOT / "dashboard" / "scripts" / "build_ye_strategy_html.py"
    spec = importlib.util.spec_from_file_location("build_ye_strategy_html", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dashboard = module.build_etf_dashboard()
    items = dashboard["items"]
    assert len(items) == 45
    assert len({item["symbol"] for item in items}) == 45
    assert all(len(item["series"]) >= min(int(item["listed_sessions"]), 253) for item in items)
    assert sum(item["return_1y"] is not None for item in items) >= 40
    assert all(
        {"rank", "momentum_score", "roc20", "roc60", "ma120_bias", "final_entry_pass"}
        <= item.keys()
        for item in items
    )


def test_backtest_html_exposes_etf_dashboard_controls() -> None:
    backtest = (ROOT / "dashboard" / "public" / "ye-backtest.html").read_text(
        encoding="utf-8"
    )
    for marker in ("全池 ETF 观察台", "近3个月", "近1年", "按策略排名", "etfTrendChart"):
        assert marker in backtest


def test_daily_html_opens_with_plain_language_overview() -> None:
    script = ROOT / "dashboard" / "scripts" / "build_ye_strategy_html.py"
    spec = importlib.util.spec_from_file_location("build_ye_strategy_html_daily", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    daily = module.build_daily_page()
    match = re.search(r'<section class="section" id="dailyOverview">(.*?)</section>', daily, re.S)
    assert match
    overview = html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))
    assert "今日决策路径综述" in overview
    overview_length = len(re.sub(r"\s+", "", overview))
    assert 500 <= overview_length <= 800
