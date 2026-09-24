import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from etf_rotation import daily_report
from etf_rotation.live import atomic_json, publish_blocked, validate_account

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-14"


@pytest.fixture
def report_root(tmp_path):
    for name in ("config", "market_data", "src", "scripts", "skills"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    state = json.loads((ROOT / f"results/audit/{DATE}_live_run_card.json").read_text())["account_state"]
    atomic_json(tmp_path / "results/live/account_state.json", state)
    return tmp_path


def test_pending_cash_still_publishes_full_frozen_analysis(report_root):
    account_path = report_root / "results/live/account_state.json"
    before = account_path.read_bytes()
    publish_blocked(report_root, DATE, "ValueError: 账户状态未确认")
    report = daily_report.read_json(report_root / f"results/live/{DATE}_daily_report.json")
    assert report["analysis"]["status"] == "complete"
    assert len(report["analysis"]["ranking"]) == 51
    assert report["analysis"]["held"]["symbol"] == "159985.SZ"
    assert report["analysis"]["technical_target"] == "159985.SZ"
    assert report["review"]["coverage"] == 1
    assert report["valuation"]["purchase_pnl"] == pytest.approx(-383.5, abs=.01)
    assert report["valuation"]["daily_pnl"] == pytest.approx(-221, abs=.01)
    assert report["valuation"]["account_equity"] is None
    assert report["valuation"]["account_return"] is None
    assert not report["orders"] and not report["target_is_executable"]
    assert account_path.read_bytes() == before
    assert daily_report.validate_delivery(report_root, DATE)["delivery"] == "PASS"
    with pytest.raises(ValueError):
        validate_account(json.loads(before), DATE)


def test_stale_running_message_and_old_buy_are_replaced(report_root, monkeypatch):
    monkeypatch.setattr(daily_report, "analyze", lambda *_: {"status": "unavailable", "ranking": [], "candidates": []})
    atomic_json(report_root / "results/live/readiness_report.json", {"signal_date": DATE, "status": "RUNNING", "blocking_items": ["本次运行尚未完成"]})
    atomic_json(report_root / f"results/live/{DATE}_order_plan.json", {"signal_date": DATE, "actions": [{"side": "buy", "symbol": "513200.SH"}], "execution": {"orders": [{"side": "buy"}]}})
    publish_blocked(report_root, DATE, "资讯缺失")
    plan = daily_report.read_json(report_root / f"results/live/{DATE}_order_plan.json")
    ready = daily_report.read_json(report_root / "results/live/readiness_report.json")
    assert ready["blocking_items"] == ["资讯缺失"]
    assert not plan["actions"] and not plan["execution"]["orders"]
    page = (report_root / "outputs/ETF轮动策略_今日日报.html").read_text()
    assert "本次运行尚未完成" not in page and "今日决策路径综述" in page


def test_missing_evidence_is_partial_not_fallback_buy(report_root, monkeypatch):
    def missing(*_):
        raise ValueError("审核原文哈希不符")
    monkeypatch.setattr(daily_report, "validate_live_review", missing)
    publish_blocked(report_root, DATE, "审核原文哈希不符")
    report = daily_report.read_json(report_root / f"results/live/{DATE}_daily_report.json")
    assert report["analysis"]["status"] == "unavailable"
    assert report["review"]["coverage"] is None
    assert report["analysis"]["candidates"] == []
    assert report["valuation"]["purchase_pnl"] is not None
    assert daily_report.validate_delivery(report_root, DATE)["release"] == "BLOCKED"


def test_missing_everything_still_has_dated_report(tmp_path):
    publish_blocked(tmp_path, DATE, "配置或数据缺失")
    page = (tmp_path / "outputs/ETF轮动策略_今日日报.html").read_text()
    assert DATE in page and "暂无可执行买入计划" in page
    assert daily_report.validate_delivery(tmp_path, DATE)["delivery"] == "PASS"


@pytest.mark.parametrize("damage", ["html", "plan", "link", "status"])
def test_delivery_catches_tampering(report_root, monkeypatch, damage):
    monkeypatch.setattr(daily_report, "analyze", lambda *_: {"status": "unavailable", "ranking": [], "candidates": []})
    publish_blocked(report_root, DATE, "资金待核")
    if damage == "html":
        (report_root / "outputs/ETF轮动策略_今日日报.html").write_text("old report")
    elif damage == "link":
        path = report_root / "outputs/ETF轮动策略_今日日报.html"
        path.write_text(path.read_text().replace("运行卡</a>", "运行卡</a><a href='missing.html'>bad</a>"))
    elif damage == "plan":
        atomic_json(report_root / f"results/live/{DATE}_order_plan.json", {"signal_date": DATE, "actions": [{"side": "buy"}]})
    else:
        atomic_json(report_root / "results/live/readiness_report.json", {"signal_date": DATE, "status": "READY"})
    with pytest.raises(ValueError):
        daily_report.validate_delivery(report_root, DATE)


def test_future_account_cannot_backfill_historical_holdings(report_root):
    account = daily_report.read_json(report_root / "results/live/account_state.json")
    with pytest.raises(ValueError, match="晚于信号日"):
        daily_report.known_position(account, "2026-09-10")


def test_cash_pending_without_actual_fill_is_not_trusted(report_root):
    account = daily_report.read_json(report_root / "results/live/account_state.json")
    account["actual_execution"]["fills"] = []
    with pytest.raises(ValueError, match="持仓事实"):
        daily_report.known_position(account, DATE)


def test_shared_projection_preserves_formal_frozen_ranking():
    state = json.loads((ROOT / "results/live/2026-09-10_order_plan.json").read_text())["account_state"]
    analysis = daily_report.analyze(ROOT, "2026-09-10", state)
    actual = pd.DataFrame(analysis["ranking"]).set_index("symbol")
    expected = pd.read_csv(ROOT / "tests/fixtures/2026-09-10_ranking.csv").set_index("symbol")
    fields = ["rank", "rank_5d_ago", "rank_20d_ago", "momentum_score", "selection_score",
              "technical_entry_pass", "soft_exit_confirmation", "hot_exit_protection"]
    pd.testing.assert_frame_equal(actual[fields].sort_index(), expected[fields].sort_index(), check_dtype=False)


def test_ready_and_sell_only_use_same_report_sections(report_root, monkeypatch):
    monkeypatch.setattr(daily_report, "analyze", lambda *_: {"status": "unavailable", "ranking": [], "candidates": []})
    state = json.loads((ROOT / "results/live/2026-09-10_order_plan.json").read_text())["account_state"]
    atomic_json(report_root / "results/live/account_state.json", state)
    for status, sides in [("READY", ["sell", "buy"]), ("SELL_ONLY", ["sell"])]:
        actions = [{"side": side, "symbol": "513200.SH" if side == "sell" else "159985.SZ"} for side in sides]
        atomic_json(report_root / "results/live/readiness_report.json", {"signal_date": "2026-09-10", "status": status})
        atomic_json(report_root / "results/live/2026-09-10_order_plan.json", {"signal_date": "2026-09-10", "actions": actions, "target_symbol": None if status == "SELL_ONLY" else "159985.SZ", "execution": {"orders": actions}})
        report = daily_report.collect(report_root, "2026-09-10")
        assert report["status"] == status
        page = daily_report.render(report)
        for label in ["今日决策路径综述", "今日筛选漏斗", "今日卫星检查", "今日收益"]:
            assert label in page
        if status == "SELL_ONLY":
            assert all(x["side"] == "sell" for x in report["orders"])


def test_report_text_escapes_untrusted_failure(tmp_path):
    publish_blocked(tmp_path, DATE, "<script>alert(1)</script>")
    page = (tmp_path / "outputs/ETF轮动策略_今日日报.html").read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_receipt_links_are_plain_markdown_and_only_point_to_real_files(report_root, monkeypatch):
    monkeypatch.setattr(daily_report, "analyze", lambda *_: {"status": "unavailable", "ranking": [], "candidates": []})
    page = report_root / "outputs/ETF轮动策略_回测.html"
    page.parent.mkdir(exist_ok=True)
    page.write_text("<html>historical backtest</html>")
    publish_blocked(report_root, DATE, "资金待核")
    before = (report_root / "results/live/account_state.json").read_bytes()
    receipt = daily_report.delivery_receipt(report_root, DATE)
    assert "<heartbeat>" not in receipt and "<message>" not in receipt and "```" not in receipt
    assert "订单：BLOCKED" in receipt and "资金待核" in receipt
    assert "159985.SZ" in receipt and "6500股" in receipt
    assert "目标仓位100%" not in receipt  # Unknown cash cannot become a full-weight order.
    assert "GitHub" not in receipt  # Git synchronization must be checked separately.
    for label, path in [("今日日报", report_root / "outputs/ETF轮动策略_今日日报.html"),
                        ("回测", page),
                        ("运行卡", report_root / f"results/audit/{DATE}_live_run_card.json")]:
        assert f"[{label}](<{path}>)" in receipt and path.is_file()
    assert (report_root / "results/live/account_state.json").read_bytes() == before


def test_receipt_omits_missing_backtest_and_rejects_wrong_day(tmp_path):
    root = tmp_path / "project with spaces"
    publish_blocked(root, DATE, "数据未到齐")
    receipt = daily_report.delivery_receipt(root, DATE)
    assert "回测文件缺失" in receipt and "[回测]" not in receipt
    assert f"[今日日报](<{root}/outputs/ETF轮动策略_今日日报.html>)" in receipt
    with pytest.raises(ValueError, match="日期不一致"):
        daily_report.delivery_receipt(root, "2026-09-15")


@pytest.mark.parametrize("field", ["account", "target_symbol", "orders"])
def test_delivery_rejects_report_plan_fact_mismatch(tmp_path, field):
    publish_blocked(tmp_path, DATE, "测试")
    path = tmp_path / f"results/live/{DATE}_daily_report.json"
    report = daily_report.read_json(path)
    report[field] = {"positions": []} if field == "account" else "159985.SZ" if field == "target_symbol" else [{"side": "buy"}]
    atomic_json(path, report)
    with pytest.raises(ValueError, match="不一致"):
        daily_report.delivery_receipt(tmp_path, DATE)


def test_official_entry_cash_pending_delivers_without_mutating_account(tmp_path):
    # Real entry in a disposable checkout: no market fetch or production writes.
    for name in ("config", "src", "scripts", "skills"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "market_data").symlink_to(ROOT / "market_data", target_is_directory=True)
    account = json.loads((ROOT / f"results/audit/{DATE}_live_run_card.json").read_text())["account_state"]
    account_path = tmp_path / "results/live/account_state.json"
    atomic_json(account_path, account)
    before = account_path.read_bytes()
    market_before = (tmp_path / "config/market.yaml").read_bytes()
    result = subprocess.run(
        [sys.executable, "scripts/run_after_close.py", "--date", DATE, "--skip-collect"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stdout + result.stderr
    assert account_path.read_bytes() == before
    assert (tmp_path / "config/market.yaml").read_bytes() == market_before
    report = daily_report.read_json(tmp_path / f"results/live/{DATE}_daily_report.json")
    assert report["analysis"]["status"] == "complete"
    assert report["analysis"]["held"]["symbol"] == "159985.SZ"
    assert not report["orders"]
    assert daily_report.validate_delivery(tmp_path, DATE)["delivery"] == "PASS"
    receipt = subprocess.run(
        [sys.executable, "scripts/validate_daily_delivery.py", "--date", DATE, "--format", "markdown"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True, text=True, timeout=20)
    assert receipt.returncode == 0, receipt.stderr
    assert "订单：BLOCKED" in receipt.stdout and "[今日日报](<" in receipt.stdout
    assert account_path.read_bytes() == before


def test_ready_pipeline_renders_and_binds_deliverables_in_isolated_checkout(tmp_path):
    # Copy writable inputs; never share the live account or generated data.
    day = "2026-09-10"
    for name in ("config", "src", "scripts", "skills", "market_data", "results", "dashboard"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__", ".live-run.lock"))
    shutil.copyfile(ROOT / "run_strategies.py", tmp_path / "run_strategies.py")
    account = daily_report.read_json(ROOT / "results/audit/2026-09-09_execution_reconciliation.json")["baseline_acceptance"]["account_snapshot"]
    atomic_json(tmp_path / "results/live/account_state.json", account)
    result = subprocess.run(
        [sys.executable, "scripts/run_after_close.py", "--date", day, "--skip-collect"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    delivery = daily_report.validate_delivery(tmp_path, day)
    assert delivery["release"] == "READY" and delivery["analysis"] == "complete"
    report = daily_report.read_json(tmp_path / f"results/live/{day}_daily_report.json")
    assert report["target_is_executable"]
    expected = daily_report.read_json(ROOT / f"results/live/{day}_order_plan.json")
    actual = daily_report.read_json(tmp_path / f"results/live/{day}_order_plan.json")
    assert actual["target_symbol"] == expected["target_symbol"]
    assert actual["actions"] == expected["actions"]
    page = (tmp_path / "outputs/ETF轮动策略_今日日报.html").read_text()
    buy = next(x for x in actual["execution"]["orders"] if x["side"] == "buy")
    assert f"数量参考：{buy['buy_estimate']['estimated_quantity_at_last_close']}" in page
    assert "目标仓位100%" in page and "数量参考：None" not in page
