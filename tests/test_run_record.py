from pathlib import Path

import pytest

from etf_rotation import daily_report, run_record
from etf_rotation.live import atomic_json, publish_blocked, price_risk_check
from scripts import run_after_close

DATE = "2026-10-08"
ROOT = Path(__file__).resolve().parents[1]


def test_manifest_hashes_only_dated_decisions_and_not_the_input_inventory(tmp_path, monkeypatch):
    price = tmp_path / "market_data/prices/test.csv"
    price.parent.mkdir(parents=True)
    price.write_text("datetime,close\n2026-10-08,1\n")
    calls = []
    sha256 = run_record.sha256
    def counted(path):
        calls.append(path)
        return sha256(path)
    monkeypatch.setattr(run_record, "sha256", counted)
    publish_blocked(tmp_path, DATE, "无已核对账户")
    manifest = daily_report.read_json(tmp_path / f"results/audit/{DATE}_run_manifest.json")
    assert manifest["schema_version"] == 2
    assert len(calls) == 6  # plan + JSON/MD/2 HTML reports + compact manifest
    assert set(manifest["counts"]) == {"critical_files", "source_files", "price_files"}
    assert manifest["counts"]["critical_files"] == 5
    assert manifest["counts"]["source_files"] == 0
    assert manifest["price_files"] == [{"path": "market_data/prices/test.csv"}]
    assert price not in calls
    price.write_text(price.read_text() + "2026-10-09,2\n")
    assert daily_report.validate_delivery(tmp_path, DATE)["delivery"] == "PASS"


def test_legacy_full_tree_hashes_are_optional_not_a_daily_delivery_dependency(tmp_path):
    publish_blocked(tmp_path, DATE, "观察")
    code = tmp_path / "src/example.py"
    code.parent.mkdir()
    code.write_text("# historical implementation\n")
    manifest_path = tmp_path / f"results/audit/{DATE}_run_manifest.json"
    manifest = daily_report.read_json(manifest_path)
    manifest.pop("schema_version")
    manifest["source_files"] = [{"path": "src/example.py", "sha256": run_record.sha256(code)}]
    atomic_json(manifest_path, manifest)
    card_path = tmp_path / f"results/audit/{DATE}_live_run_card.json"
    card = daily_report.read_json(card_path)
    card["audit"]["run_manifest_sha256"] = run_record.sha256(manifest_path)
    atomic_json(card_path, card)
    assert daily_report.validate_delivery(tmp_path, DATE, deep=True)["delivery"] == "PASS"
    code.write_text("# maintenance after publication\n")
    assert daily_report.validate_delivery(tmp_path, DATE)["delivery"] == "PASS"
    with pytest.raises(ValueError, match="清单哈希"):
        daily_report.validate_delivery(tmp_path, DATE, deep=True)


@pytest.mark.parametrize("suffix", ["order_plan.json", "daily_report.json", "daily_report.md"])
def test_changed_daily_decision_or_report_is_still_rejected(tmp_path, suffix):
    publish_blocked(tmp_path, DATE, "观察")
    path = tmp_path / f"results/live/{DATE}_{suffix}"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="清单哈希"):
        daily_report.validate_delivery(tmp_path, DATE)


def test_changed_html_is_still_rejected(tmp_path):
    publish_blocked(tmp_path, DATE, "观察")
    path = tmp_path / "outputs/ETF轮动策略_今日日报.html"
    path.write_text(path.read_text().replace("正式订单未放行", "买入已放行"))
    with pytest.raises(ValueError, match="清单哈希"):
        daily_report.validate_delivery(tmp_path, DATE)


def test_new_renderer_does_not_invalidate_an_existing_report(tmp_path, monkeypatch):
    publish_blocked(tmp_path, DATE, "观察")
    monkeypatch.setattr(daily_report, "render", lambda *_args, **_kwargs: "updated template")
    assert daily_report.validate_delivery(tmp_path, DATE)["delivery"] == "PASS"


def test_failure_renders_once_after_orders_are_revoked(tmp_path, monkeypatch, capsys):
    from etf_rotation import risk_release
    monkeypatch.setattr(run_after_close, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["run", "--date", DATE])
    def fail(_):
        raise ValueError("数据缺失")
    monkeypatch.setattr(run_after_close, "execute", fail)
    def no_sell(root, date, failure):
        plan = daily_report.read_json(root / f"results/live/{date}_order_plan.json")
        assert not plan["actions"] and not plan["execution"]["orders"]
        assert not (root / "outputs/ETF轮动策略_今日日报.html").exists()
        return None
    monkeypatch.setattr(risk_release, "publish_sell_only", no_sell)
    collect = daily_report.collect
    calls = []
    def counted(*args):
        calls.append(args)
        return collect(*args)
    monkeypatch.setattr(daily_report, "collect", counted)
    with pytest.raises(SystemExit) as exc:
        run_after_close.main()
    assert exc.value.code == 2 and len(calls) == 1
    assert "[今日日报](<" in capsys.readouterr().out


def test_price_warnings_reuse_ranking_and_match_independent_calculation():
    report = daily_report.read_json(ROOT / f"results/live/{DATE}_daily_report.json")
    expected = price_risk_check(ROOT, DATE, report["account"])
    actual = price_risk_check(ROOT, DATE, report["account"], ranking=report["analysis"]["ranking"])
    assert actual == expected == report["price_risk_check"]
