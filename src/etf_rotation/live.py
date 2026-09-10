"""Small shared checks for live accounting, plans and publication."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import html
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def live_lock(root: Path):
    path = root / "results/live/.live-run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
        else:
            import fcntl
        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("已有实盘任务运行，请待其完成后重试") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def valid_number(value: object, *, positive: bool = False) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value)) and (
            float(value) > 0 if positive else float(value) >= 0
        )
    except (ValueError, TypeError):
        return False


def validate_account(account: dict, date: str | None = None) -> None:
    if account.get("confirmation_status") not in {"confirmed", "assumed_authorized"}:
        raise ValueError("账户状态未确认")
    if date and str(account.get("as_of", "")).split("_")[0] != date:
        raise ValueError("账户日期与信号日不一致")
    if account.get("pending_orders"):
        raise ValueError("存在未处理订单")
    if not valid_number(account.get("available_cash")) or not valid_number(account.get("total_equity"), positive=True):
        raise ValueError("账户现金或权益无效")
    positions = account.get("positions")
    if not isinstance(positions, list) or len(positions) > 1:
        raise ValueError("账户必须为空仓或单一ETF")
    value = float(account["available_cash"])
    for p in positions:
        if not p.get("symbol") or not p.get("opened_on"):
            raise ValueError("持仓缺少代码或开仓日期")
        opened = pd.Timestamp(p["opened_on"])
        as_of = pd.Timestamp(str(account.get("as_of", "")).split("_")[0])
        if pd.isna(opened) or pd.isna(as_of) or opened > as_of:
            raise ValueError("开仓日期晚于账户日期或日期无效")
        for key in ("quantity", "average_cost", "market_price"):
            if not valid_number(p.get(key), positive=True):
                raise ValueError(f"持仓字段无效: {key}")
        if float(p["quantity"]) != int(float(p["quantity"])):
            raise ValueError("ETF数量必须为整数")
        mark = float(p["quantity"]) * float(p["market_price"])
        if not math.isclose(mark, float(p.get("market_value", -1)), abs_tol=.011):
            raise ValueError("持仓市值与数量、价格不一致")
        value += mark
    if not math.isclose(value, float(account["total_equity"]), abs_tol=.011):
        raise ValueError("现金加持仓市值不等于账户权益")


def raw_quote(root: Path, symbol: str, date: str) -> tuple[float, float]:
    path = root / "market_data" / "live_quotes" / f"{date}.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("date") != date or snapshot.get("adjust") != "NONE" or snapshot.get("final") is not True:
        raise ValueError("实盘必须使用当日完整、不复权行情")
    row = snapshot["quotes"][symbol]
    if not all(valid_number(row.get(key), positive=True) for key in ("open", "close")):
        raise ValueError(f"实盘价格无效: {symbol}")
    return float(row["open"]), float(row["close"])


def validate_fills(fills: dict, plan: dict, date: str) -> None:
    if fills.get("signal_date") != date or plan.get("signal_date") != date:
        raise ValueError("成交文件与计划信号日不一致")
    records = fills.get("fills", [])
    expected = [(x["side"], x["symbol"]) for x in plan["actions"] if x["side"] in {"sell", "buy"}]
    actual = [(x.get("side"), x.get("symbol")) for x in records]
    if actual != expected or len(set(actual)) != len(actual):
        raise ValueError("成交必须对应计划，且换仓须先卖后买")
    for row in records:
        status = row.get("status")
        if status not in {"filled", "partial", "cancelled", "unfilled"}:
            raise ValueError("成交状态无效")
        quantity = row.get("quantity")
        if not valid_number(quantity) or float(quantity) != int(float(quantity)):
            raise ValueError("成交数量必须为非负整数")
        if status in {"filled", "partial"}:
            if not valid_number(quantity, positive=True) or not valid_number(row.get("price"), positive=True):
                raise ValueError("已成交数量和价格必须大于0")
        elif float(quantity) != 0:
            raise ValueError("取消或未成交订单的成交数量必须为0")
        if row["side"] == "sell" and status == "filled":
            held = next((x for x in plan.get("account_state", {}).get("positions", []) if x["symbol"] == row["symbol"]), None)
            if held and float(quantity) != float(held["quantity"]):
                raise ValueError("完整卖出数量与计划账户不符，应记录为部分成交")


def validate_authorized_plan(root: Path, plan: dict, account: dict, prior: str) -> None:
    """Never apply a draft, a changed plan or an existing execution twice."""
    card_path = root / "results" / "audit" / f"{prior}_live_run_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    release = card.get("release", {}).get("readiness")
    if card.get("signal_date") != prior or release not in {"READY", "SELL_ONLY"}:
        raise ValueError("上一计划未放行，禁止假定成交")
    if release == "SELL_ONLY":
        actions = plan.get("actions", [])
        if (not actions or any(x.get("side") != "sell" for x in actions)
                or plan.get("target_symbol") is not None
                or card.get("decision", {}).get("actions") != actions):
            raise ValueError("仅卖出放行不允许任何买单或目标仓位")
    if plan.get("signal_date") != prior or card.get("decision", {}).get("target_symbol") != plan.get("target_symbol"):
        raise ValueError("计划日期或目标与运行卡不一致")
    manifest = json.loads((root / card["audit"]["run_manifest"]).read_text(encoding="utf-8"))
    if hashlib.sha256((root / card["audit"]["run_manifest"]).read_bytes()).hexdigest() != card["audit"]["run_manifest_sha256"]:
        raise ValueError("运行卡与清单哈希不一致")
    records = {x["path"]: x["sha256"] for x in manifest["critical_files"]}
    relative = f"results/live/{prior}_order_plan.json"
    if records.get(relative) != hashlib.sha256((root / relative).read_bytes()).hexdigest():
        raise ValueError("计划在放行后发生变化")
    if fingerprint(plan.get("account_state", {}).get("positions")) != fingerprint(account.get("positions")) or float(plan["account_state"]["available_cash"]) != float(account["available_cash"]):
        raise ValueError("当前账户与上一计划起点不符")
    rec_path = root / "results" / "audit" / f"{prior}_execution_reconciliation.json"
    if rec_path.exists():
        status = json.loads(rec_path.read_text(encoding="utf-8")).get("status")
        if status != "not_required":
            raise ValueError(f"已有成交/取消/异常记录({status})，不得用假定记录覆盖")
    if (root / "results" / "live" / f"{prior}_actual_fills.json").exists():
        raise ValueError("已有用户成交文件，必须先处理真实回单")


def apply_live_cooldown(ranking: pd.DataFrame, account: dict, day: str,
                        calendar: pd.DatetimeIndex, cooldown: int) -> pd.DataFrame:
    result = ranking.copy()
    result["live_cooldown_blocked"] = False
    for symbol, sold in account.get("last_exit_dates", {}).items():
        # Exit is recorded on execution day. The strategy's exit signal is the
        # preceding session, so five post-signal sessions include the sale day.
        elapsed = int(((calendar >= pd.Timestamp(sold)) & (calendar <= pd.Timestamp(day))).sum())
        if elapsed <= cooldown:
            result.loc[result["symbol"].eq(symbol), "live_cooldown_blocked"] = True
    technical = result["technical_entry_pass"].astype(bool) & ~result["live_cooldown_blocked"]
    core_available = bool((technical & result["pool_role"].eq("core")).any())
    result["final_entry_pass"] = technical & (result["pool_role"].eq("core") | ~pd.Series(core_available, index=result.index))
    return result


def price_risk_check(root: Path, date: str, account: dict) -> list[str]:
    """Keep price warnings visible when the news/publication stage fails."""
    if not account.get("positions"):
        return ["账户记录为空仓"]
    try:
        import yaml
        from .data import load_panel, universe_keys
        from .ye import _rules, anchor_competitive_rank
        from .etfwin import etfwin_features
        from .execution import entry_eligibility
        market = yaml.safe_load((root / "config/market.yaml").read_text())
        config = yaml.safe_load((root / "config/ye_strategy.yaml").read_text())
        market["project"]["data_end"] = date
        panel = load_panel(market, root / "market_data/prices")
        symbols = universe_keys(market)
        values = config["rules"]
        close = panel["close"][symbols]
        day = pd.Timestamp(date)
        features = etfwin_features(close, _rules(values))
        eligible, _, _ = entry_eligibility(panel, symbols, values)
        satellites = config["enhanced_selection"]["universe_architecture"]["challenger_symbols"]
        ranks = anchor_competitive_rank(features.ranking_score, eligible,
                                        [s for s in symbols if s not in satellites], satellites)
        dual = ranks.gt(ranks.shift(values["rank_change_short_days"])) & ranks.gt(ranks.shift(values["rank_change_long_days"]))
        warnings = []
        for position in account["positions"]:
            symbol = position["symbol"]
            price, ma = close.at[day, symbol], close[symbol].rolling(values["ma_days"]).mean().at[day]
            roc = features.roc_short.at[day, symbol]
            if not all(valid_number(x, positive=True) for x in (price, ma)) or not math.isfinite(roc):
                raise ValueError("持仓价格历史不完整")
            if price < ma:
                warnings.append(f"{symbol}：跌破MA120，触发价格硬退出")
            if roc < 0:
                warnings.append(f"{symbol}：ROC20转负，触发软退出价格条件")
            if dual.at[day, symbol]:
                warnings.append(f"{symbol}：5日与20日排名同时下滑，触发软退出价格条件")
        return warnings or ["未触发价格退出条件；这不等于完整策略已放行"]
    except Exception as exc:
        return [f"价格风控未完成：{exc}；不得把运行失败解释为继续持有"]


def blocked_html(date: str, reasons: list[str], account: dict, risks: list[str] | None = None) -> str:
    holdings = "、".join(f"{p.get('name', p.get('symbol'))}（{p.get('symbol')}）{p.get('quantity')}份" for p in account.get("positions", [])) or "空仓"
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ye 策略今日日报</title>'
            '<body style="max-width:850px;margin:50px auto;font:18px/1.8 sans-serif;padding:20px">'
            f'<h1>ye 策略日报 · {html.escape(date)}</h1><h2>本次未放行，暂无可执行买入计划</h2>'
            f'<p>当前记录持仓：{html.escape(holdings)}</p><p>阻断原因：{html.escape("；".join(reasons))}</p>'
            f'<p>价格风控：{html.escape("；".join(risks or ["未完成，不能据此判断继续持有"]))}</p>'
            '<p>软退出仍需完整策略确认；以上不是换仓放行，不得执行新增买单。</p>'
            '<p>旧日报和旧买单不代表今天的决定。待本次数据与账户检查通过后重新生成计划。</p></body></html>')


def publish_blocked(root: Path, date: str, reason: str) -> None:
    live = root / "results/live"
    account_path = live / "account_state.json"
    account = json.loads(account_path.read_text(encoding="utf-8")) if account_path.exists() else {}
    ready_path = live / "readiness_report.json"
    previous = json.loads(ready_path.read_text(encoding="utf-8")) if ready_path.exists() else {}
    readiness = previous if previous.get("signal_date") == date else {}
    readiness.update(signal_date=date, status="BLOCKED", error=reason)
    readiness["blocking_items"] = readiness.get("blocking_items") or [reason]
    atomic_json(ready_path, readiness)
    risks = price_risk_check(root, date, account)
    body = blocked_html(date, readiness["blocking_items"], account, risks)
    for path in (root / "outputs/ETF轮动策略_今日日报.html", root / "dashboard/public/ye-daily.html"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    (live / f"{date}_daily_report.md").write_text(f"# ye 策略日报｜{date}\n\nBLOCKED：本次未放行，无可执行买入计划。\n\n{reason}\n\n价格风控：{'；'.join(risks)}。软退出仍需完整策略确认。\n", encoding="utf-8")
    card = {"card_type": "ye_live_run_card", "signal_date": date, "account_state": account,
            "release": {"readiness": "BLOCKED", "blocking_items": readiness["blocking_items"], "plan_is_not_fill": True},
            "decision": {"actions": [], "target_symbol": None}, "price_risk_check": risks, "error": reason}
    atomic_json(root / "results/audit" / f"{date}_live_run_card.json", card)
    paths = [ready_path, account_path, live / f"{date}_daily_report.md", root / "outputs/ETF轮动策略_今日日报.html"]
    atomic_json(root / "results/audit" / f"{date}_run_manifest.json", {
        "signal_date": date, "status": "BLOCKED", "error": reason,
        "critical_files": [{"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths if p.exists()]})
