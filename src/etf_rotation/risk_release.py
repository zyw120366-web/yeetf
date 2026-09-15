"""Independent, authenticated sell-only release. Never selects or buys an ETF."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import yaml

from .data import load_panel, universe_keys
from .etfwin import etfwin_features
from .execution import entry_eligibility
from .live import atomic_json, raw_quote, validate_account
from .sentiment_ai import validate_live_review
from .ye import _rules, anchor_competitive_rank


def assess_sell_only(root: Path, date: str) -> dict:
    from scripts.validate_live_readiness import previous_execution_is_reconciled
    account = json.loads((root / "results/live/account_state.json").read_text())
    validate_account(account, date)
    if not previous_execution_is_reconciled(date, root):
        raise ValueError("上一订单未对账，不能发布风险卖单")
    if len(account["positions"]) != 1:
        raise ValueError("没有可卖出的单一持仓")
    market = yaml.safe_load((root / "config/market.yaml").read_text())
    config = yaml.safe_load((root / "config/ye_strategy.yaml").read_text())
    rules = config["rules"]
    # Unsupported future rule changes must not silently inherit this fallback.
    if any(rules.get(k, 1) != 1 for k in ("ma_exit_confirmation_days", "roc_exit_confirmation_days", "exit_confirmation_days")) or rules.get("macd_soft_exit_confirmation"):
        raise ValueError("当前退出确认规则不支持独立风险发布")
    market["project"]["data_end"] = date
    panel = load_panel(market, root / "market_data/prices")
    calendar = panel["close"].index
    day = pd.Timestamp(date)
    if day not in calendar:
        raise ValueError("信号日无有效行情")
    position = account["positions"][0]
    symbol = position["symbol"]
    _, raw_close = raw_quote(root, symbol, date)
    if not math.isclose(float(position["market_price"]), raw_close, abs_tol=1e-8):
        raise ValueError("账户与当日不复权收盘价不符")
    benchmark = pd.read_csv(root / "market_data/prices/510300.SH.csv", parse_dates=["datetime"])
    expected = pd.DatetimeIndex(benchmark["datetime"])
    expected = expected[expected <= day][-int(rules["ma_days"]):]
    if expected.empty or expected[-1] != day or expected.has_duplicates or not expected.is_monotonic_increasing:
        raise ValueError("持仓核验使用的交易日历不完整或未更新到信号日")
    held = panel["close"][symbol].reindex(expected)
    original = pd.read_csv(root / "market_data/prices" / f"{symbol}.csv", parse_dates=["datetime"])
    original = original.loc[original["datetime"].isin(expected)]
    if (original["datetime"].duplicated().any() or len(original) != len(expected)
            or original["close"].isna().any() or not original["close"].gt(0).all()):
        raise ValueError("持仓原始日线缺失，不能用向前填充行情放行")
    if len(held) < rules["ma_days"] or held.isna().any() or not held.gt(0).all():
        raise ValueError("持仓价格历史缺失，不得发布卖单")
    reasons = []
    ma = float(held.mean())
    if rules["exit_on_ma_break"] and float(held.iloc[-1]) < ma:
        reasons.append("跌破MA120（硬退出）")
    review_dates = []
    if not reasons:
        # Soft exits may be protected by recent news. Missing review is UNKNOWN,
        # not permission to remove protection. A hard MA exit needs no news.
        memory = int(config["enhanced_selection"]["sentiment_available"]["hot_exit_protection"]["memory_days"])
        recent = expected[-memory:]
        for review_day in recent:
            validate_live_review(root, str(review_day.date()))
            review_dates.append(str(review_day.date()))
        from scripts.build_sentiment_features import build
        features = build(root=root)
        rows = features.loc[features["symbol"].eq(symbol)].set_index("date")
        hot = config["enhanced_selection"]["sentiment_available"]["hot_exit_protection"]
        for review_day in recent:
            row = rows.loc[review_day]
            if (row["hot_score"] >= hot["hot_score_min"] and row["matched_count"] >= hot["matched_hot_stocks_min"]
                    and row["positive_dde_share"] >= hot["positive_dde_share_min"]):
                raise ValueError("存在热点软退出保护；不能独立卖出")
        symbols = universe_keys(market)
        values = etfwin_features(panel["close"][symbols], _rules(rules))
        if rules["exit_on_short_roc_negative"] and values.roc_short.at[day, symbol] < 0:
            reasons.append("ROC20转负")
        if not reasons:
            for key in symbols:
                history = pd.read_csv(root / "market_data/prices" / f"{key}.csv", parse_dates=["datetime"])
                current = history.loc[history["datetime"].eq(day), "close"]
                if len(current) != 1 or not math.isfinite(float(current.iloc[0])) or float(current.iloc[0]) <= 0:
                    raise ValueError("排名退出需要全池当日有效行情")
        eligibility, _, _ = entry_eligibility(panel, symbols, rules)
        satellites = config["enhanced_selection"]["universe_architecture"]["challenger_symbols"]
        rank = anchor_competitive_rank(values.ranking_score, eligibility, [s for s in symbols if s not in satellites], satellites)
        if rules["exit_on_dual_rank_decline"] and (
                rank.at[day, symbol] > rank[symbol].shift(rules["rank_change_short_days"]).at[day]
                and rank.at[day, symbol] > rank[symbol].shift(rules["rank_change_long_days"]).at[day]):
            reasons.append("5日与20日排名同时下滑")
    if not reasons:
        raise ValueError("未满足可独立确认的价格退出条件")
    quantity = float(position["quantity"])
    action = {"side": "sell", "symbol": symbol, "target_weight": 0.0, "reasons": reasons}
    return {"strategy": "ye 策略", "signal_date": date, "execute": "下一交易日开盘",
            "current_symbol": symbol, "target_symbol": None, "account_state": account,
            "actions": [action], "review_dates_required": review_dates,
            "risk_evidence": {"adjusted_close": float(held.iloc[-1]), "ma120": ma,
                              "unadjusted_close": raw_close, "price_window_dates": len(held)},
            "execution": {"release_required": True, "sell_only": True, "orders": [
                {**action, "confirmed_quantity": quantity,
                 "instruction": "仅卖出已核对的现有持仓，卖出后保持现金；禁止买入或借券卖空"}]}}


def publish_sell_only(root: Path, date: str, failure: str) -> bool:
    """Called only after ordinary publication failed and BLOCKED was saved."""
    try:
        plan = assess_sell_only(root, date)
    except Exception:
        return False
    live, audit = root / "results/live", root / "results/audit"
    action = plan["actions"][0]
    ready = {"signal_date": date, "status": "SELL_ONLY", "buy_allowed": False, "sell_allowed": True,
             "blocking_items": [failure], "sell_reasons": action["reasons"],
             "note": "只放行独立校验的风险卖单，不代表完整策略READY"}
    atomic_json(live / f"{date}_order_plan.json", plan)
    atomic_json(live / "readiness_report.json", ready)
    from .daily_report import publish, validate_delivery
    report = publish(root, date)
    template = {"signal_date": date, "confirmation_status": "pending", "fills": [
        {"side": "sell", "symbol": action["symbol"], "status": "unfilled", "quantity": 0, "price": 0}]}
    atomic_json(live / f"{date}_actual_fills.template.json", template)
    paths = [live / f"{date}_order_plan.json", live / "account_state.json",
             live / "readiness_report.json", live / f"{date}_daily_report.md",
             root / "market_data/live_quotes" / f"{date}.json",
             live / f"{date}_daily_report.json", root / "outputs/ETF轮动策略_今日日报.html",
             root / "dashboard/public/ye-daily.html"]
    paths += list((root / "config").glob("*.yaml"))
    paths += list((root / "market_data/prices").glob("*.csv"))
    paths += list((root / "src/etf_rotation").glob("*.py"))
    paths += [root / "scripts/build_sentiment_features.py", root / "scripts/validate_live_readiness.py"]
    for d in plan["review_dates_required"]:
        paths += [root / "market_data/sentiment" / f"{d}.json", root / "market_data/sentiment/ai_review" / f"{d}.json"]
    previous = sorted(p for p in live.glob("*_order_plan.json") if p.name[:10] < date)
    if previous:
        paths.append(previous[-1])
        rec = audit / f"{previous[-1].name[:10]}_execution_reconciliation.json"
        if rec.exists():
            paths.append(rec)
    def record(p):
        return {"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    manifest_path = audit / f"{date}_run_manifest.json"
    atomic_json(manifest_path, {"signal_date": date, "status": "SELL_ONLY", "critical_files": [record(p) for p in paths]})
    atomic_json(audit / f"{date}_live_run_card.json", {
        "card_type": "ye_live_run_card", "signal_date": date,
        "release": {"readiness": "SELL_ONLY", "buy_allowed": False, "sell_allowed": True, "plan_is_not_fill": True},
        "account_state": plan["account_state"], "sentiment_review": report["review"], "decision": {
            "current_symbol": plan["current_symbol"], "target_symbol": None, "actions": plan["actions"]},
        "audit": {"run_manifest": str(manifest_path.relative_to(root)),
                  "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}})
    validate_delivery(root, date)
    return True
