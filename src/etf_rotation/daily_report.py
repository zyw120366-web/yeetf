"""One read-only report model and renderer for READY, SELL_ONLY and BLOCKED.

Reporting never advances an account, releases an order or runs a backtest.
Missing evidence is a displayed limitation, not a substituted trading signal.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path

import pandas as pd
import yaml

from .live import atomic_json, apply_live_cooldown, raw_quote, validate_account, price_risk_check
from .sentiment_ai import validate_live_review


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def known_position(account: dict, date: str):
    """Accept a cash-only reconciliation gap for observation, never execution."""
    positions = account.get("positions", [])
    trusted = account.get("confirmation_status") in {"confirmed", "assumed_authorized"}
    if not trusted and account.get("cash_reconciliation", {}).get("status") == "pending":
        fills = account.get("actual_execution", {}).get("fills", [])
        trusted = len(positions) == 1 and any(
            f.get("side") == "buy" and f.get("symbol") == positions[0].get("symbol")
            and f.get("quantity") == positions[0].get("quantity")
            and f.get("price") == positions[0].get("average_cost") for f in fills)
    if not trusted or account.get("pending_orders") or len(positions) > 1:
        raise ValueError("持仓事实或待处理订单尚未核清")
    as_of = str(account.get("as_of", "")).split("_")[0]
    if not as_of or as_of > date:
        raise ValueError("账户快照缺失或晚于信号日，不能倒推历史持仓")
    if not positions:
        return None
    p = positions[0]
    if not p.get("opened_on") or str(p["opened_on"]) > date:
        raise ValueError("持仓开仓日期不可用")
    for key in ("quantity", "average_cost"):
        if number(p.get(key)) is None or float(p[key]) <= 0:
            raise ValueError("持仓数量或成本未知")
    if int(p["quantity"]) != p["quantity"]:
        raise ValueError("持仓数量不是整数")
    return p


def analyze(root: Path, date: str, account: dict) -> dict:
    """Use the exact formal ranking/holding functions without publishing signals."""
    from .data import load_panel, universe_keys, symbol_key
    from .ranking import ranking_for_day
    from .ye import build_ye_signals
    from .sentiment import sentiment_matrices_from_frame
    from scripts.build_sentiment_features import build
    from scripts.build_live_order_plan import holding_decision, order_candidates

    market = yaml.safe_load((root / "config/market.yaml").read_text())
    config = yaml.safe_load((root / "config/ye_strategy.yaml").read_text())
    market["project"]["data_end"] = date  # in-memory only
    symbols = universe_keys(market)
    satellites = config["enhanced_selection"]["universe_architecture"]["challenger_symbols"]
    if len(symbols) != 51 or len(set(symbols)) != 51 or len(satellites) != 6:
        raise ValueError("固定池应为45核心与6卫星")
    day = pd.Timestamp(date)
    # Reject missing current bars before load_panel can forward fill them.
    for s in symbols:
        frame = pd.read_csv(root / f"market_data/prices/{s}.csv")
        current = frame.loc[frame["datetime"].eq(date)]
        if len(current) != 1 or number(current.iloc[0]["close"]) is None or current.iloc[0]["close"] <= 0:
            raise ValueError(f"{s} 缺少当日有效日线")
    panel = load_panel(market, root / "market_data/prices")
    calendar = panel["close"].index
    validate_live_review(root, date)
    memory = int(config["enhanced_selection"]["sentiment_available"]["hot_exit_protection"]["memory_days"])
    for previous in calendar[calendar <= day][-memory:]:
        validate_live_review(root, str(previous.date()))
    sentiment, available = sentiment_matrices_from_frame(build(root=root), calendar, symbols)
    names = {symbol_key(x): x["name"] for x in market["universe"]}
    categories = {symbol_key(x): x["category"] for x in market["universe"]}
    bundle, features, eligible, listed, amount, decision = build_ye_signals(
        panel, symbols, categories, config, sentiment, available)
    ranking = ranking_for_day(panel, symbols, names, categories, config, bundle,
                              features, eligible, listed, amount, decision, sentiment, day)
    ranking = apply_live_cooldown(ranking, account, date, calendar,
                                 int(config["enhanced_selection"]["reentry_cooldown_days"]))
    candidates = order_candidates(ranking.loc[ranking.final_entry_pass])
    result = {"status": "complete", "ranking": ranking.to_dict("records"),
              "candidates": candidates[["symbol", "name", "rank"]].to_dict("records"),
              "market": {"up": int(ranking.change_1d.gt(0).sum()),
                         "down": int(ranking.change_1d.lt(0).sum()),
                         "flat": int(ranking.change_1d.eq(0).sum()),
                         "benchmark_change": panel["close"]["510300.SH"].pct_change(fill_method=None).at[day],
                         "benchmark_above_ma": bool(panel["close"].at[day, "510300.SH"] > panel["close"]["510300.SH"].rolling(120).mean().at[day])}}
    try:
        position = known_position(account, date)
        current = position["symbol"] if position else None
        target, held, reasons, switch = holding_decision(
            day, current, ranking, candidates, account, config, root=root)
        result.update(held=None if held is None else held.to_dict(), exit_reasons=reasons,
                      opportunity_switch=switch, technical_target=target, holding_trusted=True)
    except (ValueError, KeyError, RuntimeError) as exc:
        result.update(holding_trusted=False, holding_error=str(exc), technical_target=None)
    return clean(result)


def valuation(root: Path, date: str, account: dict) -> dict:
    result = {"account_return": None, "account_equity": None, "purchase_pnl": None,
              "purchase_return": None, "daily_pnl": None, "daily_return": None}
    try:
        p = known_position(account, date)
        if p:
            _, close = raw_quote(root, p["symbol"], date)
            qty, cost = float(p["quantity"]), float(p["average_cost"])
            result.update(symbol=p["symbol"], name=p.get("name", p["symbol"]), quantity=qty,
                          close=close, cost=cost, market_value=qty * close,
                          purchase_pnl=qty * (close - cost), purchase_return=close / cost - 1)
            prior = sorted(x for x in (root / "market_data/live_quotes").glob("*.json") if x.stem < date)
            calendar = pd.read_csv(root / "market_data/prices/510300.SH.csv")["datetime"]
            previous_days = calendar[calendar < date]
            previous = previous_days.iloc[-1] if len(previous_days) else None
            if str(p["opened_on"]) < date and prior and prior[-1].stem == previous:
                _, old_close = raw_quote(root, p["symbol"], previous)
                result.update(daily_pnl=qty * (close - old_close), daily_return=close / old_close - 1)
            elif str(p["opened_on"]) == date:
                result.update(daily_pnl=result["purchase_pnl"], daily_return=result["purchase_return"],
                              daily_basis="今日新买入，按成交价至收盘计算")
        validate_account(account, date)
        result["account_equity"] = float(account["total_equity"])
        capital = number(account.get("performance", {}).get("net_contributed_capital"))
        if capital is not None and capital > 0:
            result["account_return"] = result["account_equity"] / capital - 1
    except (ValueError, KeyError, OSError, IndexError) as exc:
        result["limitation"] = str(exc)
    return clean(result)


def collect(root: Path, date: str) -> dict:
    account = read_json(root / "results/live/account_state.json")
    readiness = read_json(root / "results/live/readiness_report.json")
    plan = read_json(root / f"results/live/{date}_order_plan.json")
    status = readiness.get("status") if readiness.get("signal_date") == date else "BLOCKED"
    if status not in {"READY", "SELL_ONLY", "BLOCKED"}:
        status = "BLOCKED"
    reasons = readiness.get("blocking_items") or []
    if readiness.get("signal_date") != date:
        reasons = ["放行日期与日报日期不一致"]
    if plan.get("signal_date") != date:
        status, reasons, plan = "BLOCKED", ["缺少当日订单计划"], {}
    try:
        review = validate_live_review(root, date)
        review = {k: review.get(k) for k in ("status", "input_count", "reviewed_count", "coverage", "snapshot_hash")}
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        review = {"status": "unverified", "coverage": None, "error": str(exc)}
    if status in {"READY", "SELL_ONLY"}:
        try:
            validate_account(account, date)
            if status == "READY" and review["status"] != "complete":
                raise ValueError("完整计划缺少有效资讯审核")
        except ValueError as exc:
            status, reasons = "BLOCKED", [str(exc)]
    analysis = {"status": "unavailable", "ranking": [], "candidates": []}
    try:
        analysis = analyze(root, date, account)
    except Exception as exc:
        analysis["error"] = f"{type(exc).__name__}: {exc}"
    snapshot = read_json(root / f"market_data/sentiment/{date}.json")
    sources = {k: {"ok": v.get("ok"), "rows": len(v.get("rows") or [])}
               for k, v in snapshot.get("sources", {}).items()}
    # Never show an order from a failed or mismatched release as executable.
    orders = plan.get("execution", {}).get("orders", []) if status in {"READY", "SELL_ONLY"} else []
    if status == "SELL_ONLY" and any(x.get("side") != "sell" for x in orders):
        status, reasons, orders = "BLOCKED", ["仅卖出状态中出现非卖出订单"], []
    return clean({"schema_version": 1, "signal_date": date, "status": status,
                  "blocking_items": reasons, "account": account, "valuation": valuation(root, date, account),
                  "analysis": analysis, "review": review, "sources": sources,
                  "orders": orders, "actions": plan.get("actions", []) if status != "BLOCKED" else [],
                  "price_risk_check": price_risk_check(root, date, account),
                  "available_links": [p for p in ["outputs/ETF轮动策略_回测.html", "outputs/ETF轮动策略_策略与回测.html",
                                      f"market_data/sentiment/ai_review/{date}.json"] if (root / p).is_file()],
                  "target_symbol": plan.get("target_symbol") if status != "BLOCKED" else None,
                  "target_is_executable": status in {"READY", "SELL_ONLY"},
                  "cash_management_status": "以券商回单为准；未核验不阻断ETF日报"})


def pct(value):
    return "待核" if number(value) is None else f"{float(value):+.2%}"


def money(value):
    return "待核" if number(value) is None else f"{float(value):+,.2f}元"


def describe_action(report: dict) -> str:
    a, status = report["analysis"], report["status"]
    if status == "BLOCKED":
        if a.get("holding_trusted") and a.get("held"):
            held = a["held"]
            detail = "；".join(a.get("exit_reasons", [])) or "未触发卖出或换仓，观察持有"
            return f"持仓观察：{held['name']}，{detail}。正式订单未放行，暂无可执行买入计划。"
        return "正式订单未放行，暂无可执行买入计划；持仓结论见可用证据。"
    actions = report["actions"]
    labels = {"buy": "买入", "sell": "卖出", "hold": "持有"}
    names = {x["symbol"]: x["name"] for x in a.get("ranking", [])}
    descriptions = []
    for action in actions:
        label = labels.get(action.get("side"), "待核")
        name = names.get(action.get("symbol"), action.get("symbol") or "现金")
        weight = number(action.get("target_weight"))
        descriptions.append(f"{label} {name}" + (f"（目标仓位{weight:.0%}）" if weight is not None else ""))
    text = "；".join(descriptions)
    return "次日开盘：" + (text or "待核") + ("；仅卖出转现金，禁止买入。" if status == "SELL_ONLY" else "。计划不是成交。")


def sections(report: dict):
    a, v, review = report["analysis"], report["valuation"], report["review"]
    overview = [describe_action(report)]
    if v.get("symbol"):
        overview.append(f"{v['name']} {v['quantity']:g}股，收盘{v['close']:.3f}元；今日持仓{money(v['daily_pnl'])}，本次持仓{money(v['purchase_pnl'])}（{pct(v['purchase_return'])}，未计费用）。")
    held = a.get("held")
    if held:
        overview.append(f"持仓核心/参考排名第{held['rank']:g}，ROC20 {pct(held['roc20'])}、ROC60 {pct(held['roc60'])}，MA120乖离{pct(held['ma120_bias'])}。")
    if a.get("market"):
        m = a["market"]
        overview.append(f"全池{m['up']}涨、{m['down']}跌、{m['flat']}平；沪深300ETF {pct(m['benchmark_change'])}，在MA120{'上' if m['benchmark_above_ma'] else '下'}方。市场判断只作背景，不新增择时开关。")
    if report["status"] == "BLOCKED":
        overview.append("订单阻断原因：" + "；".join(report["blocking_items"]) + "。已确认持仓与未知现金分开处理，未知账户收益不填零。")
    ranking = a.get("ranking", [])
    def names(rows):
        return "、".join(f"{r['name']}（{r['symbol']}）" for r in rows) or "无"
    if ranking:
        core_top = [r for r in ranking if r["pool_role"] == "core" and number(r["rank"]) is not None and r["rank"] <= 5]
        tech = [r for r in ranking if r["technical_entry_pass"]]
        final = [r for r in ranking if r["final_entry_pass"]]
        blocked = [r for r in ranking if r.get("live_cooldown_blocked")]
        funnel = [f"固定池{len(ranking)}只：45核心＋6卫星；基础资格{sum(bool(r['pool_eligible']) for r in ranking)}只通过。",
                  "核心前5：" + names(core_top), f"技术门槛通过{len(tech)}只：" + names(tech),
                  f"真实冷却与核心优先后{len(final)}只：" + names(final), "真实卖出冷却排除：" + names(blocked),
                  "候选合格不等于重买；先检查旧仓退出及换仓条件，正式动作以放行计划为准。"]
        satellites = [r for r in ranking if r["pool_role"] == "challenger"]
        satellite = ["6只卫星：" + names(satellites),
                     "卫星技术合格：" + names([r for r in satellites if r["technical_entry_pass"]]),
                     "卫星最终补位：" + names([r for r in satellites if r["final_entry_pass"]]),
                     "卫星使用相对核心的虚拟排名，核心有合格候选时不补位。"]
    else:
        funnel = ["全池筛选暂不可确认：" + a.get("error", "证据不足") + "。不使用旧排名或价格回退冒充今日完整筛选。"]
        satellite = ["今日卫星条件未知，不生成新增订单。"]
    checks = []
    if held:
        switch = a.get("opportunity_switch", {})
        checks = [f"持有：{held['name']}；收盘在MA120{'上' if held['above_ma120'] else '下'}方。",
                  "退出/让位：" + ("；".join(a.get("exit_reasons", [])) or "未触发"),
                  f"机会换仓：{switch.get('confirmation_streak', 0)}/{switch.get('required_confirmation_days', 2)}，持有{switch.get('held_trading_days', '—')}个交易日；候选分差{pct(switch.get('score_gap'))}。"]
    else:
        checks = [a.get("holding_error") or "无已确认持仓或完整持仓指标不可用；不能据此称继续持有已获放行。"]
        checks += report.get("price_risk_check", [])
        checks += ["价格提示不等于退出已放行；软退出仍需核验近期资讯保护。"]
    news = [f"审核：{review.get('reviewed_count', '—')}/{review.get('input_count', '—')}，覆盖{pct(review.get('coverage'))}；{review['status']}。",
            "；".join(f"{k}：{v['rows']}条，{'成功' if v['ok'] else '失败'}" for k, v in report["sources"].items()),
            "审核覆盖是对已采集条目的覆盖，不是全网新闻完整性；DDE缺失不能当作已证实的零资金流。"]
    if held:
        news.append(f"持仓直接匹配数量：{held.get('sentiment_matched_count', '—')}；正DDE占比：{pct(held.get('sentiment_positive_dde_share'))}。不把泛行业题材扩散到不相关ETF。")
    return [("今日决策路径综述", overview), ("今日筛选漏斗", funnel),
            ("持有、卖出与换仓", checks), ("今日卫星检查", satellite), ("资讯审核", news)]


def render(report: dict, *, public=False) -> str:
    esc = html.escape
    date = report["signal_date"]
    root = "../../" if public else "../"
    links = report.get("available_links", [])
    navigation = []
    for target, label in [("outputs/ETF轮动策略_回测.html", "回测"), ("outputs/ETF轮动策略_策略与回测.html", "策略介绍")]:
        if target in links:
            navigation.append(f'<a href="{root}{target}">{label}</a>')
    navigation.append(f'<a href="{root}results/audit/{date}_live_run_card.json">运行卡</a>')
    evidence = (f'<a href="{root}market_data/sentiment/ai_review/{date}.json">逐条审核证据</a> · '
                if f"market_data/sentiment/ai_review/{date}.json" in links else "")
    v = report["valuation"]
    blocks = []
    for i, (title, paragraphs) in enumerate(sections(report)):
        blocks.append(f'<section class="section" id="{"dailyOverview" if i == 0 else "detail" + str(i)}"><h2>{title}</h2>' + "".join(f"<p>{esc(p)}</p>" for p in paragraphs if p) + "</section>")
    rows = []
    for r in report["analysis"].get("ranking", []):
        cells = [r["name"] + "（" + r["symbol"] + "）", "核心" if r["pool_role"] == "core" else "卫星",
                 str(r["rank"]), pct(r["change_1d"]), pct(r["momentum_score"]), pct(r["roc20"]), pct(r["roc60"]), pct(r["ma120_bias"]),
                 "通过" if r["pool_eligible"] else "未通过", "通过" if r["normal_entry"] else "未通过",
                 "通过" if r["emerging_entry"] else "未通过", "通过" if r["quality_extension"] else "未通过",
                 "冷却排除" if r.get("live_cooldown_blocked") else "观察候选" if r["final_entry_pass"] else "未入选"]
        rows.append("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in cells) + "</tr>")
    headers = ["ETF", "角色", "排名", "今日", "动量分", "ROC20", "ROC60", "MA120乖离", "资格", "常规", "新趋势", "延伸", "筛选结果"]
    table = '<section class="section"><h2>全池逐层筛选明细</h2><div class="scroll"><table><thead><tr>' + "".join(f"<th>{x}</th>" for x in headers) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>"
    orders = '<section class="section"><h2>执行边界</h2><p>' + esc(report["status"]) + "：" + esc(describe_action(report)) + "</p>"
    for order in report["orders"]:
        estimate = order.get("buy_estimate", {})
        quantity = order.get("confirmed_quantity", estimate.get("estimated_quantity_at_last_close"))
        side = {"buy": "买入", "sell": "卖出"}.get(order.get("side"), "待核")
        orders += f"<p>{side} {esc(str(order.get('symbol')))}；数量参考：{esc(str(quantity if quantity is not None else '开盘核算'))}；{esc(str(order.get('instruction', '实际开盘按计划核算，禁止把估算当成交。')))}</p>"
        if estimate.get("quantity_rule"):
            orders += f"<p>{esc(estimate['quantity_rule'])}</p>"
    orders += "<p>账户总权益：" + money(v["account_equity"]) + "；账户状态：" + esc(str(report["account"].get("confirmation_status", "未知"))) + "。持仓盈亏未计费用，账户收益与新增本金不可混淆。</p></section>"
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ye 策略日报 · {date}</title>
<style>body{{margin:0;background:#f5f7fa;color:#223044;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1200px;margin:auto;padding:28px}}h1{{font-size:30px}}h2{{font-size:21px}}.section,.stat{{background:white;border:1px solid #e1e6ee;border-radius:12px;padding:22px;margin:18px 0}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.stat strong{{display:block;font-size:27px}}small{{color:#64758b}}a{{color:#315faf}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:10px;border-bottom:1px solid #e5eaf2;white-space:nowrap;text-align:left}}th{{background:#edf2f8}}@media(max-width:720px){{main{{padding:14px}}.stats{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>ye 策略 · 今日日报</h1><p>{date} 收盘｜数据与订单分开核验</p>
<p>{' · '.join(navigation)}</p>
<div class="stats"><div class="stat">策略实盘开启以来<strong>{pct(v['account_return'])}</strong><small>资金未核时不计算</small></div><div class="stat">本次买入收益<strong>{money(v['purchase_pnl'])}</strong><small>{pct(v['purchase_return'])}，未计费用</small></div><div class="stat">今日收益<strong>{money(v['daily_pnl'])}</strong><small>{pct(v['daily_return'])}；仅已确认持仓</small></div></div>
{''.join(blocks)}{table}{orders}
<p>{evidence}<a href="{root}results/live/{date}_daily_report.json">日报结构化数据</a></p>
</main></body></html>'''


def render_markdown(report: dict) -> str:
    lines = [f"# ye 策略日报｜{report['signal_date']}"]
    for title, paragraphs in sections(report):
        lines += ["", "## " + title, "", "\n\n".join(p for p in paragraphs if p)]
    lines += ["", "## 放行", "", report["status"] + "；计划不是成交。"]
    return "\n".join(lines) + "\n"


def publish(root: Path, date: str) -> dict:
    report = collect(root, date)
    atomic_json(root / f"results/live/{date}_daily_report.json", report)
    for relative, content in [(f"results/live/{date}_daily_report.md", render_markdown(report)),
                              ("outputs/ETF轮动策略_今日日报.html", render(report)),
                              ("dashboard/public/ye-daily.html", render(report, public=True))]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return report


def validate_delivery(root: Path, date: str) -> dict:
    """Validate the deliverable, independently of permission to trade."""
    from html.parser import HTMLParser
    report = read_json(root / f"results/live/{date}_daily_report.json")
    plan = read_json(root / f"results/live/{date}_order_plan.json")
    ready = read_json(root / "results/live/readiness_report.json")
    card = read_json(root / f"results/audit/{date}_live_run_card.json")
    manifest_path = root / f"results/audit/{date}_run_manifest.json"
    manifest = read_json(manifest_path)
    for item in (report, plan, ready, card, manifest):
        if item.get("signal_date") != date:
            raise ValueError("交付物日期不一致")
    status = ready.get("status")
    if status not in {"READY", "SELL_ONLY", "BLOCKED"}:
        raise ValueError("交付放行状态不是终态")
    if report.get("status") != status or card.get("release", {}).get("readiness") != status:
        raise ValueError("日报、订单放行与运行卡不一致")
    # Plans intentionally omit ancillary account notes and authorization logs.
    # Compare financial facts, not the shape of those two representations.
    account_fields = ("as_of", "confirmation_status", "positions", "available_cash",
                      "total_equity", "performance", "pending_orders")
    accounts = [item or {} for item in (report.get("account"), plan.get("account_state"), card.get("account_state"))]
    if any(accounts[0].get(key) != account.get(key) for account in accounts[1:] for key in account_fields):
        raise ValueError("日报、计划与运行卡的账户快照不一致")
    if report.get("target_symbol") != plan.get("target_symbol") or report.get("target_symbol") != card.get("decision", {}).get("target_symbol"):
        raise ValueError("日报、计划与运行卡的目标不一致")
    if report.get("orders") != plan.get("execution", {}).get("orders", []):
        raise ValueError("日报与计划的订单不一致")
    if status == "BLOCKED" and (plan.get("actions") or plan.get("execution", {}).get("orders") or report.get("orders")):
        raise ValueError("阻断状态不得留有可执行订单")
    if status == "SELL_ONLY" and (plan.get("target_symbol") is not None or any(x.get("side") != "sell" for x in report["orders"])):
        raise ValueError("仅卖出交付含新增目标")
    ranking = report.get("analysis", {}).get("ranking", [])
    if report.get("analysis", {}).get("status") == "complete":
        if len(ranking) != 51 or len({x["symbol"] for x in ranking}) != 51 or sum(x["pool_role"] == "core" for x in ranking) != 45:
            raise ValueError("完整日报池角色或数量不符")
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.targets = []
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.targets += [v for k, v in attrs if k == "href"]
    for relative in ("outputs/ETF轮动策略_今日日报.html", "dashboard/public/ye-daily.html"):
        path = root / relative
        text = path.read_text(encoding="utf-8")
        if date not in text or "dailyOverview" not in text or "<br" in text:
            raise ValueError("日报日期、正文或格式不合格")
        parser = Links()
        parser.feed(text)
        for target in parser.targets:
            if not (path.parent / target).is_file():
                raise ValueError(f"日报链接不存在：{target}")
    for key in ("critical_files", "source_files", "price_files"):
        for record in manifest.get(key, []):
            if hashlib.sha256((root / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"清单哈希不一致：{record['path']}")
    if card.get("audit", {}).get("run_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("运行卡未绑定当前清单")
    return {"delivery": "PASS", "signal_date": date, "release": status,
            "analysis": report["analysis"]["status"], "daily_report": str(root / "outputs/ETF轮动策略_今日日报.html")}


def delivery_receipt(root: Path, date: str) -> str:
    """Read-only, validated user reply; no XML wrapper or guessed Git status."""
    validate_delivery(root, date)
    report = read_json(root / f"results/live/{date}_daily_report.json")
    v, review, account = report["valuation"], report["review"], report["account"]
    lines = [f"{date} 收盘日报", describe_action(report)]
    if v.get("symbol"):
        lines.append(f"持仓：{v['name']}（{v['symbol']}）{v['quantity']:g}股；今日{money(v['daily_pnl'])}，本次{money(v['purchase_pnl'])}（{pct(v['purchase_return'])}，未计费用）。")
    else:
        lines.append("持仓与收益：见日报可用证据，缺失项待核。")
    held = report["analysis"].get("held")
    if held and number(held.get("rank")) is not None:
        reason = "；".join(report["analysis"].get("exit_reasons", [])) or "未触发卖出或换仓"
        lines.append(f"持仓核心/参考排名第{held['rank']:g}；{reason}。")
    if number(v.get("account_equity")) is not None:
        lines.append(f"账户权益：{v['account_equity']:,.2f}元；累计{pct(v.get('account_return'))}。")
    state = str(account.get("confirmation_status", "未知"))
    if account.get("cash_reconciliation", {}).get("status") == "pending":
        state += "（资金待核，不代表已确认成交未记录）"
    coverage = "未核验" if number(review.get("coverage")) is None else f"{review['coverage']:.0%}"
    lines += [f"对账：{state}；AI审核{review.get('reviewed_count', '—')}/{review.get('input_count', '—')}，覆盖{coverage}。",
              f"订单：{report['status']}；交付检查：PASS。"]
    if report["status"] != "READY":
        lines.append("原因：" + "；".join(report.get("blocking_items", [])))
    links = []
    for label, relative in [("今日日报", "outputs/ETF轮动策略_今日日报.html"),
                            ("回测", "outputs/ETF轮动策略_回测.html"),
                            ("运行卡", f"results/audit/{date}_live_run_card.json")]:
        path = root / relative
        if path.is_file():
            links.append(f"[{label}](<{path.absolute()}>)")
        else:
            lines.append(f"{label}文件缺失，未生成链接。")
    return "\n\n".join(lines + [" · ".join(links)]) + "\n"
