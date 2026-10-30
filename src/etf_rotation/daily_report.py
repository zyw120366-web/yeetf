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
from .evidence import review_context


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
    from scripts.build_live_order_plan import holding_decision, order_candidates, exit_reasons

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
    evidence = review_context(root, date, calendar, config, validator=validate_live_review)
    try:
        news = build(root=root)
        bad_dates = [pd.Timestamp(d) for d, c in evidence["checks"].items() if c["status"] != "complete"]
        news = news.loc[news.date.le(day) & ~news.date.isin(bad_dates)]
    except (ValueError, OSError, KeyError, TypeError) as exc:
        from .sentiment import SENTIMENT_FIELDS
        news = pd.DataFrame(columns=["date", "symbol", *SENTIMENT_FIELDS])
        evidence["status"] = "partial"
        evidence["issues"].append("资讯特征构建失败；价格指标仍单独展示")
        evidence["diagnostic"] = f"{type(exc).__name__}: {exc}"
    sentiment, available = sentiment_matrices_from_frame(news, calendar, symbols)
    names = {symbol_key(x): x["name"] for x in market["universe"]}
    categories = {symbol_key(x): x["category"] for x in market["universe"]}
    bundle, features, eligible, listed, amount, decision = build_ye_signals(
        panel, symbols, categories, config, sentiment, available)
    ranking = ranking_for_day(panel, symbols, names, categories, config, bundle,
                              features, eligible, listed, amount, decision, sentiment, day)
    ranking = apply_live_cooldown(ranking, account, date, calendar,
                                 int(config["enhanced_selection"]["reentry_cooldown_days"]))
    complete = evidence["status"] == "complete"
    if not complete:
        # Keep the exact formal price projection, but NEVER expose fallback
        # decisions computed with absent news as live candidates or protection.
        for field in ("selection_score", "technical_entry_pass", "final_entry_pass",
                      "confirmed_normal_entry", "weak_edge_confirmed", "historical_fallback",
                      "emerging_entry", "quality_extension", "missing_data_soft_exit_protection", "target_weight"):
            ranking[field] = None
        if not evidence["complete"]["soft_exit"] or "diagnostic" in evidence:
            ranking["hot_exit_protection"] = None
            ranking["soft_exit_confirmation"] = None
    candidates = order_candidates(ranking.loc[ranking.final_entry_pass.eq(True)])
    result = {"status": "complete" if complete else "partial", "prices_status": "complete",
              "news_dependencies": evidence, "ranking": ranking.to_dict("records"),
              "candidates": candidates[["symbol", "name", "rank"]].to_dict("records"),
              "market": {"up": int(ranking.change_1d.gt(0).sum()),
                         "down": int(ranking.change_1d.lt(0).sum()),
                         "flat": int(ranking.change_1d.eq(0).sum()),
                         "benchmark_change": panel["close"]["510300.SH"].pct_change(fill_method=None).at[day],
                         "benchmark_above_ma": bool(panel["close"].at[day, "510300.SH"] > panel["close"]["510300.SH"].rolling(120).mean().at[day])}}
    try:
        position = known_position(account, date)
        current = position["symbol"] if position else None
        if complete:
            target, held, reasons, switch = holding_decision(
                day, current, ranking, candidates, account, config, root=root)
            result.update(held=None if held is None else held.to_dict(), exit_reasons=reasons,
                          opportunity_switch=switch, technical_target=target,
                          holding_trusted=True, holding_decision_status="complete")
        else:
            held = ranking.loc[ranking.symbol.eq(current)]
            reasons = []
            if len(held):
                checked = held.iloc[0].copy()
                # Unknown protection cannot approve a soft exit; a known MA
                # break remains visible independent of candidate selection.
                if checked["soft_exit_confirmation"] is None:
                    checked["soft_exit_confirmation"] = False
                reasons = exit_reasons(checked)
            result.update(held=held.iloc[0].to_dict() if len(held) else None,
                          holding_trusted=True, holding_decision_status="unknown",
                          exit_reasons=reasons, opportunity_switch={"status": "unknown"}, technical_target=None)
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
            result["source_files"] = [f"market_data/live_quotes/{date}.json"]
            calendar = pd.read_csv(root / "market_data/prices/510300.SH.csv")["datetime"]
            previous_days = calendar[calendar < date]
            previous = previous_days.iloc[-1] if len(previous_days) else None
            if str(p["opened_on"]) < date and previous:
                try:
                    _, old_close = raw_quote(root, p["symbol"], previous)
                    result.update(daily_pnl=qty * (close - old_close), daily_return=close / old_close - 1,
                                  daily_basis=f"已确认持仓按上一交易日{previous}不复权收盘至今日收盘计算")
                    result["source_files"].append(f"market_data/live_quotes/{previous}.json")
                except (OSError, ValueError, KeyError) as exc:
                    result["daily_limitation"] = f"缺少{previous}有效不复权收盘报价"
            elif str(p["opened_on"]) == date:
                result.update(daily_pnl=result["purchase_pnl"], daily_return=result["purchase_return"],
                              daily_basis="今日新买入，按成交价至收盘计算")
        validate_account(account, date)
        result["account_equity"] = float(account["total_equity"])
        capital = number(account.get("performance", {}).get("net_contributed_capital"))
        if capital is not None and capital > 0:
            result["account_return"] = result["account_equity"] / capital - 1
    except (ValueError, KeyError, OSError, IndexError) as exc:
        result["limitation"] = ("持仓已确认；新增入金、现金及费用待核，仅账户整体收益未知"
                                if result.get("symbol") and account.get("cash_reconciliation", {}).get("status") == "pending"
                                else str(exc))
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
    analysis = {"status": "unavailable", "ranking": [], "candidates": []}
    try:
        analysis = analyze(root, date, account)
    except Exception as exc:
        analysis["error"] = f"{type(exc).__name__}: {exc}"
        analysis["issues"] = ["价格或配置证据不足；已确认持仓与可用估值仍单独展示"]
    # A historical confirmation does not establish today's position if an
    # intervening live order may have filled and has not been reconciled.
    from scripts.validate_live_readiness import previous_execution_is_reconciled
    try:
        position_pending = not previous_execution_is_reconciled(date, root)
    except (OSError, ValueError, KeyError, TypeError):
        position_pending = True
    if position_pending:
        issue = "上一计划成交未对账，当前实际持仓和收益待核"
        status = "BLOCKED"
        reasons = list(dict.fromkeys([*reasons, issue]))
        if analysis.get("ranking"):
            analysis["status"] = "partial"
        analysis.update(holding_trusted=False, holding_decision_status="unknown",
                        held=None, exit_reasons=[], opportunity_switch={"status": "unknown"},
                        technical_target=None, candidates=[])
        for row in analysis.get("ranking", []):
            row["final_entry_pass"] = None
            row["live_cooldown_blocked"] = None
        deps = analysis.setdefault("news_dependencies", {})
        deps["issues"] = list(dict.fromkeys([*deps.get("issues", []), issue]))
    checked = analysis.get("news_dependencies", {}).get("checks", {}).get(date)
    try:
        if checked:
            if checked["status"] != "complete":
                raise ValueError(checked["message"])
            review = checked["review"]
        else:
            review = validate_live_review(root, date)
            review = {k: review.get(k) for k in ("status", "input_count", "reviewed_count", "coverage", "snapshot_hash")}
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        review = {"status": "unverified", "coverage": None, "error": str(exc)}
    if status in {"READY", "SELL_ONLY"}:
        try:
            if status == "READY" and plan.get("execution", {}).get("funding_scope") == "sale_proceeds_only":
                from .rotation_execution import validate_proceeds_plan
                validate_proceeds_plan(account, plan, date)
                if (analysis.get("status") != "complete" or analysis.get("technical_target") != plan.get("target_symbol")
                        or not analysis.get("exit_reasons")):
                    raise ValueError("卖出款换仓缺少完整正式买入及退出证据")
            elif status == "SELL_ONLY":
                from .live import validate_sell_position
                validate_sell_position(account, date)
            else:
                validate_account(account, date)
            if status == "READY" and review["status"] != "complete":
                raise ValueError("完整计划缺少有效资讯审核")
        except ValueError as exc:
            status, reasons = "BLOCKED", [str(exc)]
    snapshot = read_json(root / f"market_data/sentiment/{date}.json")
    sources = {k: {"ok": v.get("ok"), "rows": len(v.get("rows") or [])}
               for k, v in snapshot.get("sources", {}).items()}
    dde_available = any(number(row.get("dde_net", row.get("ddejingliang"))) is not None
                        for source in snapshot.get("sources", {}).values()
                        for row in source.get("rows", []) or [])
    # Never show an order from a failed or mismatched release as executable.
    orders = plan.get("execution", {}).get("orders", []) if status in {"READY", "SELL_ONLY"} else []
    if status == "SELL_ONLY" and any(x.get("side") != "sell" for x in orders):
        status, reasons, orders = "BLOCKED", ["仅卖出状态中出现非卖出订单"], []
    unknown_valuation = {"account_return": None, "account_equity": None, "purchase_pnl": None,
                         "purchase_return": None, "daily_pnl": None, "daily_return": None}
    return clean({"schema_version": 1, "signal_date": date, "status": status,
                  "blocking_items": reasons, "account": account,
                  "position_reconciliation_required": position_pending,
                  "valuation": unknown_valuation if position_pending else valuation(root, date, account),
                  "analysis": analysis, "review": review, "sources": sources,
                  "news_data_quality": {"dde_available": dde_available},
                  "backtest_as_of": read_json(root / "results/comparison/latest_signals.json").get("signal_date"),
                  "orders": orders, "actions": plan.get("actions", []) if status != "BLOCKED" else [],
                  "price_risk_check": [] if position_pending else price_risk_check(root, date, account, ranking=analysis.get("ranking")),
                  "available_links": [p for p in ["outputs/ETF轮动策略_回测.html", "outputs/ETF轮动策略_策略与回测.html",
                                      f"market_data/sentiment/ai_review/{date}.json"] if (root / p).is_file()],
                  "target_symbol": plan.get("target_symbol") if status != "BLOCKED" else None,
                  "target_is_executable": status in {"READY", "SELL_ONLY"},
                  "funding_scope": plan.get("execution", {}).get("funding_scope"),
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
            detail = "；".join(a.get("exit_reasons", []))
            if a.get("holding_decision_status") == "unknown":
                detail = detail + "；换仓待核" if detail else "完整退出与换仓结论待核"
            else:
                detail = detail or "未触发卖出或换仓，观察持有"
            return f"持仓观察：{held['name']}，{detail}。正式订单未放行（买卖均禁止），暂无可执行买入计划。"
        return "正式订单未放行，暂无可执行买入计划；持仓结论见可用证据。"
    actions = report["actions"]
    labels = {"buy": "买入", "sell": "卖出", "hold": "持有"}
    names = {x["symbol"]: x["name"] for x in a.get("ranking", [])}
    descriptions = []
    for action in actions:
        label = labels.get(action.get("side"), "待核")
        name = names.get(action.get("symbol"), action.get("symbol") or "现金")
        weight = number(action.get("target_weight"))
        allocation = ("（仅使用本次卖出净款）" if action.get("side") == "buy" and report.get("funding_scope") == "sale_proceeds_only"
                      else f"（目标仓位{weight:.0%}）" if weight is not None else "")
        descriptions.append(f"{label} {name}" + allocation)
    text = "；".join(descriptions)
    suffix = ("；仅卖出转现金，禁止买入。" if status == "SELL_ONLY" else
              "；旧仓全部成交后，仅用卖出净款换仓；买入数量开盘重算，历史现金仍待核。" if report.get("funding_scope") == "sale_proceeds_only" else "。计划不是成交。")
    return "次日开盘：" + (text or "待核") + suffix


def sections(report: dict):
    a, v, review = report["analysis"], report["valuation"], report["review"]
    overview = [describe_action(report)]
    if report.get("position_reconciliation_required"):
        positions = report.get("account", {}).get("positions", [])
        if len(positions) == 1:
            p = positions[0]
            as_of = str(report["account"].get("as_of", "未知")).split("_")[0]
            overview.append(f"账户最后确认记录：{as_of} {p.get('name', p.get('symbol'))} {p.get('quantity')}股；上一计划成交未核，今天实际持仓和收益待核。")
        else:
            overview.append("上一计划成交未核，今天实际持仓和收益待核。")
    if v.get("symbol"):
        overview.append(f"{v['name']} {v['quantity']:g}股，收盘{v['close']:.3f}元；今日持仓{money(v['daily_pnl'])}，本次持仓{money(v['purchase_pnl'])}（{pct(v['purchase_return'])}，未计费用）。")
    held = a.get("held")
    if held:
        overview.append(f"持仓核心/参考排名第{held['rank']:g}，ROC20 {pct(held['roc20'])}、ROC60 {pct(held['roc60'])}，MA120乖离{pct(held['ma120_bias'])}。")
    if a.get("market"):
        m = a["market"]
        overview.append(f"全池{m['up']}涨、{m['down']}跌、{m['flat']}平；沪深300ETF {pct(m['benchmark_change'])}，在MA120{'上' if m['benchmark_above_ma'] else '下'}方。市场判断只作背景，不新增择时开关。")
    if report["status"] == "BLOCKED":
        overview.append("尚需核清：" + "；".join(x.rstrip("。；") for x in report["blocking_items"]) + "。")
    limitations = a.get("news_dependencies", {}).get("issues", []) or a.get("issues", [])
    if limitations:
        suffix = ("。价格排名可看；实际持仓和操作待核。" if report.get("position_reconciliation_required") else
                  "。价格排名可看；依赖这些资讯的判断显示待核，不当作不通过。" if a.get("ranking") else "。不沿用旧日结论。")
        overview.append("待核事项：" + "；".join(limitations) + suffix)
    ranking = a.get("ranking", [])
    def names(rows):
        return "、".join(f"{r['name']}（{r['symbol']}）" for r in rows) or "无"
    if ranking:
        core_top = [r for r in ranking if r["pool_role"] == "core" and number(r["rank"]) is not None and r["rank"] <= 5]
        tech = [r for r in ranking if r["technical_entry_pass"]]
        final = [r for r in ranking if r["final_entry_pass"]]
        blocked = [r for r in ranking if r.get("live_cooldown_blocked")]
        price_pass = [r for r in ranking if r["base_entry_pass"]]
        funnel = [f"固定池{len(ranking)}只：45核心＋6卫星；基础资格{sum(bool(r['pool_eligible']) for r in ranking)}只通过。",
                  "核心前5：" + names(core_top),
                  f"常规价格条件通过{len(price_pass)}只（尚非买入结论）：" + names(price_pass)]
        if a["status"] == "complete":
            funnel += [f"技术与资讯门槛通过{len(tech)}只：" + names(tech),
                       f"真实冷却与核心优先后{len(final)}只：" + names(final)]
        elif report.get("position_reconciliation_required"):
            funnel += ["今日资讯已单独列示；上一计划未对账，真实冷却和最终候选待核。"]
        else:
            funnel += ["资讯确认、新趋势记忆及最终候选待核；不能把未知候选写成0只，也不能据此买入。"]
        cooldown_text = "真实卖出冷却排除：待核" if report.get("position_reconciliation_required") else "真实卖出冷却排除：" + names(blocked)
        funnel += [cooldown_text, "候选合格不等于买入，仍须检查旧仓退出与换仓，并通过订单核验。"]
        satellites = [r for r in ranking if r["pool_role"] == "challenger"]
        satellite = ["6只卫星：" + names(satellites),
                     "卫星常规价格条件通过：" + names([r for r in satellites if r["base_entry_pass"]])]
        satellite += (["卫星技术合格：" + names([r for r in satellites if r["technical_entry_pass"]]),
                       "卫星最终补位：" + names([r for r in satellites if r["final_entry_pass"]])]
                      if a["status"] == "complete" else
                      ["上一计划未对账，卫星最终补位待核，不生成新增订单。"] if report.get("position_reconciliation_required") else
                      ["卫星资讯确认及最终补位待核，不生成新增订单。"])
    else:
        funnel = ["全池价格筛选暂不可确认，缺少有效价格或配置证据。具体诊断已保存在运行卡，不以旧日排名代替。"]
        satellite = ["今日卫星条件未知，不生成新增订单。"]
    checks = []
    if report.get("position_reconciliation_required"):
        checks = ["上一计划可能已成交；当前持仓、持有、卖出和换仓判断均待核，不能以旧持仓替代。"]
    elif held and a.get("holding_decision_status") != "unknown":
        switch = a.get("opportunity_switch", {})
        checks = [f"持有：{held['name']}；收盘在MA120{'上' if held['above_ma120'] else '下'}方。",
                  "退出/让位：" + ("；".join(a.get("exit_reasons", [])) or "未触发"),
                  f"机会换仓：{switch.get('confirmation_streak', 0)}/{switch.get('required_confirmation_days', 2)}，持有{switch.get('held_trading_days', '—')}个交易日；候选分差{pct(switch.get('score_gap'))}。"]
    else:
        checks = ([f"持仓事实：{v['name']} {v['quantity']:g}股已确认；分析不完整不表示没有持仓。"]
                  if v.get("symbol") else ["持仓事实尚未核清。"])
        checks += report.get("price_risk_check", [])
        soft_known = a.get("news_dependencies", {}).get("complete", {}).get("soft_exit") and "diagnostic" not in a.get("news_dependencies", {})
        checks += [("卖出：" + ("；".join(a.get("exit_reasons", [])) or "已核验的退出条件未触发") + "；是否可执行仍以订单放行为准。" if soft_known else "卖出：软退出的近期资讯保护待核，不能把未知当作保护不存在；价格硬退出仍由独立核验处理。"),
                   "换仓：完整合格候选和连续确认待核，不累计未经核验的确认天数。"]
    coverage = "待核" if number(review.get("coverage")) is None else f"{review['coverage']:.0%}"
    news = [f"当日逐条审核：{review.get('reviewed_count', '—')}/{review.get('input_count', '—')}，覆盖{coverage}。",
            "；".join(f"{k}：{v['rows']}条，{'成功' if v['ok'] else '失败'}" for k, v in report["sources"].items()),
            "审核覆盖是对已采集条目的覆盖，不是全网新闻完整性；DDE缺失不能当作已证实的零资金流。"]
    if held:
        dde = pct(held.get("sentiment_positive_dde_share")) if report.get("news_data_quality", {}).get("dde_available") else "未知（来源未提供）"
        news.append(f"持仓直接匹配数量：{held.get('sentiment_matched_count', '—')}；正DDE占比：{dde}。不把泛行业题材扩散到不相关ETF。")
    news_issues = [x for x in limitations if x != "上一计划成交未对账，当前实际持仓和收益待核"]
    if news_issues:
        news.append("近期连续证据：" + "；".join(news_issues))
    return [("今日决策路径综述", overview), ("今日筛选漏斗", funnel),
            ("持有、卖出与换仓", checks), ("今日卫星检查", satellite), ("资讯审核", news)]


def render(report: dict, *, public=False) -> str:
    """Present the verified decision first; evidence remains available on demand."""
    esc = html.escape
    date = report["signal_date"]
    root = "../../" if public else "../"
    a, v = report["analysis"], report["valuation"]
    ranking = a.get("ranking", [])
    names = {row["symbol"]: row["name"] for row in ranking}
    def display(rows):
        return "、".join(f"{row['name']}（{row['symbol']}）" for row in rows) or "无"
    def gate(value):
        return "待核" if value is None else "通过" if value else "未通过"

    status = report["status"]
    if status == "BLOCKED":
        action_title = "暂不下单"
        action_text = "当前没有放行的买卖订单。"
    else:
        action_title = "下一交易日计划"
        action_text = describe_action(report)
    order_lines = []
    visible_orders = report.get("orders", []) if status in {"READY", "SELL_ONLY"} else []
    if status == "SELL_ONLY":
        visible_orders = [order for order in visible_orders if order.get("side") == "sell"]
    for order in visible_orders:
        side = {"buy": "买入", "sell": "卖出"}.get(order.get("side"), "待核")
        symbol = order.get("symbol") or ""
        quantity = order.get("confirmed_quantity", order.get("buy_estimate", {}).get("estimated_quantity_at_last_close"))
        quantity_text = "开盘核算" if quantity is None else f"{quantity:g}股"
        quantity_text = ("数量参考：" + quantity_text + "（非成交数量）"
                         if order.get("side") == "buy" else "计划数量：" + quantity_text)
        order_lines.append(f"{side} {names.get(symbol, symbol)}（{symbol}）：{quantity_text}。{order.get('instruction', '')}")
    reasons = [str(item).rstrip("。；") for item in report.get("blocking_items", []) if item]
    if status == "BLOCKED":
        short_reasons = []
        for reason in reasons:
            if "上一计划成交" in reason or "上一笔" in reason and "成交" in reason:
                summary = "上一笔计划的实际成交未核对"
            elif "账户状态未确认" in reason:
                summary = "账户持仓或资金未核清"
            else:
                summary = reason
            if summary not in short_reasons:
                short_reasons.append(summary)
        action_text += "原因：" + ("；".join(short_reasons) if short_reasons else "账户或数据尚未核清") + "。"

    account = report.get("account", {})
    positions = account.get("positions") or []
    as_of = str(account.get("as_of", ""))[:10]
    position = positions[0] if len(positions) == 1 and as_of and as_of <= date else None
    if position:
        label = "用户报告持仓" if "user_report" in str(account.get("as_of", "")) else "账户记录"
        holding = f"{label}（{as_of}）：{position.get('name', position.get('symbol'))}（{position.get('symbol')}）{position.get('quantity')}股，成本{position.get('average_cost')}元。"
        if report.get("position_reconciliation_required"):
            holding += "上一计划未对账，不能据此确认完整账户或执行订单。"
    elif not positions and account.get("confirmation_status") in {"confirmed", "assumed_authorized"} and not report.get("position_reconciliation_required"):
        holding = "账户记录：空仓。"
    else:
        holding = "当前持仓待核；不使用历史或回测持仓代替。"

    held = a.get("held")
    if held and a.get("holding_decision_status") == "complete":
        why = "持仓：" + ("；".join(a.get("exit_reasons", [])) or "未触发卖出条件") + "。"
        switch = a.get("opportunity_switch") or {}
        if switch.get("enabled"):
            streak, required = switch.get("confirmation_streak"), switch.get("required_confirmation_days")
            why += (f" 换仓确认：{streak}/{required}。" if streak is not None and required is not None else " 换仓确认待核。")
    else:
        why = "持仓的卖出和换仓判断待核。"
    market = a.get("market") or {}
    market_text = (f"市场：{market['up']}涨、{market['down']}跌、{market['flat']}平；沪深300ETF {pct(market.get('benchmark_change'))}。"
                   if all(key in market for key in ("up", "down", "flat")) else "市场数据待核。")
    core_top = [row for row in ranking if row.get("pool_role") == "core" and number(row.get("rank")) is not None and row["rank"] <= 5]
    price_pass = [row for row in ranking if row.get("base_entry_pass") is True]
    technical_pass = [row for row in ranking if row.get("technical_entry_pass") is True]
    final_pass = [row for row in ranking if row.get("final_entry_pass") is True]
    if not ranking:
        funnel = ["全池筛选数据待核。"]
    else:
        funnel = ["核心排名前5：" + display(core_top),
                  f"常规价格条件通过 {len(price_pass)} 只：" + display(price_pass)]
        if a.get("status") == "complete":
            funnel.append(f"账户及优先级筛选后 {len(final_pass)} 只：" + display(final_pass))
        elif all(row.get("technical_entry_pass") is not None for row in ranking):
            funnel.append(f"技术条件通过 {len(technical_pass)} 只：" + display(technical_pass) + "；最终候选待核，不能当作买入指令。")
        else:
            funnel.append("完整候选待核；不能把缺失信息当作零或买入信号。")
    satellites = [row for row in ranking if row.get("pool_role") == "challenger"]
    if satellites:
        satellite_pass = [row for row in satellites if row.get("base_entry_pass") is True]
        funnel.append("卫星常规价格条件：" + (display(satellite_pass) if satellite_pass else "无通过"))

    review = report.get("review", {})
    coverage = review.get("coverage")
    review_text = (f"资讯审核 {review.get('reviewed_count', '—')}/{review.get('input_count', '—')}，覆盖 {coverage:.0%}（仅指已采集资讯）。"
                   if number(coverage) is not None else "资讯审核未完成或未核验。")
    if not report.get("news_data_quality", {}).get("dde_available"):
        review_text += " DDE数据未提供。"

    links = report.get("available_links", [])
    navigation = []
    for target, label in [("outputs/ETF轮动策略_回测.html", "回测"), ("outputs/ETF轮动策略_策略与回测.html", "策略介绍")]:
        if target in links:
            if label == "回测" and report.get("backtest_as_of"):
                label += f"（截至{report['backtest_as_of']}）"
            navigation.append(f'<a href="{root}{target}">{esc(label)}</a>')
    navigation.append(f'<a href="{root}results/audit/{date}_live_run_card.json">运行卡</a>')
    if f"market_data/sentiment/ai_review/{date}.json" in links:
        navigation.append(f'<a href="{root}market_data/sentiment/ai_review/{date}.json">逐条审核证据</a>')
    navigation.append(f'<a href="{root}results/live/{date}_daily_report.json">日报数据</a>')
    rows = []
    for row in ranking:
        cells = [row["name"] + "（" + row["symbol"] + "）", "核心" if row["pool_role"] == "core" else "卫星",
                 str(row["rank"]), pct(row["change_1d"]), pct(row["momentum_score"]), pct(row["roc20"]), pct(row["roc60"]), pct(row["ma120_bias"]),
                 gate(row["pool_eligible"]), gate(row["normal_entry"]), gate(row["emerging_entry"]), gate(row["quality_extension"]),
                 "冷却排除" if row.get("live_cooldown_blocked") else "待核" if row["final_entry_pass"] is None else "观察候选" if row["final_entry_pass"] else "未入选"]
        rows.append("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in cells) + "</tr>")
    headers = ["ETF", "角色", "排名", "今日", "动量分", "ROC20", "ROC60", "MA120乖离", "资格", "常规价格", "新趋势", "延伸", "筛选结果"]
    table = '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{esc(x)}</th>" for x in headers) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    detail_sections = sections(report)
    audit_detail = "".join(f'<h3>{esc(title)}</h3>' + "".join(f'<p>{esc(line)}</p>' for line in paragraphs if line)
                           for title, paragraphs in detail_sections)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ye 策略日报 · {esc(date)}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f8;color:#172337;font:16px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1040px;margin:auto;padding:32px 24px 64px}}h1{{font-size:28px;margin:0}}h2{{font-size:21px;margin:0 0 14px}}h3{{font-size:16px;margin:24px 0 8px}}p{{margin:8px 0}}.top{{display:flex;justify-content:space-between;align-items:end;gap:16px;margin-bottom:22px}}.date,.muted,small{{color:#64748b}}.nav a,a{{color:#345fb1}}.nav{{font-size:14px;display:flex;gap:16px;flex-wrap:wrap}}.card{{background:#fff;border:1px solid #e3e8ef;border-radius:16px;padding:23px;margin:16px 0;box-shadow:0 3px 14px #23314d08}}.decision{{background:#172b47;color:#fff;border:0;padding:28px}}.decision .eyebrow{{color:#bbcadb}}.decision h2{{font-size:29px;margin:4px 0 12px}}.decision p{{color:#e2e9f3}}.decision.warning{{background:#493523}}.holding{{border-top:1px solid #ffffff33;padding-top:13px;margin-top:16px}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.stat{{background:#fff;border:1px solid #e3e8ef;border-radius:14px;padding:18px}}.stat span,.stat small{{display:block}}.stat strong{{display:block;font-size:24px;margin:3px 0}}.points{{display:grid;gap:8px}}.point{{background:#f5f7fa;border-radius:9px;padding:11px 14px}}.note{{font-size:14px;color:#64748b}}details.card summary{{cursor:pointer;font-weight:650;font-size:17px}}details.card[open] summary{{margin-bottom:15px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:9px;border-bottom:1px solid #e5eaf2;white-space:nowrap;text-align:left}}th{{background:#edf2f8}}@media(max-width:720px){{main{{padding:20px 14px 40px}}.top{{align-items:start;flex-direction:column}}.stats{{grid-template-columns:1fr}}.decision h2{{font-size:25px}}}}
</style></head><body><main>
<header class="top"><div><h1>ye 策略日报</h1><div class="date">{esc(date)} 收盘 · 下一交易日计划</div></div><nav class="nav">{' '.join(navigation)}</nav></header>
<section class="card decision{' warning' if status == 'BLOCKED' else ''}" id="dailyOverview"><div class="eyebrow">先看怎么做</div><h2>{esc(action_title)}</h2><p>{esc(action_text)}</p>{''.join(f'<p>{esc(line)}</p>' for line in order_lines)}<p class="holding">{esc(holding)}</p></section>
<div class="stats"><div class="stat"><span>今日收益</span><strong>{money(v.get('daily_pnl'))}</strong><small>{pct(v.get('daily_return'))} · 持仓口径</small></div><div class="stat"><span>本次买入收益</span><strong>{money(v.get('purchase_pnl'))}</strong><small>{pct(v.get('purchase_return'))} · 未计费用</small></div><div class="stat"><span>策略开启以来</span><strong>{pct(v.get('account_return'))}</strong><small>账户资金未核时不计算</small></div></div>
<section class="card"><h2>今天为什么</h2><p>{esc(why)}</p><p>{esc(market_text)}</p><p class="note">{esc(review_text)}</p></section>
<section class="card"><h2>今天筛选出了什么</h2><div class="points">{''.join(f'<div class="point">{esc(line)}</div>' for line in funnel)}</div><p class="note">候选不等于订单；先确认持仓和放行状态。</p></section>
<details class="card"><summary>查看全部ETF筛选明细</summary>{table}</details>
<details class="card"><summary>查看审核与决策依据</summary>{audit_detail}</details>
</main></body></html>'''


def render_markdown(report: dict) -> str:
    lines = [f"# ye 策略日报｜{report['signal_date']}"]
    for title, paragraphs in sections(report):
        lines += ["", "## " + title, "", "\n\n".join(p for p in paragraphs if p)]
    lines += ["", "## 放行", "", report["status"] + "；计划不是成交。"]
    return "\n".join(lines) + "\n"


def publish(root: Path, date: str, *, report: dict | None = None) -> dict:
    report = collect(root, date) if report is None else report
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


def validate_delivery(root: Path, date: str, *, deep: bool = False) -> dict:
    """Validate decisions and delivery; legacy whole-tree hashes are opt-in."""
    from .run_record import delivery_files
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
    analysis = report.get("analysis", {})
    if analysis.get("status") in {"complete", "partial"}:
        if len(ranking) != 51 or len({x["symbol"] for x in ranking}) != 51 or sum(x["pool_role"] == "core" for x in ranking) != 45:
            raise ValueError("完整日报池角色或数量不符")
    if status == "READY" and analysis.get("status") != "complete":
        raise ValueError("完整放行不可使用部分分析")
    if analysis.get("status") == "partial":
        if analysis.get("candidates") or analysis.get("technical_target") is not None or any(r.get("final_entry_pass") is not None for r in ranking):
            raise ValueError("部分资讯分析不得伪装成完整候选")
    required_evidence = (report.get("valuation", {}).get("source_files", [])
                         + analysis.get("news_dependencies", {}).get("files", []))
    bound = {r["path"] for key in ("critical_files", "evidence_files", "price_files") for r in manifest.get(key, [])}
    if not set(required_evidence).issubset(bound):
        raise ValueError("估值或资讯依赖未纳入清单")
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
        if analysis.get("status") in {"complete", "partial"} and any(r["symbol"] not in text for r in ranking):
            raise ValueError("日报未显示完整51池价格明细")
        parser = Links()
        parser.feed(text)
        for target in parser.targets:
            if not (path.parent / target).is_file():
                raise ValueError(f"日报链接不存在：{target}")
    records = {r["path"]: r for key in ("critical_files", "source_files", "price_files") for r in manifest.get(key, [])}
    if not set(delivery_files(date)).issubset(records):
        raise ValueError("清单缺少当日计划或日报绑定")
    # Code changes and newly appended price bars do not invalidate yesterday's
    # frozen decision. Deep verification only makes sense in its Git snapshot.
    checked = list(records) if deep else delivery_files(date)
    for relative in checked:
        expected = records[relative].get("sha256")
        if expected is not None and hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"清单哈希不一致：{relative}")
        if relative in delivery_files(date) and expected is None:
            raise ValueError(f"当日交付缺少绑定：{relative}")
    if card.get("audit", {}).get("run_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("运行卡未绑定当前清单")
    return {"delivery": "PASS", "signal_date": date, "release": status,
            "analysis": report["analysis"]["status"], "daily_report": str(root / "outputs/ETF轮动策略_今日日报.html")}


def delivery_receipt(root: Path, date: str, *, deep: bool = False) -> str:
    """Read-only, validated user reply; no XML wrapper or guessed Git status."""
    validate_delivery(root, date, deep=deep)
    report = read_json(root / f"results/live/{date}_daily_report.json")
    v, review, account = report["valuation"], report["review"], report["account"]
    lines = [f"{date} 收盘日报", describe_action(report)]
    if v.get("symbol"):
        lines.append(f"持仓：{v['name']}（{v['symbol']}）{v['quantity']:g}股；今日{money(v['daily_pnl'])}，本次{money(v['purchase_pnl'])}（{pct(v['purchase_return'])}，未计费用）。")
    else:
        lines.append("持仓与收益：见日报可用证据，缺失项待核。")
    held = report["analysis"].get("held")
    if held and number(held.get("rank")) is not None:
        reason = ("完整退出/换仓结论待核" if report["analysis"].get("holding_decision_status") == "unknown"
                  else "；".join(report["analysis"].get("exit_reasons", [])) or "未触发卖出或换仓")
        lines.append(f"持仓核心/参考排名第{held['rank']:g}；{reason}。")
    if report["analysis"].get("status") == "partial":
        prefix = "对账或数据待核：" if report.get("position_reconciliation_required") else "连续资讯缺口："
        lines.append(prefix + "；".join(report["analysis"]["news_dependencies"]["issues"]))
        lines.extend(report.get("price_risk_check", []))
    if number(v.get("account_equity")) is not None:
        lines.append(f"账户权益：{v['account_equity']:,.2f}元；累计{pct(v.get('account_return'))}。")
    state = str(account.get("confirmation_status", "未知"))
    if report.get("position_reconciliation_required"):
        state += "（上一计划成交未核，当前持仓待核）"
    elif account.get("cash_reconciliation", {}).get("status") == "pending":
        state += "（资金待核，不代表已确认成交未记录）"
    coverage = "未核验" if number(review.get("coverage")) is None else f"{review['coverage']:.0%}"
    lines += [f"对账：{state}；AI审核{review.get('reviewed_count', '—')}/{review.get('input_count', '—')}，覆盖{coverage}。",
              f"订单：{report['status']}；交付检查：PASS。"]
    if report["status"] != "READY":
        lines.append("原因：" + "；".join(x.rstrip("。；") for x in report.get("blocking_items", [])))
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
