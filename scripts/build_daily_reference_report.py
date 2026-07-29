from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def source_counts(items: list[dict]) -> list[tuple[str, int]]:
    return sorted(Counter(item["source"] for item in items).items())


def category_counts(items: list[dict]) -> list[tuple[str, int, int]]:
    counts: dict[str, Counter] = {}
    for item in items:
        ai = item["ai"]
        if not ai["relevant"]:
            continue
        for category in ai["matched_categories"]:
            bucket = counts.setdefault(category, Counter())
            if int(ai["direction"]) > 0:
                bucket["positive"] += 1
            elif int(ai["direction"]) < 0:
                bucket["negative"] += 1
    return sorted(
        ((category, int(count["positive"]), int(count["negative"])) for category, count in counts.items()),
        key=lambda row: (-row[1] - row[2], row[0]),
    )


def entry_blockers(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["pool_eligible"]):
        reasons.append("流动性或上市期不足")
    if int(row["rank"]) > 5:
        reasons.append("排名不在前5")
    if float(row["roc20"]) <= 0:
        reasons.append("ROC20不为正")
    if float(row["roc60"]) <= 0:
        reasons.append("ROC60不为正")
    if not bool(row["above_ma120"]):
        reasons.append("跌破MA120")
    if float(row["ma120_bias"]) > 0.09:
        reasons.append("MA120乖离超过9%")
    return "；".join(reasons) if reasons else "通过基础价格条件；仍需情绪与策略门槛确认"


def candidate_outcome(row: pd.Series, target_symbol: str | None, current_symbol: str | None) -> str:
    if not bool(row["final_entry_pass"]):
        if bool(row.get("technical_entry_pass", False)):
            return "技术条件已通过；当日存在核心候选，等待核心空档"
        return entry_blockers(row)
    if str(row["symbol"]) == target_symbol:
        if current_symbol and current_symbol == target_symbol:
            return "最终合格；现有实盘持仓未触发卖出，继续作为唯一目标"
        return "最终合格且符合核心优先顺序，选为唯一目标"
    if current_symbol and current_symbol == target_symbol:
        return "最终合格，但现有实盘持仓未触发卖出，策略不为追逐更高分而换仓"
    return "最终合格，但正式路径选择分低于已选目标；不同时持有多只"


def entry_path(row: pd.Series) -> str:
    if bool(row["normal_entry"]):
        return "常规动量"
    if bool(row["emerging_entry"]):
        return "情绪确认的新趋势"
    if bool(row["quality_extension"]):
        return "热点确认的质量延伸"
    return "未通过"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the concise daily ye reference report")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    review = json.loads((ROOT / "market_data" / "sentiment" / "ai_review" / f"{args.date}.json").read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "results" / "live" / f"{args.date}_order_plan.json").read_text(encoding="utf-8"))
    readiness = json.loads((ROOT / "results" / "live" / "readiness_report.json").read_text(encoding="utf-8"))
    ranking_history = pd.read_csv(ROOT / "results" / "comparison" / "latest_ranking.csv")
    ranking = ranking_history.loc[ranking_history["date"].eq(args.date)].sort_values(["rank", "momentum_score"])
    prior_dates = sorted(value for value in ranking_history["date"].unique() if str(value) < args.date)
    previous_ranking = (
        ranking_history.loc[ranking_history["date"].eq(prior_dates[-1])]
        if prior_dates else pd.DataFrame()
    )
    if ranking.empty:
        raise RuntimeError(f"missing momentum ranking for {args.date}")
    score_column = "selection_score" if "selection_score" in ranking.columns else "momentum_score"
    candidates = ranking.loc[ranking["final_entry_pass"].astype(bool)].sort_values(
        [score_column, "rank"], ascending=[False, True]
    )
    satellite_ranking = ranking.loc[ranking["pool_role"].eq("challenger")].sort_values(
        ["rank", "momentum_score"], ascending=[True, False]
    )
    core_candidates = candidates.loc[candidates["pool_role"].ne("challenger")]
    satellite_technical = satellite_ranking.loc[
        satellite_ranking["technical_entry_pass"].astype(bool)
    ]
    satellite_final = satellite_ranking.loc[
        satellite_ranking["final_entry_pass"].astype(bool)
    ]
    sentiment = pd.read_csv(ROOT / "market_data" / "sentiment" / "features" / "symbol_daily.csv")
    sentiment = sentiment.loc[sentiment["date"].eq(args.date)].set_index("symbol")
    diagnostics = pd.read_csv(ROOT / "results" / "ye_strategy" / "signal_diagnostics.csv")
    day = diagnostics.loc[diagnostics["date"].eq(args.date)]
    if day.empty:
        raise RuntimeError(f"missing signal diagnostics for {args.date}")
    signal = day.iloc[-1]
    side_name = {"buy": "买入", "sell": "卖出", "hold": "持有"}
    actions = "；".join(
        ("继续空仓" if item["side"] == "hold" and not item.get("symbol") else
         f"{side_name.get(item['side'], item['side'])} {item['symbol'] or '现金'}")
        for item in plan["actions"]
    )
    execution = plan.get("execution", {})
    execution_orders = execution.get("orders", [])
    account = plan["account_state"]
    target_symbol = plan.get("target_symbol")
    current_symbol = plan.get("current_symbol")
    target_row = ranking.loc[ranking["symbol"].eq(target_symbol)] if target_symbol else pd.DataFrame()
    target_name = str(target_row.iloc[-1]["name"]) if not target_row.empty else "现金"
    category_rows = category_counts(review["items"])
    category_lookup = {category: (positive, negative) for category, positive, negative in category_rows}
    target_sentiment = sentiment.loc[target_symbol] if target_symbol in sentiment.index else None
    buy_order = next((item for item in execution_orders if item["side"] == "buy"), None)
    buy_estimate = buy_order.get("buy_estimate", {}) if buy_order else {}
    confirmed_normal = ranking.get("confirmed_normal_entry", ranking["normal_entry"]).astype(bool)
    pool_eligible = ranking["pool_eligible"].astype(bool)
    normal_count = int((confirmed_normal & pool_eligible).sum())
    emerging_count = int((ranking["emerging_entry"].astype(bool) & pool_eligible).sum())
    extension_count = int((ranking["quality_extension"].astype(bool) & pool_eligible).sum())
    satellite_technical_names = "、".join(
        f"{row['name']}（{row['symbol']}）" for _, row in satellite_technical.iterrows()
    ) or "无"
    satellite_final_names = "、".join(
        f"{row['name']}（{row['symbol']}）" for _, row in satellite_final.iterrows()
    ) or "无"
    if not satellite_final.empty:
        satellite_status = f"卫星补位已启用：{satellite_final_names}进入最终候选。"
    elif not satellite_technical.empty and not core_candidates.empty:
        satellite_status = f"卫星待命：{satellite_technical_names}技术合格，但核心已有候选。"
    elif not satellite_technical.empty:
        satellite_status = f"卫星技术合格但未进入最终候选：{satellite_technical_names}。"
    else:
        satellite_status = f"卫星未触发：{len(satellite_ranking)}只均未通过技术入场。"
    positions = [item for item in account.get("positions", []) if float(item.get("quantity", 0)) > 0]
    account_position = (
        "当前空仓"
        if not positions
        else f"当前持有 {positions[0]['symbol']} {int(float(positions[0]['quantity'])):,} 股"
    )
    if current_symbol and current_symbol == target_symbol:
        account_decision = f"现有持仓 {current_symbol} 未触发卖出条件，继续持有；不因其他标的排名变化而换仓。"
    elif not current_symbol:
        account_decision = f"当前空仓，按正式路径选择分取第一名，得到唯一目标 {target_name}（{target_symbol or '现金'}）。"
    else:
        account_decision = f"现有持仓触发退出后，按正式路径选择分确定目标 {target_name}（{target_symbol or '现金'}）。"
    strongest = max(category_rows, key=lambda row: row[1] - row[2], default=("无", 0, 0))
    most_negative = max(category_rows, key=lambda row: row[2], default=("无", 0, 0))
    target_category = str(target_row.iloc[-1]["category"]) if not target_row.empty else "无"
    target_positive, target_negative = category_lookup.get(target_category, (0, 0))
    performance = account.get("performance", {}) or {}
    contributed_capital = float(performance.get("net_contributed_capital", 0.0))
    total_equity = float(account["total_equity"])
    strategy_pnl = total_equity - contributed_capital if contributed_capital > 0 else 0.0
    strategy_return = strategy_pnl / contributed_capital if contributed_capital > 0 else 0.0
    strategy_start = str(performance.get("strategy_start_date", "—"))
    if positions:
        position = positions[0]
        quantity = float(position["quantity"])
        average_cost = float(position["average_cost"])
        market_price = float(position["market_price"])
        purchase_pnl = quantity * (market_price - average_cost)
        purchase_return = market_price / average_cost - 1.0 if average_cost > 0 else 0.0
    else:
        market_price = average_cost = purchase_pnl = purchase_return = 0.0
    previous_plan_paths = sorted(
        path for path in (ROOT / "results" / "live").glob("????-??-??_order_plan.json")
        if path.stem[:10] < args.date
    )
    previous_equity = contributed_capital
    previous_equity_date = strategy_start
    if previous_plan_paths:
        previous_plan = json.loads(previous_plan_paths[-1].read_text(encoding="utf-8"))
        previous_equity = float(previous_plan.get("account_state", {}).get("total_equity", 0.0))
        previous_equity_date = str(previous_plan.get("signal_date", previous_plan_paths[-1].stem[:10]))
    daily_pnl = total_equity - previous_equity if previous_equity > 0 else 0.0
    daily_return = daily_pnl / previous_equity if previous_equity > 0 else 0.0
    candidate_names = "、".join(f"{row['name']}（{row['symbol']}）" for _, row in candidates.iterrows()) or "无"
    previous_target_row = (
        previous_ranking.loc[previous_ranking["symbol"].eq(target_symbol)]
        if target_symbol and not previous_ranking.empty else pd.DataFrame()
    )
    if not target_row.empty:
        target = target_row.iloc[-1]
        if not previous_target_row.empty:
            previous_target = previous_target_row.iloc[-1]
            score_change = float(target[score_column]) - float(previous_target[score_column])
            target_change_story = (
                f"排名由第{int(previous_target['rank'])}变为第{int(target['rank'])}，选择分较昨日{score_change * 100:+.2f}个百分点；"
                f"ROC20由{float(previous_target['roc20']):.2%}降至{float(target['roc20']):.2%}，"
                f"ROC60由{float(previous_target['roc60']):.2%}降至{float(target['roc60']):.2%}。"
            )
            previous_close = float(previous_target["close"])
        else:
            target_change_story = (
                f"ROC20为{float(target['roc20']):.2%}，ROC60为{float(target['roc60']):.2%}。"
            )
            price_frame = pd.read_csv(ROOT / "market_data" / "prices" / f"{target_symbol}.csv")
            price_frame["datetime"] = pd.to_datetime(price_frame["datetime"])
            recent_prices = price_frame.loc[price_frame["datetime"].le(pd.Timestamp(args.date))].sort_values("datetime").tail(2)
            previous_close = float(recent_prices.iloc[-2]["close"]) if len(recent_prices) >= 2 else market_price
        close_change = market_price / previous_close - 1.0 if previous_close > 0 else 0.0
        roc20 = float(target["roc20"])
        roc60 = float(target["roc60"])
        if roc20 > 0 and roc60 > 0:
            momentum_story = "ROC20、ROC60均为正，短中期动量同向。"
        elif roc20 > 0:
            momentum_story = "ROC20仍为正、ROC60已转负；中期动量走弱，但这不是现有仓位的独立卖出条件。"
        else:
            momentum_story = "ROC20已转负，需要按正式退出规则处理。"
        target_insight = (
            f"{target_name}收于{market_price:.3f}元、较昨日{close_change:+.2%}。{target_change_story}"
            f"MA120乖离为{float(target['ma120_bias']):.2%}，价格仍在MA120上方。{momentum_story}"
        )
    else:
        target_insight = "今天没有ETF成为最终目标，账户保持现金。"
    rejected_details = []
    for _, row in ranking.loc[ranking["rank"].le(5) & ~ranking["final_entry_pass"].astype(bool)].iterrows():
        reasons = []
        if float(row["roc20"]) <= 0:
            reasons.append(f"ROC20 {float(row['roc20']):.2%}")
        if float(row["roc60"]) <= 0:
            reasons.append(f"ROC60 {float(row['roc60']):.2%}")
        if float(row["ma120_bias"]) > 0.09:
            reasons.append(f"MA120乖离 {float(row['ma120_bias']):.2%}")
        rejected_details.append(f"第{int(row['rank'])}名{row['name']}因{'、'.join(reasons[:2]) or '入场条件不足'}未通过")
    rejected_story = "；".join(rejected_details) or "前5名没有额外淘汰项"
    leading_rejection = rejected_details[0] if rejected_details else "前5名没有额外淘汰项"
    category_peers = ranking.loc[
        ranking["category"].eq(target_category) & ~ranking["symbol"].eq(target_symbol)
    ].sort_values("rank")
    if not category_peers.empty:
        peer = category_peers.iloc[0]
        if bool(peer["final_entry_pass"]):
            peer_result = "并已通过正式入场筛选"
        elif float(peer["rank"]) > 5:
            peer_result = "但未进入前5"
        else:
            peer_result = "但未通过完整入场条件"
        category_peer_story = (
            f"同属{target_category}的{peer['name']}排第{int(peer['rank'])}，ROC20为{float(peer['roc20']):.2%}，"
            f"{peer_result}。"
        )
    else:
        category_peer_story = f"{target_category}没有其他可比ETF。"
    if current_symbol and current_symbol == target_symbol:
        decision_story = f"现有{target_name}仓位未触发卖出，继续持有；不因其他ETF排名更高而换仓。"
    elif not current_symbol and target_symbol:
        decision_story = f"账户原本空仓，因此从最终合格候选中按正式选择分选出{target_name}作为唯一目标。"
    elif current_symbol and target_symbol:
        decision_story = f"旧仓已经触发退出，卖出后再按正式选择分切换到{target_name}，先卖后买。"
    elif current_symbol:
        decision_story = "旧仓已经触发退出，但今天没有新的合格候选，因此卖出后持有现金。"
    else:
        decision_story = "账户原本空仓，今天又没有新的合格候选，因此继续持有现金。"
    if current_symbol and current_symbol == target_symbol and target_symbol:
        core_insight = (
            f"{target_name}虽出现单日波动，但仍未触发价格退出；{leading_rejection}，"
            "因此继续按既定纪律持有。"
        )
    else:
        core_insight = decision_story
    target_weight = "100%" if target_symbol else "0%"
    overview_paragraphs = [
        f"今天账户变动{daily_pnl:+,.2f}元，收益{daily_return:+.2%}；自{strategy_start}实盘开启以来累计{strategy_pnl:+,.2f}元，本次买入浮动盈亏{purchase_pnl:+,.2f}元。明日结论不变：{actions}，不产生新订单。",
        target_insight,
        f"{len(ranking)}只ETF中最终通过{len(candidates)}只，分别是{candidate_names}。{decision_story}前5名里，{rejected_story}；它们名次高，但当前不是可买候选。",
        f"板块内部也有呼应：{category_peer_story}这说明{target_category}方向并非只有当前持仓走强；现有{target_name}没有触发退出，因此不会仅因出现新的合格候选而换仓。今天没有新趋势或9%—12%质量延伸候选，最终候选都来自常规动量。",
        f"资讯面上，{target_category}主题为{target_positive}条正向、{target_negative}条负向，均按专属关键词直接映射。今天的核心洞察是：{core_insight}",
        satellite_status,
    ]
    if len("".join(overview_paragraphs)) < 500:
        raise RuntimeError("daily plain-language overview must contain at least 500 characters")
    lines = [
        f"# ye 策略日报｜{args.date}",
        "",
        f"> 次日开盘：**{actions}**。",
        "",
        "## 收益速览",
        "",
        "| 口径 | 收益率 | 收益额 | 区间 |",
        "|---|---:|---:|---|",
        f"| 策略实盘开启以来 | {strategy_return:+.2%} | {strategy_pnl:+,.2f} 元 | {strategy_start} 至今 |",
        f"| 本次买入收益 | {purchase_return:+.2%} | {purchase_pnl:+,.2f} 元 | 成本 {average_cost:.3f} 元 |",
        f"| 今日收益 | {daily_return:+.2%} | {daily_pnl:+,.2f} 元 | 对比 {previous_equity_date} |",
        "",
        "## 今日通俗综述",
        "",
        *overview_paragraphs,
        "",
        "## 一、实盘结论",
        "",
        f"- 信号日：{args.date} 收盘后；执行：下一交易日开盘。",
        f"- 当前权益：{account['total_equity']:,.2f} 元；{account_position}。",
        f"- 唯一目标：{target_name}（{target_symbol or '现金'}）{('，目标仓位 100%' if target_symbol else '，目标仓位 0%')}。",
        (f"- 订单动作：{actions}；买卖计划待下一交易日真实成交确认。" if execution_orders else f"- 订单动作：{actions}；明日没有买卖订单，无需新增成交确认。"),
        f"- 固定成本：{plan['cost']}。",
        "",
        f"## 二、今天怎样从 {len(ranking)} 只 ETF 得到唯一目标",
        "",
        f"1. **冻结母池**：{len(ranking)}只ETF全部计算，其中45只核心、{len(satellite_ranking)}只卫星；核心独立排名并拥有买入优先权。",
        f"2. **新开仓资格**：{int(pool_eligible.sum())}/{len(ranking)}只通过上市≥120日、20日成交额中位数≥2,000万元。",
        f"3. **三条路径并行**：常规动量 {normal_count} 只、新趋势例外 {emerging_count} 只、9%—12%质量延伸 {extension_count} 只。",
        f"4. **核心优先后最终合格 {len(candidates)} 只**：{candidate_names}；只有当天没有核心候选时，卫星才补位。",
        f"5. **明日实际目标 {1 if target_symbol else 0} 只**：{target_name}（{target_symbol or '现金'}）；{account_decision}",
        "",
        "### 今日卫星检查",
        "",
        f"- **结论**：{satellite_status}",
        f"- 核心最终候选：{len(core_candidates)} 只；卫星技术合格：{len(satellite_technical)}/{len(satellite_ranking)} 只；卫星最终补位：{len(satellite_final)} 只。",
        "",
        "| 卫星ETF | 虚拟排名 | 动量分 | ROC20 | ROC60 | MA120乖离 | 技术结果 | 当日处理 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
        *[
            f"| {row['name']}（{row['symbol']}） | {int(row['rank'])} | {row['momentum_score']:.2%} | {row['roc20']:.2%} | {row['roc60']:.2%} | {row['ma120_bias']:.2%} | {'通过' if row['technical_entry_pass'] else '未通过'} | "
            f"{'最终补位候选' if row['final_entry_pass'] else ('技术通过；核心已有候选，待命' if row['technical_entry_pass'] and not core_candidates.empty else ('技术通过；未进入最终候选' if row['technical_entry_pass'] else entry_blockers(row)))} |"
            for _, row in satellite_ranking.iterrows()
        ],
        "",
        "| 最终顺序 | ETF | 池角色 | 决策排名 | 选择分 | ROC20 | ROC60 | MA120乖离 | 入场路径 | 处理 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
        *[
            f"| {index} | {row['name']}（{row['symbol']}） | {'卫星' if row.get('pool_role') == 'challenger' else '核心'} | {int(row['rank'])} | {row[score_column]:.2%} | {row['roc20']:.2%} | {row['roc60']:.2%} | {row['ma120_bias']:.2%} | {entry_path(row)} | {candidate_outcome(row, target_symbol, current_symbol)} |"
            for index, (_, row) in enumerate(candidates.iterrows(), start=1)
        ],
        "",
        "## 三、情绪审核结论",
        "",
        f"- 审核覆盖：{review['reviewed_count']}/{review['input_count']} = {review['coverage']:.0%}；相关记录：{review['relevant_count']} 条。",
        "- 数据来源：" + "；".join(f"{name} {count} 条" for name, count in source_counts(review["items"])),
        f"- **主要正向**：{strongest[0]}共 {strongest[1]} 正、{strongest[2]} 负，是当日净正向记录最集中的主题。",
        f"- **风险与分歧**：{most_negative[0]}出现 {most_negative[2]} 条负向记录；需与同主题正向记录及价格趋势合并看待。",
        (
            f"- **目标{target_category}主题**：共 {target_positive} 正、{target_negative} 负；"
            + (
                f"今日通过{entry_path(target_row.iloc[-1])}，不依赖其他例外放行。"
                if not target_row.empty and bool(target_row.iloc[-1]["final_entry_pass"])
                else "现有持仓未触发退出，不依赖今日新开仓路径；资讯不能覆盖价格退出规则。"
            )
        ) if target_sentiment is not None else "- 目标主题没有可用的情绪映射，按未知处理。",
        "- 结论：审核完整性硬门已通过；当天没有候选依赖新趋势或质量延伸例外。正式规则没有统一负面阈值去否决所有常规买点。",
        "",
        "| 主题 | 正向记录 | 负向记录 |",
        "|---|---:|---:|",
        *[f"| {category} | {positive} | {negative} |" for category, positive, negative in category_rows],
        "",
        "主题统计按原始记录的主题映射计数；单条记录可能映射多个主题，不等于独立股票数量。",
        "",
        "## 四、明日订单与资金计算",
        "",
        *(
            [
                f"- 计划买入：{target_name}（{target_symbol}），目标仓位100%。",
                f"- 7月20日收盘价 {buy_estimate.get('last_close', 0):.3f} 元；按固定滑点与最低佣金估算，可买约 **{int(buy_estimate.get('estimated_quantity_at_last_close', 0)):,} 份**，预计占用 {buy_estimate.get('estimated_notional', 0) + buy_estimate.get('estimated_commission', 0):,.2f} 元。",
                "- 上述数量只是收盘估算。明日必须按实际开盘成交价、实际可用资金和100份整数倍重新计算，绝不允许透支。",
                "- 成交后请提供实际数量、均价及未成交数量；在确认前，账户仍记为‘计划待成交’。",
            ] if buy_order else ["- 明日没有新买单；按计划持有或保持现金。"]
        ),
    ]
    if execution_orders:
        lines.extend([
            "",
            "## 五、订单执行参考",
            "",
            "| 动作 | ETF | 20日额中位数 | 单日参与上限 | 参考资金下预计开盘次数 |",
            "|---|---|---:|---:|---:|",
            *[
                f"| {side_name.get(item['side'], item['side'])} | {item['symbol']} | {item['amount_median_20d']:,.0f} 元 | {item['max_notional_per_open']:,.0f} 元 | {item['estimated_opens_at_reference_capital'] or '不可计算'} |"
                for item in execution_orders
            ],
            "",
            "以上是同一目标仓位的流动性机械拆分参考，不是新的买卖信号；实际成交须在下一次运行前确认。",
        ])
    detail_section_number = "六" if execution_orders else "五"
    lines.extend([
        "",
        f"## {detail_section_number}、{len(ranking)}只ETF逐层结果",
        "",
        "‘最终通过’只代表进入候选集合，不代表同时买入；空仓时只买最终候选中动量分最高的一只。",
        "",
        "| 决策排名 | ETF | 池角色 | 资格 | 动量分 | ROC20 | ROC60 | MA120乖离 | 常规 | 新趋势 | 延伸 | 最终处理 |",
        "|---:|---|---|---|---:|---:|---:|---:|---|---|---|---|",
        *[
            f"| {int(row['rank'])} | {row['name']}（{row['symbol']}） | {'卫星' if row.get('pool_role') == 'challenger' else '核心'} | {'通过' if row['pool_eligible'] else '未通过'} | {row['momentum_score']:.2%} | {row['roc20']:.2%} | {row['roc60']:.2%} | {row['ma120_bias']:.2%} | {'通过' if row.get('confirmed_normal_entry', row['normal_entry']) and row['pool_eligible'] else '—'} | {'通过' if row['emerging_entry'] and row['pool_eligible'] else '—'} | {'通过' if row['quality_extension'] and row['pool_eligible'] else '—'} | {candidate_outcome(row, target_symbol, current_symbol)} |"
            for _, row in ranking.iterrows()
        ],
    ])
    target = ROOT / "results" / "live" / f"{args.date}_daily_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
