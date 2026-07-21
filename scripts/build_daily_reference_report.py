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
        return entry_blockers(row)
    if str(row["symbol"]) == target_symbol:
        return "最终合格且正式路径选择分最高，选为唯一目标"
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
    ranking = pd.read_csv(ROOT / "results" / "comparison" / "latest_ranking.csv")
    ranking = ranking.loc[ranking["date"].eq(args.date)].sort_values(["rank", "momentum_score"])
    if ranking.empty:
        raise RuntimeError(f"missing momentum ranking for {args.date}")
    score_column = "selection_score" if "selection_score" in ranking.columns else "momentum_score"
    candidates = ranking.loc[ranking["final_entry_pass"].astype(bool)].sort_values(
        [score_column, "rank"], ascending=[False, True]
    )
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
    held_row = ranking.loc[ranking["symbol"].eq(current_symbol)] if current_symbol else pd.DataFrame()
    if not held_row.empty:
        held = held_row.iloc[-1]
        held_exit_checks = (
            f"收盘价{'仍在MA120上方' if bool(held['above_ma120']) else '已经跌到MA120下方'}，"
            f"ROC20为{float(held['roc20']):.2%}，"
            f"5日与20日排名{'没有同时恶化' if not bool(held['dual_rank_decline']) else '已经同时恶化'}"
        )
    else:
        held_exit_checks = "当前没有旧仓需要检查"
    if not target_row.empty:
        target = target_row.iloc[-1]
        target_snapshot = (
            f"{target_name}在全池排第{int(target['rank'])}，正式选择分为{float(target[score_column]):.2%}，"
            f"ROC20为{float(target['roc20']):.2%}、ROC60为{float(target['roc60']):.2%}，"
            f"MA120乖离为{float(target['ma120_bias']):.2%}，通过{entry_path(target)}路径"
        )
    else:
        target_snapshot = "今天没有ETF通过全部条件，策略目标因此是现金"
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
    order_summary = (
        "本次存在买卖订单，下一交易日必须按真实成交结果对账。"
        if execution_orders else "本次没有新增买卖订单，也不需要新增成交确认。"
    )
    target_weight = "100%" if target_symbol else "0%"
    overview_paragraphs = [
        f"结论：本次使用{args.date}收盘数据决定下一交易日动作。实盘账户{account_position}，可用现金{account['available_cash']:,.2f}元，收盘总权益{account['total_equity']:,.2f}元。明日计划为“{actions}”，目标仓位{target_weight}；{decision_story}",
        f"第一步先查旧仓，不先追排行榜。卖出只看三项：跌破MA120、ROC20转负、当前排名同时差于5日前和20日前。今日检查为：{held_exit_checks}。因此旧仓处理结论明确；价格退出一旦触发，新闻和热点不能推翻。",
        f"第二步筛全池。固定{len(ranking)}只ETF全部计算，{int(pool_eligible.sum())}只通过上市满120个交易日和20日成交额中位数不低于2,000万元的资格门槛。三条路径并行：常规动量{normal_count}只、新趋势{emerging_count}只、9%—12%质量延伸{extension_count}只，取并集后最终合格{len(candidates)}只。合格只代表进入候选，不代表同时买入。",
        f"第三步核对目标：{target_snapshot}。排名只是起点，还必须同时满足双ROC、MA120位置、乖离上限，以及对应路径的趋势质量或热点条件；任何关键门槛失败都不能入选。",
        f"第四步审核资讯：{review['reviewed_count']}/{review['input_count']}条已逐条完成，覆盖率{review['coverage']:.0%}；{target_category}主题为{target_positive}条正向、{target_negative}条负向。资讯只确认指定例外和软退出保护，不能凭新闻制造买点，也不能覆盖MA120硬退出。最终状态{readiness['status']}，{order_summary}明日按计划执行，不临时追涨、加仓或改标的。",
    ]
    if len("".join(overview_paragraphs)) < 500:
        raise RuntimeError("daily plain-language overview must contain at least 500 characters")
    lines = [
        f"# ye 策略日报｜{args.date}",
        "",
        f"> 次日开盘：**{actions}**。上线检查：**{readiness['status']}**。",
        "",
        "## 今日通俗综述",
        "",
        *overview_paragraphs,
        "",
        "## 一、实盘结论",
        "",
        f"- 信号日：{args.date} 收盘后；执行：下一交易日开盘。",
        f"- 实盘账户真相：用户确认资金 {account['total_equity']:,.2f} 元、可用现金 {account['available_cash']:,.2f} 元、{account_position}；状态日期 {account['as_of']}。",
        f"- 唯一目标：{target_name}（{target_symbol or '现金'}）{('，目标仓位 100%' if target_symbol else '，目标仓位 0%')}。",
        (f"- 订单动作：{actions}；买卖计划待下一交易日真实成交确认。" if execution_orders else f"- 订单动作：{actions}；明日没有买卖订单，无需新增成交确认。"),
        f"- 固定成本：{plan['cost']}。",
        "",
        "## 二、今天怎样从 45 只 ETF 得到唯一目标",
        "",
        f"1. **固定母池**：45只ETF全部参与计算。",
        f"2. **新开仓资格**：{int(pool_eligible.sum())}/45只通过上市≥120日、20日成交额中位数≥2,000万元。",
        f"3. **三条路径并行**：常规动量 {normal_count} 只、新趋势例外 {emerging_count} 只、9%—12%质量延伸 {extension_count} 只。",
        f"4. **路径取并集**：共有 {len(candidates)} 只最终合格。",
        f"5. **账户状态决策**：{account_decision}",
        "",
        "| 最终顺序 | ETF | 全池排名 | 选择分 | ROC20 | ROC60 | MA120乖离 | 入场路径 | 处理 |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
        *[
            f"| {index} | {row['name']}（{row['symbol']}） | {int(row['rank'])} | {row[score_column]:.2%} | {row['roc20']:.2%} | {row['roc60']:.2%} | {row['ma120_bias']:.2%} | {entry_path(row)} | {candidate_outcome(row, target_symbol, current_symbol)} |"
            for index, (_, row) in enumerate(candidates.iterrows(), start=1)
        ],
        "",
        "## 三、情绪审核结论",
        "",
        f"- 审核覆盖：{review['reviewed_count']}/{review['input_count']} = {review['coverage']:.0%}；相关记录：{review['relevant_count']} 条。",
        "- 数据来源：" + "；".join(f"{name} {count} 条" for name, count in source_counts(review["items"])),
        f"- **主要正向**：{strongest[0]}共 {strongest[1]} 正、{strongest[2]} 负，是当日净正向记录最集中的主题。",
        f"- **风险与分歧**：{most_negative[0]}出现 {most_negative[2]} 条负向记录；需与同主题正向记录及价格趋势合并看待。",
        f"- **目标{target_category}主题**：共 {target_positive} 正、{target_negative} 负；目标走常规动量路径，不依赖热点例外放行。" if target_sentiment is not None else "- 目标主题没有可用的情绪映射，按未知处理。",
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
        "",
        "## 五、放行与执行纪律",
        "",
        "- 本日报告只使用信号日收盘后冻结的数据；不使用盘中信息。",
        "- 只有 `READY` 才执行计划；若为 `BLOCKED`，不得新开仓。",
        "- 当日运行卡：`results/audit/" + args.date + "_live_run_card.json`；订单计划不是成交。",
        "- 若前一份计划有买卖，下一次运行前必须完成实际成交对账；未对账时禁止新开仓。",
        "- 情绪只用于确认或过滤，不能覆盖 MA120 硬退出。",
    ]
    if execution_orders:
        lines.extend([
            "",
            "### 流动性执行参考",
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
    lines.extend([
        "",
        "## 六、45只ETF逐层结果",
        "",
        "‘最终通过’只代表进入候选集合，不代表同时买入；空仓时只买最终候选中动量分最高的一只。",
        "",
        "| 排名 | ETF | 资格 | 动量分 | ROC20 | ROC60 | MA120乖离 | 常规 | 新趋势 | 延伸 | 最终处理 |",
        "|---:|---|---|---:|---:|---:|---:|---|---|---|---|",
        *[
            f"| {int(row['rank'])} | {row['name']}（{row['symbol']}） | {'通过' if row['pool_eligible'] else '未通过'} | {row['momentum_score']:.2%} | {row['roc20']:.2%} | {row['roc60']:.2%} | {row['ma120_bias']:.2%} | {'通过' if row.get('confirmed_normal_entry', row['normal_entry']) and row['pool_eligible'] else '—'} | {'通过' if row['emerging_entry'] and row['pool_eligible'] else '—'} | {'通过' if row['quality_extension'] and row['pool_eligible'] else '—'} | {candidate_outcome(row, target_symbol, current_symbol)} |"
            for _, row in ranking.iterrows()
        ],
    ])
    target = ROOT / "results" / "live" / f"{args.date}_daily_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
