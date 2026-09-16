---
name: ye-daily-execution
description: 运行ye收盘流程、生成日报与次日ETF计划、核对实盘成交或查看运行结果时使用。只查看已有报告时不重跑；策略研究和规则修改不属于每日执行。
---

# ye 每日执行

在含 `config/ye_strategy.yaml` 的项目根目录工作。全局Skill是项目目录的链接，项目版本是唯一维护源。使用已验证环境；本Mac为 `PYTHONPATH=src python3`，不在日运行中安装或升级依赖。

## 先分清用户要什么

- **查看日报/链接/状态**：读取指定日现有日报与运行卡，直接交付；过期或缺失须说明，不把旧页当今天。
- **运行/补跑**：读取精简的 `RESEARCH_MEMORY.md`、`RESEARCH_STATUS.md`、账户真源与上一计划，执行下方流程。历史档案仅在追溯或研究时按需读。
- **报告真实成交或未执行**：用户事实优先；按 [成交确认](references/execution-confirmation-schema.md) 处理后再运行。不要重复追问已确认买卖，也不要猜未知资金。

## 不可省略的边界

- 策略、ETF池、仓位、成本只读配置；不在日报中主观改买卖、不运行研究脚本、不用回测持仓代替实盘。
- 账户来自 `results/live/account_state.json` 和真实成交。常设授权只由唯一入口核验上一日已放行且绑定哈希的计划后记账为 `assumed_authorized`；不要另跑账户推进脚本。
- 用户/券商的成交、取消、入金或异常优先于假定记录；BLOCKED、pending/exception不可假定买卖。SELL_ONLY仅卖已核验持仓并转现金。
- 未知现金/本金保持null；已确认股数可计算持仓观察，但不能伪造账户总收益或可执行仓位。分析完成与订单放行是两件事。
- 信号只用该日及以前数据，下一交易日开盘执行；实际估值与假定成交使用不复权报价。已审计历史不得用事后行情或新闻补写；已交付的完整观察日报同样冻结历史，不以订单READY为前提。

## 唯一日运行流程

1. 确认信号日已收盘且为中国市场交易日；休市只记录“休市，未生成信号”。基准日线缺失可能是数据失败，不可直接当休市。补跑按交易日顺序处理，禁止用今天的账户倒推过去。
2. 刷新行情、冻结资讯、导出审核队列：

   ```sh
   PYTHONPATH=src python3 scripts/fetch_prices.py --force
   PYTHONPATH=src python3 scripts/collect_daily_sentiment.py --date YYYY-MM-DD
   PYTHONPATH=src python3 scripts/export_sentiment_review_queue.py --date YYYY-MM-DD
   ```

3. 读取 `market_data/sentiment/review_queue/YYYY-MM-DD.json` 全部行。在当前对话逐条审核（含无关和跨源重复行），按 [审核格式](references/review-schema.md) 用 `apply_patch` 写 `market_data/sentiment/manual_drafts/YYYY-MM-DD.json`。机械整理可脚本化，但不能用关键词批处理代替语义审核。专属原文证据才可映射ETF，不扩散类别；保留哈希，模型信息按当前可见信息填写（示例名称不是固定值，不可见填unknown），不需要外部API密钥。

   ```sh
   PYTHONPATH=src python3 scripts/commit_manual_sentiment_review.py --date YYYY-MM-DD --reviews market_data/sentiment/manual_drafts/YYYY-MM-DD.json
   PYTHONPATH=src python3 scripts/run_after_close.py --date YYYY-MM-DD --skip-collect
   ```

4. 入口处理账户、正式信号、订单与报告。审核失败/覆盖不足禁止新增；原有独立风险卖出核验不变。READY、SELL_ONLY、BLOCKED共用日报生成器；数据充足时即使资金待核也输出完整观察，证据不足的部分明确标未知。不得为补日报手写第二套筛选、订单或HTML，不得把退出码2当作无需交付。
5. 读取当日 `*_daily_report.json` 与 `*_order_plan.json`，核对用户事实、关键退出/换仓理由。入口所有状态均内置机器交付检查；生成最终回复所用的已校验摘要及链接：

   ```sh
   PYTHONPATH=src python3 scripts/validate_daily_delivery.py --date YYYY-MM-DD --format markdown
   ```

   同一命令检查日期、51池角色、账户/目标/订单一致性、清单哈希、HTML和链接；不改账户、不重跑策略。PASS只表示交付一致，不等于交易READY。校验失败不得宣称已完成。
6. 更新当前状态和一条简短记忆；检查差异与敏感文件，只提交本任务改动。正式日运行及实质修改通常推送当前分支，过程研究无需单独推送。无差异不造空提交；同步失败不改变交易结论。

## 给用户的结果

先写次日动作（受阻时写“持仓观察，订单未放行”），再给真实股数、可确认的收益、关键理由、审核覆盖及资金/放行状态、GitHub同步。说今日洞察，不复述整套规则，无字数下限。

运行后采用上述命令的摘要与链接，补充实际核验的Git同步结果；不手写路径、不复制旧日数值。只查看已有报告时无需重跑或重新冻结，说明其实际日期，链接仅指向存在的文件。普通回复将Markdown链接放末尾；heartbeat将链接单独放XML之外，不能塞进message或代码块。稳定日报为 `outputs/ETF轮动策略_今日日报.html`；不保证它代表所查询的历史日。不要新建会话、不要用HTML源码面板充当预览，不声称未经验证的页面已渲染。

## 条件性事项

- 收盘宝仅按券商实际回单核验；无访问条件或超出申报时段记录未核，不阻断ETF日报，不推算真实到账利息。历史回测现金规则不变。
- 月末或用户要求复盘时才运行 `experiments/strategy_ablation.py` 与 `experiments/monthly_live_review.py --month YYYY-MM`；只输出研究，不改正式订单。不要把研究放进每日入口。
