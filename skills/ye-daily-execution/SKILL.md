---
name: ye-daily-execution
description: 在中国市场收盘后执行唯一正式的 ye ETF 轮动策略。用户提出“运行 ye”“复核今天盘面与新闻”“生成明日 ETF 计划”“生成今日日报”“查看实盘订单、运行卡、成交对账或审核状态”时使用：核对实际成交，冻结日线与资讯，在当前对话逐条审核全部资讯，覆盖率不足 100% 或前一订单未对账时禁止新增仓位，再生成、校验并报告下一交易日开盘计划、日报、HTML、运行审计与实盘偏差状态；不需要外部 API 密钥。
---

# ye 每日执行

## 目标与边界

在包含 `config/ye_strategy.yaml` 的 yeetf 仓库根目录运行。项目可以位于任意本地路径；找不到时先定位该配置文件，仍找不到才询问用户。

在本 macOS 工作区使用 `PYTHONPATH=src python3` 执行 Python 命令。若项目目录中另有已验证的 `.venv-review`，可改用其解释器，但不得安装、升级或替换依赖来改变正式运行环境。

每次运行前必须完整读取 `RESEARCH_MEMORY.md` 和 `RESEARCH_STATUS.md`。前者保存长期结论与变更记录，后者只保存当前冻结状态和账户事实；两者都不得覆盖配置和账户真源。

- `config/ye_strategy.yaml` 是唯一正式策略规则；日常执行不得修改规则、ETF 池、成本、研究参数或网页口径。
- 正式母池为45只核心冠军＋6只挑战者空档补位。核心池独立形成排名、类别宽度和核心持仓的排名退出，并拥有第一买入权；只有当天无合格核心候选时，挑战者才可参与；挑战者持仓遇合格核心候选时下一开盘让位。核心旧仓还须检查正式机会换仓：掉出前5、完整合格核心候选动量分领先至少5个百分点、同一候选连续2日、旧仓持有满5日时，下一开盘全仓换仓。不得在日报或人工判断中改回51只全局混排或核心/挑战者平权竞争。
- `config/strategy_governance.yaml` 定义正式冻结与放行边界；`config/research_hypotheses.yaml` 只登记独立研究，不产生任何日常信号。
- 只使用信号日收盘数据，于下一交易日开盘执行。目标只能是现金 0% 或单一 ETF 100%。不加仓、减仓、网格、主观覆盖或分批止盈。
- 所有资讯审核均在当前 Codex 对话中完成；不要求也不使用 `OPENAI_API_KEY`。用户可以直接要求逐条解释。
- 用户于2026-08-06授予常设执行授权：未另行报告时，视为完整执行上一交易日的开盘计划。日线更新后先运行 `PYTHONPATH=src python3 scripts/advance_authorized_live_account.py --date YYYY-MM-DD`，以当日实际开盘价和固定成本记账；券商回单、手工成交、出入金或未完成订单一旦报告，立即覆盖该假定记录。回测影子持仓始终不得代替真实账户。
- 严格区分正式策略与任何研究项目：正式流程不得运行 `research_*`、`summarize_*`、候选筛选或其他策略脚本。

## 收盘后固定流程

将 `YYYY-MM-DD` 替换为已收盘的交易日。先读取上一交易日的 `results/live/*_order_plan.json`，记录用户确认的实际成交；没有确认时继续数据与风控复核，但不虚构实际仓位。

0. 若上一份计划含买入或卖出，先读取 [成交确认格式](references/execution-confirmation-schema.md)。在用户提供券商成交数量、价格和未完成数量后，用 `apply_patch` 写入 `results/live/上一信号日_actual_fills.json`，再运行：

   ```sh
   PYTHONPATH=src python3 scripts/reconcile_actual_fills.py --date 上一信号日
   ```

   未提供成交确认时，明确标记“待确认”，不得把计划当作持仓；后续 `READY` 会禁止新增仓位。

1. 确认该日存在于基准交易日历，刷新日线并冻结资讯：

   ```sh
   PYTHONPATH=src python3 scripts/fetch_prices.py --force
   PYTHONPATH=src python3 scripts/collect_daily_sentiment.py --date YYYY-MM-DD
   PYTHONPATH=src python3 scripts/export_sentiment_review_queue.py --date YYYY-MM-DD
   ```

   随后在常设执行授权下运行：

   ```sh
   PYTHONPATH=src python3 scripts/advance_authorized_live_account.py --date YYYY-MM-DD
   ```

2. 读取 `market_data/sentiment/review_queue/YYYY-MM-DD.json`。逐行审核，包含无关资讯；保留原始 `source_hash`，不得合并、漏审、重复或补造。按 [审核草稿格式](references/review-schema.md) 用 `apply_patch` 写入 `market_data/sentiment/manual_drafts/YYYY-MM-DD.json`。

   审核时禁止把类别标签自动扩散到同类全部 ETF；`matched_symbols` 必须获得冻结行中专属关键词的直接支持。同一公司跨来源重复时仍逐行审核，系统在下游特征层自动只计一次。草稿必须记录当前 Codex 的模型家族、可见快照信息与使用界面；精确快照未暴露时如实记录 `not_exposed_by_codex`。

3. 提交并验证审核记录；失败不得绕过：

   ```sh
   PYTHONPATH=src python3 scripts/commit_manual_sentiment_review.py --date YYYY-MM-DD --reviews market_data/sentiment/manual_drafts/YYYY-MM-DD.json
   ```

4. 运行唯一正式入口。它会更新当天数据截止日、重建正式回测/信号、交易审计、订单计划、日报、三个 HTML、上线检查、SHA-256 清单与每日运行卡：

   ```sh
   PYTHONPATH=src python3 scripts/run_after_close.py --date YYYY-MM-DD --skip-collect
   ```

   ETF计划生成后，于15:00—15:30核对收盘宝自动申报状态。它只管理实际闲置现金：大于1,000元时按1,000元整数倍参与1天期通用回购，手续费十万分之一；记录下单时实时年化。下一交易日开盘前本息应可用。任何产品异常均保留现金，不得改动ETF计划。

5. 阅读并交叉核对以下文件，再回答用户：

   - `results/live/YYYY-MM-DD_order_plan.json`
   - `results/live/YYYY-MM-DD_daily_report.md`
   - `results/live/readiness_report.json`
   - `results/audit/YYYY-MM-DD_run_manifest.json`
   - `results/audit/YYYY-MM-DD_live_run_card.json`
   - `results/comparison/latest_signals.json`

   同时确认 `latest_ranking.csv` 恰有51只、`pool_role` 为45个 `core` 与6个 `challenger`；任一不符视为池架构校验失败，不能放行新仓。

6. 用 `apply_patch` 维护 `RESEARCH_STATUS.md` 的当前快照，并在 `RESEARCH_MEMORY.md` 的变更记录中增加一条简洁的当日运行或实质改动。没有写入记忆的项目改动不算完成；不得把详细日报复制进记忆。

7. 正式校验通过并完成上述记忆维护后，检查 Git 差异，确认不含 token、密码、`.env`、缓存、临时文件或 ZIP。由Agent按改动重要性自主判断是否提交并推送：正式日运行、规则/代码或账户真源等关键节点通常同步；过程性研究不要求单独推送。需要同步时使用：

   ```sh
   git status --short --branch
   git add -A
   git commit -m "daily: run YYYY-MM-DD close"
   git push origin HEAD
   ```

   没有新差异时不制造空提交。无论是否选择推送，都要如实说明同步状态；推送失败不改变已经生成的交易信号或 `READY/BLOCKED`，但必须向用户明确报告“日报已完成、GitHub 未同步”及失败原因。

## 强制放行规则

只有下列条件同时满足，才可称“次日计划可执行”：

- `readiness_report.json` 的状态为 `READY`；
- `ai_review_complete=true`，且审核状态为 `complete`、`coverage=1.0`、输入条数等于审核条数；
- 运行清单存在，且其中信号日期、价格快照、审核日期相互一致；
- 每日运行卡存在，且策略标识、计划、审核覆盖率、运行清单哈希和放行状态相互一致；
- 若上一份计划需要成交确认，对账结果必须为 `confirmed` 或用户常设授权下的 `assumed_authorized`；后者必须在账户真源和日报中明确披露，券商回单优先覆盖。

审核覆盖率不是“对相关资讯 100%”，而是对队列**全部行** 100%。任一资讯源失败、队列缺失、哈希不一致、草稿格式不合法或覆盖不足时，禁止新增仓位；已有仓位的价格卖出规则继续计算并如实报告。

## 审核判断原则

- 与固定 ETF 池无实质关系：`relevant=false`，但仍填写全部字段。
- 相关资讯仅根据冻结文本识别主题、ETF、方向、期限、证据和风险；不确定时降低置信度并写入风险。
- 情绪只能确认或否决边缘买点、识别新趋势、允许高质量延伸，或短暂保护性退出；不能推翻 MA120 硬退出，也不能凭新闻制造价格趋势。
- 不把叙事、传闻或盘中波动写成事实；证据字段必须可追溯至原队列文本。

## 对用户的简洁交付格式

第一句直接写：`次日开盘：买入 / 卖出 / 换仓 / 持有 / 空仓`。

随后仅列出目标 ETF、目标仓位（0%/100%）、实际成交确认/对账状态、AI 审核覆盖率、`READY/BLOCKED`、GitHub 同步结果、固定成本和任何阻塞项。计划待成交确认时必须写“计划待成交确认”。最后给出三个正式 HTML 和当日运行卡的本地链接；审核细节保留在日报和审计文件中。

## 月度复盘（不属于每日流程）

仅在月末收盘后或用户明确要求时运行：

```sh
PYTHONPATH=src python3 experiments/strategy_ablation.py
PYTHONPATH=src python3 experiments/monthly_live_review.py --month YYYY-MM
```

月度复盘比较正式策略、纯价格核心和实盘账户路径，只输出研究报告，不修改正式信号、订单、日报或任何参数。不得把月度研究脚本接入 `scripts/run_after_close.py`。
