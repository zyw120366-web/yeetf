# ye ETF 轮动策略

本项目只维护一个正式策略：**ye 策略**。`config/ye_strategy.yaml` 是唯一规则真源；`config/etfwin_official.yaml` 只定义同口径对照策略，不参与 ye 选股或实盘订单。

## 项目入口

- [策略规范](docs/策略规范.md)：当前完整买入、持有、卖出、仓位、情绪与成本规则。
- [每日执行](docs/每日执行.md)：收盘后生成下一交易日开盘计划的固定流程。
- [工程审计](docs/工程审计.md)：模块边界、输出真源、哈希清单和验证要求。
- [策略治理与实盘审计](docs/策略治理与实盘审计.md)：规则冻结、研究假设登记、每日运行卡与真实成交对账。
- `run_strategies.py`：重建 ye 与 etfwin 的同快照回测和当日信号。
- `scripts/run_after_close.py`：AI 审核提交后的正式每日入口。
- `dashboard/scripts/build_ye_strategy_html.py`：生成策略、日报和回测三个静态 HTML。
- `outputs/`：三份可直接打开的正式 HTML；生成路径始终位于仓库内部，不依赖电脑用户名或克隆位置。
- `skills/ye-daily-execution/`：与项目一起版本管理的每日执行 Skill 真源。

## 固定验证

```sh
PYTHONPATH=.:src python3 -m pytest -q
PYTHONPATH=src python3 run_strategies.py
PYTHONPATH=src python3 scripts/build_trade_audit.py
PYTHONPATH=src python3 dashboard/scripts/build_ye_strategy_html.py
PYTHONPATH=src python3 scripts/validate_live_readiness.py --date YYYY-MM-DD
PYTHONPATH=src python3 scripts/build_live_run_card.py --date YYYY-MM-DD
```

每日使用时直接在 Codex 对话中调用 `$ye-daily-execution`。不需要 `OPENAI_API_KEY`；当日所有盘面与资讯由当前 Codex 对话逐条审核。

## 两台电脑同步

仓库地址：`https://github.com/zyw120366-web/yeetf.git`。仓库必须保持为 Private，因为其中包含实盘账户状态、成交对账和完整研究记录。

首次在新电脑安装：

```sh
git clone https://github.com/zyw120366-web/yeetf.git
cd yeetf
python3 scripts/install_codex_skill.py
PYTHONPATH=.:src python3 -m pytest -q
```

每天开始修改前先在 GitHub Desktop 点击 `Pull origin`；完成运行或修改后填写提交说明，依次点击 `Commit to main` 和 `Push origin`。不要在两台电脑都存在未推送修改时同时编辑同一文件。

Skill 由仓库中的 `skills/ye-daily-execution` 统一维护。在 macOS/Linux 上安装器会建立链接，仓库更新后 Skill 自动同步；若系统不允许链接则回退为复制，更新仓库后需要重新运行安装器。

任何 GitHub token、券商密码、API key 或 `.env` 文件都不得提交。缓存、虚拟环境、临时文件和打包 ZIP 已由 `.gitignore` 排除。
