# ye ETF轮动策略

项目只维护一个正式策略：**ye策略**。`config/ye_strategy.yaml` 是唯一规则真源；etfwin仅作同口径参考，不参与实盘订单。

当前ETF池采用“45只核心冠军＋6只挑战者空档补位”：核心候选拥有第一买入权；挑战者只在核心无候选时参与，并在核心候选恢复时让位。

## 从这里开始

- [当前状态](RESEARCH_STATUS.md)：今天的账户、持仓、计划与放行结果。
- [项目记忆](RESEARCH_MEMORY.md)：长期研究结论、否定项和每次实质改动。
- [策略规范](docs/策略规范.md)：买入、卖出、仓位、情绪和成本规则。
- [每日执行](docs/每日执行.md)：收盘后固定SOP。
- [工程与治理](docs/工程与治理.md)：真源、模块边界、变更纪律与验收。

## 常用入口

```sh
# 每日正式运行（先按Skill完成采集与逐条审核）
PYTHONPATH=src python3 scripts/run_after_close.py --date YYYY-MM-DD --skip-collect

# 月度研究复盘；不参与每日信号
PYTHONPATH=src python3 experiments/strategy_ablation.py
PYTHONPATH=src python3 experiments/monthly_live_review.py --month YYYY-MM

# 项目验收
PYTHONPATH=.:src python3 scripts/validate_project_hygiene.py
PYTHONPATH=.:src python3 -m pytest -q
```

每日在Codex中调用 `$ye-daily-execution`。不需要外部AI密钥；当日资讯由当前对话逐条审核。

## 目录

- `config/`：规则、市场、审核和治理真源
- `src/`：策略与回测核心
- `scripts/`：正式运行和审计工具
- `experiments/`：与实盘隔离的研究
- `market_data/`：价格与冻结资讯
- `results/`：正式、实盘、审计与研究结果
- `dashboard/`：页面生成器和内部构建产物
- `outputs/`：可直接打开的三份中文HTML
- `skills/ye-daily-execution/`：每日执行Skill

## 两台电脑同步

远端仓库为 [zyw120366-web/yeetf](https://github.com/zyw120366-web/yeetf)，必须保持Private。每台电脑开始工作前先拉取，结束后提交并推送；不要让两台电脑同时保留未推送的同文件修改。

首次安装：

```sh
git clone https://github.com/zyw120366-web/yeetf.git
cd yeetf
python3 scripts/install_codex_skill.py
PYTHONPATH=.:src python3 -m pytest -q
```

不得提交GitHub token、券商密码、API key、`.env`、缓存、临时文件或ZIP。
