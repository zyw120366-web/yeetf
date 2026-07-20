# 静态策略页面

运行以下命令生成策略介绍、今日日报和回测页面：

```powershell
.\.venv-review\Scripts\python.exe dashboard\scripts\build_ye_strategy_html.py
```

生成文件位于 `dashboard/public/`，并同步复制到工作区 `outputs/`。页面只读取 `results/ye_strategy`、`results/etfwin_reference`、`results/comparison` 和 `results/live` 的正式输出。
