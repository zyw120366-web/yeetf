# 排名退出单因素研究

截止2026-09-10；结论：rejected。正式策略和账户未修改。

唯一变化：排名双降退出额外要求当前排名大于5。其余规则固定；没有扫描更多阈值。

| 数据轨道 | 版本 | 累计收益 | 年化 | 夏普 | 最大回撤 | 成交笔数 |
|---|---|---:|---:|---:|---:|---:|
| mixed | 正式对照 | 778.39% | 30.38% | 1.26 | -20.91% | 295 |
| mixed | 前五外才排名退出 | 460.49% | 23.42% | 1.02 | -32.44% | 280 |
| uniform_keyword | 正式对照 | 803.87% | 30.83% | 1.28 | -20.91% | 311 |
| uniform_keyword | 前五外才排名退出 | 474.04% | 23.78% | 1.03 | -32.44% | 300 |

## 预设验收

- mixed_return_sharpe: 未通过
- mixed_drawdown: 未通过
- mixed_double_cost: 未通过
- mixed_starts_4_of_5: 未通过
- mixed_periods_3_of_4: 未通过
- mixed_without_best_round: 未通过
- mixed_small_account: 未通过
- uniform_keyword_return_sharpe: 未通过
- uniform_keyword_drawdown: 未通过
- uniform_keyword_double_cost: 未通过
- uniform_keyword_starts_4_of_5: 未通过
- uniform_keyword_periods_3_of_4: 未通过
- uniform_keyword_without_best_round: 未通过
- uniform_keyword_small_account: 未通过

mixed包含价格回退、关键词代理、AI审核；uniform_keyword在2024年后统一使用历史关键词映射，不冒充AI。两轨道不是独立样本。
分期和不同空仓起点、双倍成本、小资金结果见CSV。trades包含两版本完整成交轮次，position_differences逐日定位持仓差异。
去最大赢家是将各自最大盈利完成持仓执行期间的组合日收益剔除，不是重新运行无该交易的策略；过渡日可能同时剔除其他持仓收益，仅作压力。
本研究未通过则关闭，不围绕该结果继续寻找更优数字；即便通过也不直接晋升实盘。
