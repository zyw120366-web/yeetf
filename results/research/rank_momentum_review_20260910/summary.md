# 排名下降与自身动量分离研究

截至2026-09-10；44次固定对照；结论：not_promoted。正式策略未改变。

只在排名双降、但自身动量分相比5日前和20日前均未下降时，暂缓排名卖出；其他退出照旧。

| 口径 | 版本 | 累计收益 | 年化 | 夏普 | 最大回撤 |
|---|---|---:|---:|---:|---:|
| mixed | 正式对照 | 778.39% | 30.38% | 1.26 | -20.91% |
| mixed | 自身动量保护 | 853.78% | 31.69% | 1.30 | -20.98% |
| uniform_keyword | 正式对照 | 803.87% | 30.83% | 1.28 | -20.91% |
| uniform_keyword | 自身动量保护 | 881.35% | 32.15% | 1.31 | -20.98% |

## 预设验收

- mixed_return_sharpe：通过
- mixed_drawdown：通过
- mixed_double_cost：通过
- mixed_starts_4_of_5：未通过
- mixed_periods_3_of_4：未通过
- mixed_without_best_round：通过
- mixed_small_account：通过
- mixed_events_8_years_3：未通过
- mixed_median_event_positive：通过
- uniform_keyword_return_sharpe：通过
- uniform_keyword_drawdown：通过
- uniform_keyword_double_cost：通过
- uniform_keyword_starts_4_of_5：未通过
- uniform_keyword_periods_3_of_4：未通过
- uniform_keyword_without_best_round：通过
- uniform_keyword_small_account：通过
- uniform_keyword_events_8_years_3：未通过
- uniform_keyword_median_event_positive：通过

## 触发与适用性

- mixed：原规则排名退出中5次自身双窗口动量未降；完整策略形成5个已完成信号分叉区间，涉及4个开始年份，区间收益差中位数+2.00%。
- uniform_keyword：原规则排名退出中5次自身双窗口动量未降；完整策略形成5个已完成信号分叉区间，涉及4个开始年份，区间收益差中位数+2.00%。

上一轮同入场完成交易的出口差异见previous_top5_matched_exits.csv，匹配样本不能覆盖全部路径差。
signal_episodes按信号目标不同到重新相同划分，并整体后移到次日开盘所在日，比较两条真实回测净值的同日历区间收益；部分成交可能跨过区间边界，不能据此做独立成交因果归因。
候选原规则触发点不等于它自己完整持仓路径的事件数。不同起点、两数据轨道、区间样本均非相互独立。去最大赢家沿用上一轮整个执行期日收益剔除，只作压力，非重新回测反事实。
历史已反复观察，不称样本外。未通过不晋升、不再围绕此候选改5/20窗口；只保留证据，不增加每日步骤。
