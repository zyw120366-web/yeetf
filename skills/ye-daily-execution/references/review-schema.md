# 情绪审核草稿格式

创建一个含 `items` 数组的 JSON 对象。该数组必须对审核队列中的每一个 `source_hash` 恰好包含一条记录，不能多、少或重复。

```json
{
  "items": [
    {
      "source_hash": "从队列原样复制",
      "relevant": true,
      "matched_categories": ["科技数字"],
      "matched_symbols": ["515050.SH"],
      "direction": 1,
      "horizon": "1-3d",
      "confidence": 0.75,
      "novelty": 0.5,
      "summary": "简短、客观的中文事实摘要。",
      "evidence": ["冻结行中支持判断的事实。"],
      "risk_flags": ["归因不确定"]
    }
  ]
}
```

约束：

- `direction` 为 -2 到 2 的整数；`confidence` 和 `novelty` 介于 0 到 1。
- `horizon` 只能是 `intraday`、`1-3d`、`1-4w`、`structural` 或 `unknown`。
- `matched_categories`、`matched_symbols`、`evidence` 与 `risk_flags` 都是字符串数组；不适用时使用空数组。
- 即使 `relevant=false` 也必须填写全部字段；没有合理推断时使用中性值（`direction: 0`、`horizon: "unknown"`）。
