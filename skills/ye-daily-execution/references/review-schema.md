# 情绪审核草稿格式

创建一个含 `items` 数组的 JSON 对象。该数组必须对审核队列中的每一个 `source_hash` 恰好包含一条记录，不能多、少或重复。

```json
{
  "review_metadata": {
    "model_family": "GPT-5",
    "model_snapshot": "not_exposed_by_codex",
    "surface": "Codex desktop",
    "reviewed_in_current_conversation": true
  },
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

- `review_metadata` 必须存在。Codex 未暴露精确后端快照时，`model_snapshot` 必须如实写 `not_exposed_by_codex`，不得猜测或伪造。

- `direction` 为 -2 到 2 的整数；`confidence` 和 `novelty` 介于 0 到 1。
- `horizon` 只能是 `intraday`、`1-3d`、`1-4w`、`structural` 或 `unknown`。
- `matched_categories`、`matched_symbols`、`evidence` 与 `risk_flags` 都是字符串数组；不适用时使用空数组。
- 即使 `relevant=false` 也必须填写全部字段；没有合理推断时使用中性值（`direction: 0`、`horizon: "unknown"`）。
- `matched_symbols` 必须由冻结行的公司名、标题或正文中的该 ETF 专属关键词直接支持；类别标签不得自动扩散到同类别全部 ETF。例如黄金、石油或泛商品资讯不能作为豆粕 ETF 的直接证据。
- 同一公司在不同来源重复出现时仍逐行审核；提交程序会保留每行并写入统一事件键，下游题材数量只计一次。
