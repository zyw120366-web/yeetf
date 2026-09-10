# 实际成交确认格式

仅在用户已提供券商成交回单或明确确认后，创建 `results/live/YYYY-MM-DD_actual_fills.json`。`YYYY-MM-DD` 是订单计划的信号日，不是成交日。

```json
{
  "signal_date": "YYYY-MM-DD",
  "confirmation_status": "complete",
  "source": "用户根据券商成交回单确认",
  "fills": [
    {
      "side": "buy",
      "symbol": "510300.SH",
      "status": "filled",
      "quantity": 1000,
      "price": 4.123,
      "broker_order_id": "可选"
    }
  ],
  "note": "若部分成交或撤单，必须如实说明。"
}
```

约束：

- `confirmation_status` 只能是 `complete`、`pending` 或 `exception`。
- `status` 只能是 `filled`、`partial`、`unfilled` 或 `cancelled`。
- `side` 和 `symbol` 必须与对应订单计划完全一致；不可补造订单。
- 只有全部计划订单均为 `filled` 且 `confirmation_status=complete` 时，对账状态才为 `confirmed`。部分成交、未成交、撤单或未确认先记为异常或待确认，禁止下一次新增仓位。
- 唯一恢复通道：上一计划全部为`cancelled`或`unfilled`且数量均为0，用户又明确确认当前账户日期、持仓、现金、权益及无待处理订单，并要求从该账户继续时，可运行`reconcile_actual_fills.py --accept-account-baseline-date 当前信号日`生成`baseline_confirmed`。原取消记录不得改写；部分成交绝不适用。
