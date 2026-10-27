"""Shared, cash-constrained execution arithmetic; no account equity inference."""
import math


def affordable_quantity(cash, price, commission_rate, minimum_commission, lot=100):
    values = (cash, price, commission_rate, minimum_commission)
    if any(not math.isfinite(float(x)) for x in values) or min(cash, commission_rate, minimum_commission) < 0 or price <= 0 or lot <= 0:
        raise ValueError("买入资金、价格或费用无效")
    quantity = math.floor(cash / (price * (1 + commission_rate)) / lot) * lot
    while quantity > 0 and quantity * price + max(minimum_commission, quantity * price * commission_rate) > cash:
        quantity -= lot
    return quantity


def proceeds_buy_quantity(plan, sell_fill, buy_price, broker_available):
    """No buy until the whole old position is sold; spend only verified net proceeds."""
    if plan.get("execution", {}).get("funding_scope") != "sale_proceeds_only":
        raise ValueError("不是卖出款专用换仓计划")
    sell, buy = plan["execution"]["orders"]
    if (sell["side"] != "sell" or buy["side"] != "buy"
            or sell_fill.get("side") != "sell" or sell_fill.get("symbol") != sell["symbol"]
            or sell_fill.get("status") != "filled"
            or sell_fill.get("quantity") != sell["confirmed_quantity"]):
        raise ValueError("旧仓未全部确认成交，禁止买入新仓")
    price, fee = float(sell_fill["price"]), float(sell_fill["fee"])
    if not math.isfinite(price) or price <= 0 or not math.isfinite(fee) or fee < 0:
        raise ValueError("卖出成交价或实际费用无效")
    net = price * sell_fill["quantity"] - fee
    if not math.isfinite(broker_available) or broker_available < 0 or net < 0:
        raise ValueError("实际可用资金无效")
    cost = buy["cost_parameters"]
    return affordable_quantity(min(net, broker_available), buy_price,
                               cost["commission_rate"], cost["minimum_commission"], cost["lot_size"])
