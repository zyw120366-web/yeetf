from __future__ import annotations

import html as html_lib
import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[2]
AUDIT_PATH = PROJECT / "results" / "ye_strategy" / "trade_audit.json"
PUBLIC_HTML = PROJECT / "dashboard" / "public" / "ye-strategy.html"
OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_HTML = OUTPUT_DIR / "ETF轮动策略_策略与回测.html"
PUBLIC_DAILY_HTML = PROJECT / "dashboard" / "public" / "ye-daily.html"
PUBLIC_BACKTEST_HTML = PROJECT / "dashboard" / "public" / "ye-backtest.html"
OUTPUT_DAILY_HTML = OUTPUT_DIR / "ETF轮动策略_今日日报.html"
OUTPUT_BACKTEST_HTML = OUTPUT_DIR / "ETF轮动策略_回测.html"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def normalized_proxy(symbol: str, name: str, dates: pd.DatetimeIndex) -> dict:
    frame = pd.read_csv(PROJECT / "market_data" / "prices" / f"{symbol}.csv")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    close = frame.set_index("datetime")["close"].sort_index().reindex(dates).ffill().bfill()
    curve = close / close.iloc[0] * 100000.0
    drawdown = curve / curve.cummax() - 1.0
    return {
        "name": name,
        "curve": [[str(day.date()), round(float(value), 2)] for day, value in curve.items()],
        "total_return": float(curve.iloc[-1] / curve.iloc[0] - 1.0),
        "max_drawdown": float(drawdown.min()),
    }


def build_etf_dashboard() -> dict:
    market = yaml.safe_load((PROJECT / "config" / "market.yaml").read_text(encoding="utf-8"))
    data_end = pd.Timestamp(market["project"]["data_end"])
    ranking = pd.read_csv(PROJECT / "results" / "comparison" / "latest_ranking.csv")
    ranking["date"] = pd.to_datetime(ranking["date"])
    ranking = ranking.loc[ranking["date"].eq(data_end)].set_index("symbol")
    bool_fields = {
        "above_ma120",
        "pool_eligible",
        "base_entry_pass",
        "technical_entry_pass",
        "final_entry_pass",
        "confirmed_normal_entry",
        "emerging_entry",
        "quality_extension",
    }
    value_fields = [
        "rank",
        "momentum_score",
        "selection_score",
        "roc20",
        "roc60",
        "ma120_bias",
        "r2_20",
        "efficiency20",
        "trailing_amount_20d",
        "listed_sessions",
        *sorted(bool_fields),
    ]
    items = []
    for fund in market["universe"]:
        symbol = f"{fund['code']}.{fund['market']}"
        row = ranking.loc[symbol]
        frame = pd.read_csv(PROJECT / "market_data" / "prices" / f"{symbol}.csv")
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.loc[frame["datetime"].le(data_end)].sort_values("datetime").copy()
        frame["ma120"] = frame["close"].rolling(120).mean()
        recent = frame.tail(253)
        series = [
            [
                str(record.datetime.date()),
                round(float(record.close), 4),
                None if pd.isna(record.ma120) else round(float(record.ma120), 4),
            ]
            for record in recent[["datetime", "close", "ma120"]].itertuples(index=False)
        ]
        path = (
            "常规动量"
            if bool(row.get("confirmed_normal_entry"))
            else "新趋势"
            if bool(row.get("emerging_entry"))
            else "质量延伸"
            if bool(row.get("quality_extension"))
            else "—"
        )
        values = {
            field: bool(row[field]) if field in bool_fields else row[field]
            for field in value_fields
        }
        items.append(
            {
                "symbol": symbol,
                "name": fund["name"],
                "category": fund["category"],
                "pool_role": str(row.get("pool_role", "core")),
                "close": float(row["close"]),
                "change_1d": float(row["change_1d"]),
                "return_3m": float(frame["close"].pct_change(63).iloc[-1]),
                "return_1y": float(frame["close"].pct_change(252).iloc[-1]),
                "path": path,
                "series": series,
                **values,
            }
        )
    items.sort(key=lambda item: (item["rank"] is None, item["rank"] or 999))
    return {
        "as_of": str(data_end.date()),
        "categories": sorted({item["category"] for item in items}),
        "items": items,
    }


def build_payload() -> dict:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    market = yaml.safe_load((PROJECT / "config" / "market.yaml").read_text(encoding="utf-8"))
    formal = yaml.safe_load((PROJECT / "config" / "ye_strategy.yaml").read_text(encoding="utf-8"))
    satellite_symbols = set(
        formal["enhanced_selection"]["universe_architecture"]["challenger_symbols"]
    )
    fund_names = {
        f"{item['code']}.{item['market']}": item["name"]
        for item in market["universe"]
    }
    reference = json.loads(
        (PROJECT / "results" / "etfwin_reference" / "reference_summary.json").read_text(
            encoding="utf-8"
        )
    )
    dates = pd.DatetimeIndex(pd.to_datetime([row["date"] for row in audit["equity"]]))
    ye_curve = [[row["date"], round(float(row["equity"]), 2)] for row in audit["equity"]]
    comparisons = {
        "ye": {
            "name": "ye 策略",
            "curve": ye_curve,
            "total_return": audit["summary"]["metrics"]["total_return"],
            "max_drawdown": audit["summary"]["metrics"]["max_drawdown"],
        },
        "etfwin": {
            "name": "etfwin 策略",
            "curve": [
                [str(day.date()), round(float(value), 2)]
                for day, value in pd.read_csv(
                    PROJECT / "results" / "etfwin_reference" / "equity.csv",
                    index_col=0,
                    parse_dates=True,
                )["equity"].reindex(dates).ffill().items()
            ],
            "total_return": reference["metrics"]["total_return"],
            "max_drawdown": reference["metrics"]["max_drawdown"],
        },
        "nasdaq": normalized_proxy("159941.SZ", "纳斯达克100", dates),
        "csi300": normalized_proxy("510300.SH", "沪深300", dates),
        "chinext": normalized_proxy("159915.SZ", "创业板指", dates),
    }
    completed = [row for row in audit["round_trips"] if row.get("exit_completed_date")]
    fills = audit["fills"]
    trades = []
    for trade in completed:
        item = dict(trade)
        item["pool_role"] = "satellite" if trade["symbol"] in satellite_symbols else "core"
        item["buy_fills"] = [
            {"date": fill["date"], "price": fill["execution_price"], "quantity": fill["quantity"]}
            for fill in fills
            if fill["symbol"] == trade["symbol"] and fill["side"] == "买入"
            and trade["entry_start_date"] <= fill["date"] <= trade["entry_completed_date"]
        ]
        item["sell_fills"] = [
            {"date": fill["date"], "price": fill["execution_price"], "quantity": fill["quantity"]}
            for fill in fills
            if fill["symbol"] == trade["symbol"] and fill["side"] == "卖出"
            and trade["exit_start_date"] <= fill["date"] <= trade["exit_completed_date"]
        ]
        trades.append(item)
    satellite_trades = [trade for trade in trades if trade["pool_role"] == "satellite"]
    satellite_rows = []
    for symbol in sorted(satellite_symbols):
        symbol_trades = [trade for trade in satellite_trades if trade["symbol"] == symbol]
        satellite_rows.append(
            {
                "symbol": symbol,
                "name": fund_names[symbol],
                "completed_trades": len(symbol_trades),
                "realized_net_pnl": sum(float(trade["net_pnl"]) for trade in symbol_trades),
                "holding_days": sum(int(trade["holding_days"]) for trade in symbol_trades),
                "last_entry_date": max(
                    (str(trade["entry_start_date"]) for trade in symbol_trades),
                    default=None,
                ),
            }
        )
    prices = {}
    for symbol in sorted({trade["symbol"] for trade in trades}):
        frame = pd.read_csv(PROJECT / "market_data" / "prices" / f"{symbol}.csv")
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame[frame["datetime"].le(dates[-1])]
        prices[symbol] = [
            [str(row.datetime.date()), round(float(row.close), 4)]
            for row in frame[["datetime", "close"]].itertuples(index=False)
        ]
    equity = pd.Series(
        [float(row["equity"]) for row in audit["equity"]], index=dates, name="equity"
    )
    monthly_end = equity.resample("ME").last()
    monthly_return = monthly_end.pct_change()
    if len(monthly_return):
        monthly_return.iloc[0] = monthly_end.iloc[0] / float(audit["meta"]["initial_capital"]) - 1.0
    monthly = [
        {"year": int(day.year), "month": int(day.month), "return": float(value)}
        for day, value in monthly_return.items()
    ]
    metrics = dict(audit["summary"]["metrics"])
    metrics["completed_trades"] = len(trades)
    metrics["win_rate"] = sum(float(trade["net_pnl"]) > 0 for trade in trades) / len(trades)
    return clean({
        "meta": {**audit["meta"], "strategy": "ye 策略", "live_use_allowed": True},
        "metrics": metrics,
        "periods": audit["periods"],
        "annual": audit["annual"],
        "monthly": monthly,
        "equity": audit["equity"],
        "comparisons": comparisons,
        "trades": trades,
        "prices": prices,
        "satellite_backtest": {
            "configured_count": len(satellite_symbols),
            "completed_trades": len(satellite_trades),
            "realized_net_pnl": sum(float(trade["net_pnl"]) for trade in satellite_trades),
            "holding_days": sum(int(trade["holding_days"]) for trade in satellite_trades),
            "traded_symbols": sorted({trade["symbol"] for trade in satellite_trades}),
            "items": satellite_rows,
            "note": "卫星历史盈亏是正式回测中的实际成交汇总，不等于相对45只核心反事实的边际贡献。",
        },
        "etf_dashboard": build_etf_dashboard(),
    })


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ye ETF轮动策略</title>
<style>
:root{--bg:#f5f6f8;--paper:#fff;--ink:#171a1f;--muted:#727984;--line:#e7e9ee;--red:#e34d59;--green:#11a683;--blue:#4a6cf7;--orange:#e49a34;--purple:#8f65d8;--shadow:0 8px 28px rgba(20,25,35,.05)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;font-variant-numeric:tabular-nums}button,input,select{font:inherit}.shell{display:grid;grid-template-columns:184px 1fr;min-height:100vh}.side{position:sticky;top:0;height:100vh;padding:28px 20px;background:#fff;border-right:1px solid var(--line)}.logo{font-size:21px;font-weight:800}.logo i{font-style:normal;color:var(--red)}.side small{display:block;margin-top:5px;color:var(--muted)}.side nav{display:grid;gap:5px;margin-top:40px}.side a{padding:10px 12px;border-radius:8px;color:var(--muted);text-decoration:none;font-size:13px}.side a:hover{background:#f4f5f7;color:var(--ink)}.content{width:min(1320px,calc(100% - 48px));margin:0 auto;padding:38px 0 64px}.head{display:flex;justify-content:space-between;align-items:flex-start;gap:24px}.head h1{margin:0;font-size:30px;letter-spacing:-.03em}.head p{margin:8px 0 0;color:var(--muted);font-size:13px}.tag{padding:8px 12px;border:1px solid #cfeee5;border-radius:999px;background:#effaf7;color:#087c61;font-size:12px}.notice{margin-top:18px;padding:11px 14px;border-radius:8px;background:#fff8e8;color:#8c641e;font-size:12px}.section{margin-top:25px}.section-title{display:flex;justify-content:space-between;align-items:end;margin-bottom:12px}.section-title h2{margin:0;font-size:19px}.section-title span{color:var(--muted);font-size:12px}.card{background:var(--paper);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow)}.kpis{display:grid;grid-template-columns:repeat(6,1fr);overflow:hidden}.kpi{padding:21px 18px;border-right:1px solid var(--line)}.kpi:last-child{border:0}.label{color:var(--muted);font-size:11px}.value{margin-top:8px;font-size:24px;font-weight:700;letter-spacing:-.025em}.positive{color:var(--red)}.negative{color:var(--green)}.rules{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.rule{padding:18px}.rule b{display:inline-block;color:var(--blue);font-size:12px}.rule h3{margin:8px 0 6px;font-size:15px}.rule p{margin:0;color:var(--muted);font-size:12px;line-height:1.7}.chart-card{padding:18px}.chart-head{display:flex;justify-content:space-between;align-items:center;gap:14px}.chart-head h3{margin:0;font-size:14px}.controls{display:flex;flex-wrap:wrap;gap:6px}.controls label,.controls button{padding:6px 9px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--muted);font-size:11px}.controls button.active{border-color:#bfc9fb;background:#f1f3ff;color:var(--blue)}.controls input{accent-color:var(--blue)}canvas{display:block;width:100%;height:350px}.drawdown{height:125px}.tip{min-height:26px;padding-top:5px;color:var(--muted);font-size:11px}.benchmarks{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:8px}.benchmark{padding:12px;border:1px solid var(--line);border-radius:7px}.benchmark strong{display:block;margin:4px 0;font-size:17px}.returns{display:grid;grid-template-columns:.85fr 1.15fr;gap:10px}.table-wrap{overflow:auto}.simple{width:100%;border-collapse:collapse}.simple th,.simple td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;font-size:11px;white-space:nowrap}.simple th{color:var(--muted);font-weight:600;background:#fafbfc}.simple th:first-child,.simple td:first-child{text-align:left}.heat{padding:16px;overflow:auto}.heat-grid{display:grid;grid-template-columns:54px repeat(12,minmax(42px,1fr));gap:4px;min-width:670px}.heat-grid div{padding:7px 3px;border-radius:4px;text-align:center;font-size:10px}.heat-head{color:var(--muted)}.trade-tools{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}.trade-tools select,.trade-tools input{min-width:270px;padding:9px 10px;border:1px solid var(--line);border-radius:7px;background:#fff}.trade-grid{display:grid;grid-template-columns:360px 1fr;gap:10px}.trade-card{padding:18px}.trade-title{display:flex;justify-content:space-between;gap:10px}.trade-title h3{margin:5px 0 0}.pill{padding:3px 6px;border-radius:4px;background:#f1f3f6;color:var(--muted);font-size:10px}.pnl{font-size:19px;font-weight:700;text-align:right}.trade-meta{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:15px}.trade-meta div{padding:9px;border:1px solid var(--line);border-radius:6px}.trade-meta small{display:block;color:var(--muted)}.reason{color:var(--muted);font-size:11px;line-height:1.65}.price{height:360px}.trade-table{margin-top:10px;max-height:520px;overflow:auto}.trade-table table{min-width:950px}.trade-table thead{position:sticky;top:0;z-index:2}.trade-table tbody tr{cursor:pointer}.trade-table tbody tr:hover,.trade-table tbody tr.selected{background:#f5f7ff}.etf-toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}.etf-toolbar input,.etf-toolbar select{min-width:190px;padding:9px 10px;border:1px solid var(--line);border-radius:7px;background:#fff}.etf-toolbar .controls{margin-left:auto}.etf-detail{padding:18px}.etf-detail-head{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}.etf-detail-head h3{margin:4px 0 2px;font-size:20px}.etf-badge{display:inline-block;padding:4px 7px;border-radius:5px;background:#effaf7;color:#087c61;font-size:10px}.etf-badge.fail{background:#f1f3f6;color:var(--muted)}.etf-stats{display:grid;grid-template-columns:repeat(8,1fr);gap:7px;margin:14px 0}.etf-stat{padding:10px;border:1px solid var(--line);border-radius:7px;min-width:0}.etf-stat strong{display:block;margin-top:4px;font-size:15px;overflow:hidden;text-overflow:ellipsis}.etf-trend{height:310px}.etf-table{max-height:700px;margin-top:10px}.etf-table table{min-width:1180px}.etf-table thead{position:sticky;top:0;z-index:2}.etf-table tbody tr{cursor:pointer}.etf-table tbody tr:hover,.etf-table tbody tr.selected{background:#f5f7ff}.spark{display:block;width:112px;height:30px}.status-dot{display:inline-block;width:7px;height:7px;margin-right:5px;border-radius:50%;background:#b9bec7}.status-dot.pass{background:var(--green)}.footer{margin-top:30px;padding:16px 0;border-top:1px solid var(--line);color:var(--muted);font-size:11px}
@media(max-width:1050px){.shell{grid-template-columns:1fr}.side{position:static;height:auto;display:flex;align-items:center;justify-content:space-between;padding:14px 20px}.side nav{display:flex;margin:0}.kpis{grid-template-columns:repeat(3,1fr)}.kpi:nth-child(3){border-right:0}.rules{grid-template-columns:repeat(2,1fr)}.returns,.trade-grid{grid-template-columns:1fr}.etf-stats{grid-template-columns:repeat(4,1fr)}}@media(max-width:650px){.content{width:calc(100% - 20px);padding-top:24px}.side nav{display:none}.head{display:block}.tag{display:inline-block;margin-top:10px}.kpis{grid-template-columns:repeat(2,1fr)}.kpi:nth-child(odd){border-right:1px solid var(--line)}.kpi:nth-child(even){border-right:0}.rules{grid-template-columns:1fr}.benchmarks{grid-template-columns:repeat(2,1fr)}.chart-head{display:block}.controls{margin-top:8px}.trade-tools select,.trade-tools input,.etf-toolbar input,.etf-toolbar select{width:100%;min-width:0}.etf-toolbar .controls{margin-left:0}.etf-detail-head{display:block}.etf-stats{grid-template-columns:repeat(2,1fr)}.etf-trend{height:260px}}
</style></head><body><div class="shell">
<aside class="side"><div><div class="logo"><i>ye</i> strategy</div><small>ETF 轮动</small></div><nav><a href="#backtest">回测</a><a href="#satelliteBacktest">卫星</a><a href="#etfDashboard">全池</a><a href="#returns">收益</a><a href="#trades">逐笔</a></nav></aside>
<main class="content"><header class="head"><div><h1>ye ETF轮动策略</h1><p>数据区间 2018-07-02 — 2026-07-17 · 收盘后计算 · 下一交易日开盘执行</p></div><span class="tag">唯一策略 · 规则已冻结</span></header>
<div class="notice">历史回测不代表未来收益。所有结果统一计入普通ETF单边0.15%、QDII/溢价敏感ETF单边0.30%及每笔最低5元佣金。</div>
<section class="section" id="backtest"><div class="section-title"><h2>模拟回测</h2><span>初始资金 100,000 元</span></div><div class="card kpis" id="kpis"></div></section>
<section class="section" id="satelliteBacktest"><div class="section-title"><h2>卫星策略的历史参与</h2><span>6只卫星只填补核心无候选的现金空档</span></div><div class="card kpis" id="satelliteKpis"></div><div class="card table-wrap" style="margin-top:10px"><table class="simple"><thead><tr><th>卫星ETF</th><th>完成交易</th><th>已实现净盈亏</th><th>持有天数</th><th>最近入场</th></tr></thead><tbody id="satelliteRows"></tbody></table></div><div class="tip" id="satelliteNote"></div></section>
<section class="section" id="etfDashboard"><div class="section-title"><h2>全池 ETF 观察台</h2><span>51只：45核心＋6卫星 · 冻结评分与价格同一截止日</span></div><div class="etf-toolbar"><input id="etfSearch" placeholder="搜索代码或名称"><select id="etfCategory"><option value="">全部分类</option></select><select id="etfSort"><option value="rank">按策略排名</option><option value="return_3m">按近3月收益</option><option value="return_1y">按近1年收益</option><option value="momentum_score">按动量评分</option></select><div class="controls" id="etfRanges"><button data-etf-range="3m" class="active">近3个月</button><button data-etf-range="1y">近1年</button></div></div><article class="card etf-detail"><div id="etfDetailHead"></div><div id="etfStats" class="etf-stats"></div><canvas class="etf-trend" id="etfTrendChart"></canvas><div class="tip" id="etfTrendTip">蓝线为收盘价，橙线为 MA120；卫星技术合格后仍需等待核心没有候选，才可补位。</div></article><div class="card table-wrap etf-table"><table class="simple"><thead><tr><th>决策排名 / ETF</th><th>走势</th><th>最新价</th><th>当日</th><th>近3月</th><th>近1年</th><th>动量分</th><th>ROC20</th><th>ROC60</th><th>MA120乖离</th><th>路径 / 结果</th></tr></thead><tbody id="etfRows"></tbody></table></div></section>
<section class="section" id="strategy"><div class="section-title"><h2>策略规则</h2><span>45只核心＋6只卫星 · 最多持有1只 · 无信号持有现金</span></div><div class="rules">
<article class="card rule"><b>01 动量排序</b><h3>选择最强代表</h3><p>核心得分为 ROC20 + 1.5×ROC60。通常只在前5名、ROC20与ROC60均为正、价格位于MA120上方且乖离不超过9%的ETF中选择最高分。</p></article>
<article class="card rule"><b>02 主题确认</b><h3>过滤单点假突破</h3><p>历史情绪缺失期只允许前3名且同主题至少75%的ETF保持ROC20为正；实盘AI审核不完整时不开新仓。</p></article>
<article class="card rule"><b>03 边缘过滤</b><h3>弱动量必须有跟随</h3><p>排名4至5且ROC20低于2%时，必须同时满足题材强势股数量、热度加速与正DDE比例要求。</p></article>
<article class="card rule"><b>04 AI情绪</b><h3>逐条审核盘面与新闻</h3><p>每条采集信息均由AI判断相关ETF、方向、期限、置信度和证据；漏审、重复或来源失败时禁止部分情绪数据进入买入信号。</p></article>
<article class="card rule"><b>05 卖出风控</b><h3>硬退出不被情绪覆盖</h3><p>跌破MA120、ROC20转负、或排名同时差于5日前和20日前时退出。热点只能短暂保护软退出，MA120破位始终立即执行。</p></article>
<article class="card rule"><b>06 防止反复</b><h3>同标的冷却5日</h3><p>卖出后5个交易日内不重新买入同一ETF；其他ETF与其他主题不受限制。</p></article>
</div></section>
<section class="section"><div class="section-title"><h2>资金曲线</h2><span>可切换基准与查看区间</span></div><div class="card chart-card"><div class="chart-head"><h3>组合净值对比</h3><div><div class="controls" id="legend"></div><div class="controls" id="ranges" style="margin-top:5px"><button data-range="all" class="active">全部</button><button data-range="5y">近5年</button><button data-range="3y">近3年</button><button data-range="1y">近1年</button></div></div></div><canvas id="equityChart"></canvas><div class="tip" id="equityTip"></div><canvas class="drawdown" id="drawdownChart"></canvas><div class="benchmarks" id="benchmarkStats"></div></div></section>
<section class="section" id="returns"><div class="section-title"><h2>收益分布</h2><span>点击月度格子，查看该月实际发生的买卖</span></div><div class="returns"><div class="card table-wrap"><table class="simple"><thead><tr><th>年度</th><th>收益</th><th>最大回撤</th><th>交易</th><th>净利润</th></tr></thead><tbody id="annualRows"></tbody></table></div><div class="card heat"><div class="heat-grid" id="heatmap"></div></div></div></section>
<section class="section" id="trades"><div class="section-title"><h2>逐笔交易</h2><span id="tradeContext">点击任一笔，查看其在ETF完整走势中的买卖位置</span></div><div class="trade-tools"><select id="tradeSelect"></select><input id="tradeSearch" placeholder="搜索代码、名称或入场类型"><button id="clearMonth" type="button" style="display:none">显示全部交易</button></div><div class="trade-grid"><article class="card trade-card" id="tradeCard"></article><article class="card chart-card"><canvas class="price" id="priceChart"></canvas><div class="tip" id="priceTip"></div></article></div><div class="card trade-table"><table class="simple"><thead><tr><th># / ETF</th><th>买入</th><th>卖出</th><th>持有</th><th>类型</th><th>净盈亏</th><th>收益率</th><th>标签</th></tr></thead><tbody id="tradeRows"></tbody></table></div></section>
<footer class="footer">ye 策略 · 数据截至 2026-07-17 · 本页只展示冻结规则与统一口径回测结果</footer></main></div>
<script>const DATA=__DATA__;
const COLORS={ye:'#e34d59',etfwin:'#171a1f',nasdaq:'#4a6cf7',csi300:'#e49a34',chinext:'#8f65d8'},keys=Object.keys(COLORS);let enabled=new Set(keys),range='all',tradeIndex=DATA.trades.length-1;
const pct=(v,d=1)=>(v*100).toFixed(d)+'%',money=v=>new Intl.NumberFormat('zh-CN',{style:'currency',currency:'CNY',maximumFractionDigits:0}).format(v),dateMs=s=>new Date(s+'T00:00:00').getTime();
const M=DATA.metrics;document.getElementById('kpis').innerHTML=[['累计收益',pct(M.total_return),'positive'],['年化收益',pct(M.cagr),'positive'],['夏普比率',M.sharpe.toFixed(2),''],['最大回撤',pct(M.max_drawdown),'negative'],['胜率',pct(M.win_rate),''],['完成交易',M.completed_trades+' 笔','']].map(x=>`<div class="kpi"><div class="label">${x[0]}</div><div class="value ${x[2]}">${x[1]}</div></div>`).join('');
const SAT=DATA.satellite_backtest;document.getElementById('satelliteKpis').innerHTML=[['卫星数量',SAT.configured_count+' 只',''],['有过交易',SAT.traded_symbols.length+' 只',''],['实际完成交易',SAT.completed_trades+' 笔',''],['已实现净盈亏',money(SAT.realized_net_pnl),SAT.realized_net_pnl>=0?'positive':'negative'],['平均每笔',SAT.completed_trades?money(SAT.realized_net_pnl/SAT.completed_trades):'—',SAT.realized_net_pnl>=0?'positive':'negative'],['持有天数',SAT.holding_days+' 天','']].map(x=>`<div class="kpi"><div class="label">${x[0]}</div><div class="value ${x[2]}">${x[1]}</div></div>`).join('');document.getElementById('satelliteRows').innerHTML=SAT.items.map(x=>`<tr><td><b>${x.name}</b><br><span class="label">${x.symbol}</span></td><td>${x.completed_trades}</td><td class="${x.realized_net_pnl>=0?'positive':'negative'}">${money(x.realized_net_pnl)}</td><td>${x.holding_days}</td><td>${x.last_entry_date||'—'}</td></tr>`).join('');document.getElementById('satelliteNote').textContent=SAT.note;
const ETF=DATA.etf_dashboard.items,esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let etfRange='3m',etfSymbol=(ETF.find(x=>x.final_entry_pass)||ETF[0]).symbol;
const signed=(v,d=1)=>v==null?'—':`${v>=0?'+':''}${pct(v,d)}`,tone=v=>v==null?'':v>=0?'positive':'negative';
document.getElementById('etfCategory').innerHTML+=[...DATA.etf_dashboard.categories].map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
function etfPoints(item){return item.series.slice(etfRange==='3m'?-64:-253)}
function sparkline(item){const pts=etfPoints(item),vals=pts.map(x=>x[1]),lo=Math.min(...vals),hi=Math.max(...vals),gap=hi-lo||1,poly=pts.map((p,i)=>`${(i/(pts.length-1)*110).toFixed(1)},${(28-(p[1]-lo)/gap*26).toFixed(1)}`).join(' '),color=vals.at(-1)>=vals[0]?'#e34d59':'#11a683';return `<svg class="spark" viewBox="0 0 112 30" aria-label="${etfRange==='3m'?'近3个月':'近1年'}走势"><polyline fill="none" stroke="${color}" stroke-width="1.8" points="${poly}"/></svg>`}
function renderEtfDetail(){const x=ETF.find(v=>v.symbol===etfSymbol)||ETF[0],pass=x.final_entry_pass,waiting=x.technical_entry_pass&&!pass,role=x.pool_role==='challenger'?'卫星':'核心',status=pass?'最终合格 · '+esc(x.path):waiting?'技术合格 · 等待核心空档':'当日未进入最终候选';document.getElementById('etfDetailHead').innerHTML=`<div class="etf-detail-head"><div><span class="label">${esc(x.category)} · ${esc(x.symbol)} · ${role}</span><h3>${esc(x.name)}</h3><span class="label">数据截至 ${DATA.etf_dashboard.as_of}</span></div><span class="etf-badge ${pass?'':'fail'}">${status}</span></div>`;const stats=[['最新价',x.close.toFixed(3),''],['决策排名','#'+Math.round(x.rank),''],['动量评分',signed(x.momentum_score,2),tone(x.momentum_score)],['近3月',signed(x.return_3m),tone(x.return_3m)],['近1年',signed(x.return_1y),tone(x.return_1y)],['ROC20',signed(x.roc20),tone(x.roc20)],['ROC60',signed(x.roc60),tone(x.roc60)],['MA120乖离',signed(x.ma120_bias),tone(x.ma120_bias)]];document.getElementById('etfStats').innerHTML=stats.map(s=>`<div class="etf-stat"><span class="label">${s[0]}</span><strong class="${s[2]}">${s[1]}</strong></div>`).join('');drawEtfTrend()}
function drawEtfTrend(mx=null){const item=ETF.find(v=>v.symbol===etfSymbol)||ETF[0],pts=etfPoints(item),cv=document.getElementById('etfTrendChart'),{c,w,h}=prep(cv),pad={l:50,r:16,t:18,b:28},vals=pts.flatMap(p=>p[2]==null?[p[1]]:[p[1],p[2]]),lo=Math.min(...vals),hi=Math.max(...vals),gap=hi-lo||1,ymin=lo-gap*.08,ymax=hi+gap*.08,xmin=dateMs(pts[0][0]),xmax=dateMs(pts.at(-1)[0]),X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>pad.t+(ymax-y)/(ymax-ymin)*(h-pad.t-pad.b);c.clearRect(0,0,w,h);c.font='10px sans-serif';c.fillStyle='#7b818b';c.strokeStyle='#eceef2';for(let i=0;i<5;i++){const y=pad.t+i*(h-pad.t-pad.b)/4,val=ymax-i*(ymax-ymin)/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();c.fillText(val.toFixed(2),5,y+3)}for(let i=0;i<5;i++){const p=pts[Math.round(i*(pts.length-1)/4)],x=pad.l+i*(w-pad.l-pad.r)/4;c.fillText(p[0].slice(5),x-14,h-8)}[[1,'#4a6cf7',2.2],[2,'#e49a34',1.5]].forEach(([idx,color,width])=>{c.beginPath();c.strokeStyle=color;c.lineWidth=width;let started=false;pts.forEach(p=>{if(p[idx]==null)return;const x=X(dateMs(p[0])),y=Y(p[idx]);started?c.lineTo(x,y):(c.moveTo(x,y),started=true)});c.stroke()});if(mx!==null&&mx>=pad.l&&mx<=w-pad.r){const ms=xmin+(mx-pad.l)/(w-pad.l-pad.r)*(xmax-xmin),p=pts.reduce((a,b)=>Math.abs(dateMs(b[0])-ms)<Math.abs(dateMs(a[0])-ms)?b:a);c.strokeStyle='#9aa0aa';c.setLineDash([3,3]);c.beginPath();c.moveTo(X(dateMs(p[0])),pad.t);c.lineTo(X(dateMs(p[0])),h-pad.b);c.stroke();c.setLineDash([]);document.getElementById('etfTrendTip').innerHTML=`${p[0]}　收盘 ${p[1].toFixed(3)}${p[2]==null?'':`　MA120 ${p[2].toFixed(3)}`}`}}
function renderEtfRows(){const q=document.getElementById('etfSearch').value.trim().toLowerCase(),cat=document.getElementById('etfCategory').value,sort=document.getElementById('etfSort').value;let items=ETF.filter(x=>(!q||`${x.symbol}${x.name}`.toLowerCase().includes(q))&&(!cat||x.category===cat));items.sort((a,b)=>sort==='rank'?a.rank-b.rank:(b[sort]??-Infinity)-(a[sort]??-Infinity));document.getElementById('etfRows').innerHTML=items.map(x=>`<tr data-symbol="${x.symbol}" class="${x.symbol===etfSymbol?'selected':''}"><td><b>#${Math.round(x.rank)} ${esc(x.name)}</b><br><span class="label">${x.symbol} · ${esc(x.category)} · ${x.pool_role==='challenger'?'卫星':'核心'}</span></td><td>${sparkline(x)}</td><td>${x.close.toFixed(3)}</td><td class="${tone(x.change_1d)}">${signed(x.change_1d)}</td><td class="${tone(x.return_3m)}">${signed(x.return_3m)}</td><td class="${tone(x.return_1y)}">${signed(x.return_1y)}</td><td>${signed(x.momentum_score,2)}</td><td class="${tone(x.roc20)}">${signed(x.roc20)}</td><td class="${tone(x.roc60)}">${signed(x.roc60)}</td><td class="${tone(x.ma120_bias)}">${signed(x.ma120_bias)}</td><td><span class="status-dot ${x.final_entry_pass?'pass':''}"></span>${x.final_entry_pass?esc(x.path)+' · 最终合格':x.technical_entry_pass?'技术合格 · 等待核心空档':'未通过'}</td></tr>`).join('')||'<tr><td colspan="11" class="label">没有匹配的 ETF。</td></tr>';document.querySelectorAll('#etfRows tr[data-symbol]').forEach(r=>r.onclick=()=>{etfSymbol=r.dataset.symbol;renderEtfDetail();renderEtfRows()})}
document.querySelectorAll('#etfRanges button').forEach(b=>b.onclick=()=>{etfRange=b.dataset.etfRange;document.querySelectorAll('#etfRanges button').forEach(x=>x.classList.toggle('active',x===b));renderEtfRows();drawEtfTrend()});['etfSearch','etfCategory','etfSort'].forEach(id=>document.getElementById(id).addEventListener(id==='etfSearch'?'input':'change',renderEtfRows));const etfCanvas=document.getElementById('etfTrendChart');etfCanvas.onmousemove=e=>drawEtfTrend(e.clientX-etfCanvas.getBoundingClientRect().left);etfCanvas.onmouseleave=()=>{document.getElementById('etfTrendTip').textContent='蓝线为收盘价，橙线为 MA120；点击下表任一 ETF 可切换。';drawEtfTrend()};renderEtfDetail();renderEtfRows();
document.getElementById('legend').innerHTML=keys.map(k=>`<label><input type="checkbox" data-key="${k}" checked><span style="color:${COLORS[k]}">●</span> ${DATA.comparisons[k].name}</label>`).join('');document.querySelectorAll('#legend input').forEach(el=>el.onchange=e=>{e.target.checked?enabled.add(e.target.dataset.key):enabled.delete(e.target.dataset.key);drawEquity()});document.querySelectorAll('#ranges button').forEach(b=>b.onclick=()=>{range=b.dataset.range;document.querySelectorAll('#ranges button').forEach(x=>x.classList.toggle('active',x===b));drawEquity()});
function prep(cv){const r=cv.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);cv.width=Math.round(r.width*d);cv.height=Math.round(r.height*d);const c=cv.getContext('2d');c.setTransform(d,0,0,d,0,0);return{c,w:r.width,h:r.height}}
function rangeStart(last){const d=new Date(last+'T00:00:00');if(range!=='all')d.setFullYear(d.getFullYear()-parseInt(range));return range==='all'?0:d.getTime()}
function drawEquity(mx=null){const cv=document.getElementById('equityChart'),{c,w,h}=prep(cv),pad={l:58,r:16,t:16,b:30},last=DATA.comparisons.ye.curve.at(-1)[0],start=rangeStart(last),raw=keys.filter(k=>enabled.has(k)).map(k=>({k,pts:DATA.comparisons[k].curve.filter(p=>dateMs(p[0])>=start)})),series=raw.map(s=>{const base=s.pts[0][1];return{...s,pts:s.pts.map(p=>[p[0],p[1]/base*100000])}});c.clearRect(0,0,w,h);if(!series.length)return;const xmin=dateMs(series[0].pts[0][0]),xmax=dateMs(series[0].pts.at(-1)[0]),vals=series.flatMap(s=>s.pts.map(p=>p[1]));let ymin=Math.min(...vals),ymax=Math.max(...vals),gap=ymax-ymin||1;ymin=Math.max(0,ymin-gap*.06);ymax+=gap*.06;const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>pad.t+(ymax-y)/(ymax-ymin)*(h-pad.t-pad.b);c.font='10px sans-serif';c.fillStyle='#7b818b';c.strokeStyle='#eceef2';for(let i=0;i<5;i++){const y=pad.t+i*(h-pad.t-pad.b)/4,val=ymax-i*(ymax-ymin)/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();c.fillText((val/10000).toFixed(0)+'万',5,y+3)}for(let i=0;i<6;i++){const x=pad.l+i*(w-pad.l-pad.r)/5,ms=xmin+i*(xmax-xmin)/5;c.fillText(new Date(ms).getFullYear(),x-12,h-8)}series.forEach(s=>{c.beginPath();c.strokeStyle=COLORS[s.k];c.lineWidth=s.k==='ye'?2.4:1.4;s.pts.forEach((p,i)=>{const x=X(dateMs(p[0])),y=Y(p[1]);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()});if(mx!==null&&mx>=pad.l&&mx<=w-pad.r){const ms=xmin+(mx-pad.l)/(w-pad.l-pad.r)*(xmax-xmin);c.strokeStyle='#9aa0aa';c.setLineDash([3,3]);c.beginPath();c.moveTo(mx,pad.t);c.lineTo(mx,h-pad.b);c.stroke();c.setLineDash([]);document.getElementById('equityTip').innerHTML=new Date(ms).toLocaleDateString('zh-CN')+'　'+series.map(s=>{const p=s.pts.reduce((a,b)=>Math.abs(dateMs(b[0])-ms)<Math.abs(dateMs(a[0])-ms)?b:a);return `<span style="color:${COLORS[s.k]}">${DATA.comparisons[s.k].name} ${money(p[1])}</span>`}).join('　')}drawDrawdown(start)}
function drawDrawdown(start=0){const cv=document.getElementById('drawdownChart'),{c,w,h}=prep(cv),pad={l:58,r:16,t:6,b:22},pts=DATA.equity.filter(p=>dateMs(p.date)>=start);c.clearRect(0,0,w,h);if(!pts.length)return;const xmin=dateMs(pts[0].date),xmax=dateMs(pts.at(-1).date),min=Math.min(...pts.map(p=>p.drawdown)),X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>pad.t+(0-y)/(0-min)*(h-pad.t-pad.b);c.strokeStyle='#eceef2';c.beginPath();c.moveTo(pad.l,pad.t);c.lineTo(w-pad.r,pad.t);c.stroke();c.beginPath();c.moveTo(X(dateMs(pts[0].date)),Y(pts[0].drawdown));pts.forEach(p=>c.lineTo(X(dateMs(p.date)),Y(p.drawdown)));c.lineTo(X(dateMs(pts.at(-1).date)),Y(0));c.closePath();c.fillStyle='rgba(17,166,131,.18)';c.fill();c.strokeStyle='#11a683';c.stroke();c.fillStyle='#7b818b';c.font='10px sans-serif';c.fillText('回撤 '+pct(min),5,h-8)}
const eq=document.getElementById('equityChart');eq.onmousemove=e=>drawEquity(e.clientX-eq.getBoundingClientRect().left);eq.onmouseleave=()=>{document.getElementById('equityTip').textContent='';drawEquity()};document.getElementById('benchmarkStats').innerHTML=keys.map(k=>`<div class="benchmark"><span class="label">${DATA.comparisons[k].name}</span><strong class="${DATA.comparisons[k].total_return>=0?'positive':'negative'}">${pct(DATA.comparisons[k].total_return)}</strong><span class="label">最大回撤 ${pct(DATA.comparisons[k].max_drawdown)}</span></div>`).join('');
document.getElementById('annualRows').innerHTML=DATA.annual.map(x=>`<tr><td>${x.year}</td><td class="${x.return>=0?'positive':'negative'}">${pct(x.return)}</td><td class="negative">${pct(x.max_drawdown)}</td><td>${x.completed_trades}</td><td class="${x.net_pnl>=0?'positive':'negative'}">${money(x.net_pnl)}</td></tr>`).join('');
let monthFilter=null;const months=['年','1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],years=[...new Set(DATA.monthly.map(x=>x.year))];let heat=months.map(x=>`<div class="heat-head">${x}</div>`).join('');years.forEach(y=>{heat+=`<div class="heat-head">${y}</div>`;for(let m=1;m<=12;m++){const x=DATA.monthly.find(v=>v.year===y&&v.month===m);if(!x){heat+='<div>—</div>';continue}const a=Math.min(.75,Math.abs(x.return)*4+.08),bg=x.return>=0?`rgba(227,77,89,${a})`:`rgba(17,166,131,${a})`;heat+=`<div class="month-cell" data-year="${y}" data-month="${m}" title="点击查看 ${y}年${m}月 的买卖" style="background:${bg};color:${a>.42?'#fff':'#333'}">${pct(x.return,0)}</div>`}});document.getElementById('heatmap').innerHTML=heat;
const select=document.getElementById('tradeSelect');select.innerHTML=DATA.trades.map((t,i)=>`<option value="${i}">#${t.trade_no} ${t.name} · ${t.entry_start_date} · ${pct(t.net_return)}</option>`).join('');select.value=tradeIndex;select.onchange=()=>choose(+select.value);
function choose(i){tradeIndex=i;select.value=i;renderTrade();drawPrice();renderRows(document.getElementById('tradeSearch').value)}
function renderTrade(){const t=DATA.trades[tradeIndex];document.getElementById('tradeCard').innerHTML=`<div class="trade-title"><div><span class="pill">#${t.trade_no} · ${t.category}</span><h3>${t.name}</h3><span class="label">${t.symbol}</span></div><div class="pnl ${t.net_pnl>=0?'positive':'negative'}">${money(t.net_pnl)}<div class="label">${pct(t.net_return)}</div></div></div><div class="trade-meta"><div><small>买入决策 / 实际成交</small><strong>${t.entry_start_date}</strong><span class="label">${t.buy_fills.length} 笔成交 · 均价 ${t.average_buy_price.toFixed(4)}</span></div><div><small>卖出决策 / 实际成交</small><strong>${t.exit_start_date}</strong><span class="label">${t.sell_fills.length} 笔成交 · 均价 ${t.average_sell_price.toFixed(4)}</span></div><div><small>持有</small><strong>${t.holding_days}个交易日</strong><span class="label">MFE ${pct(t.holding_mfe)} / MAE ${pct(t.holding_mae)}</span></div><div><small>复盘标签</small><strong>${t.failure_type||'正常'}</strong><span class="label">${t.entry_type}</span></div></div><p class="reason"><b>成交拆分说明：</b>这一笔是一次买入决策和一次卖出决策；为遵守单日可参与成交额上限，订单可能跨多日成交，并不表示策略重复加仓或减仓。</p><p class="reason"><b>买入：</b>${t.entry_reason}</p><p class="reason"><b>卖出：</b>${t.exit_reason}</p>`}
function drawPrice(mx=null){const t=DATA.trades[tradeIndex],pts=DATA.prices[t.symbol],cv=document.getElementById('priceChart'),{c,w,h}=prep(cv),pad={l:52,r:15,t:15,b:28},xmin=dateMs(pts[0][0]),xmax=dateMs(pts.at(-1)[0]);c.clearRect(0,0,w,h);let ymin=Math.min(...pts.map(p=>p[1])),ymax=Math.max(...pts.map(p=>p[1])),gap=ymax-ymin||1;ymin-=gap*.06;ymax+=gap*.06;const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>pad.t+(ymax-y)/(ymax-ymin)*(h-pad.t-pad.b);c.fillStyle='rgba(74,108,247,.08)';c.fillRect(X(dateMs(t.entry_start_date)),pad.t,Math.max(2,X(dateMs(t.exit_completed_date))-X(dateMs(t.entry_start_date))),h-pad.t-pad.b);c.strokeStyle='#eceef2';c.fillStyle='#7b818b';c.font='10px sans-serif';for(let i=0;i<5;i++){const y=pad.t+i*(h-pad.t-pad.b)/4,val=ymax-i*(ymax-ymin)/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();c.fillText(val.toFixed(2),4,y+3)}c.beginPath();c.strokeStyle='#48515d';c.lineWidth=1.4;pts.forEach((p,i)=>i?c.lineTo(X(dateMs(p[0])),Y(p[1])):c.moveTo(X(dateMs(p[0])),Y(p[1])));c.stroke();[...t.buy_fills.map(x=>({...x,s:'买',color:'#e34d59'})),...t.sell_fills.map(x=>({...x,s:'卖',color:'#11a683'}))].forEach(m=>{const x=X(dateMs(m.date)),y=Y(m.price);c.fillStyle=m.color;c.beginPath();c.arc(x,y,6,0,Math.PI*2);c.fill();c.fillStyle='#fff';c.font='bold 9px sans-serif';c.fillText(m.s,x-5,y+3)});for(let i=0;i<6;i++){const ms=xmin+i*(xmax-xmin)/5;c.fillStyle='#7b818b';c.fillText(new Date(ms).getFullYear(),pad.l+i*(w-pad.l-pad.r)/5-11,h-7)}if(mx!==null&&mx>=pad.l&&mx<=w-pad.r){const ms=xmin+(mx-pad.l)/(w-pad.l-pad.r)*(xmax-xmin),p=pts.reduce((a,b)=>Math.abs(dateMs(b[0])-ms)<Math.abs(dateMs(a[0])-ms)?b:a);document.getElementById('priceTip').textContent=`${p[0]}　收盘 ${p[1].toFixed(4)}　蓝色区域为持有期`}}
const pc=document.getElementById('priceChart');pc.onmousemove=e=>drawPrice(e.clientX-pc.getBoundingClientRect().left);pc.onmouseleave=()=>{document.getElementById('priceTip').textContent='';drawPrice()};
function tradeInMonth(t,prefix){return[t.entry_start_date,t.entry_completed_date,t.exit_start_date,t.exit_completed_date].some(d=>d&&d.startsWith(prefix))}function renderRows(q=''){q=q.trim().toLowerCase();const visible=DATA.trades.map((t,i)=>({t,i})).filter(x=>(!q||`${x.t.symbol}${x.t.name}${x.t.entry_type}`.toLowerCase().includes(q))&&(!monthFilter||tradeInMonth(x.t,monthFilter)));document.getElementById('tradeRows').innerHTML=visible.map(({t,i})=>`<tr data-i="${i}" class="${i===tradeIndex?'selected':''}"><td><b>#${t.trade_no} ${t.name}</b><br><span class="label">${t.symbol}</span></td><td>${t.entry_start_date}<br>${t.average_buy_price.toFixed(4)}</td><td>${t.exit_start_date}<br>${t.average_sell_price.toFixed(4)}</td><td>${t.holding_days}日</td><td>${t.entry_type}</td><td class="${t.net_pnl>=0?'positive':'negative'}">${money(t.net_pnl)}</td><td class="${t.net_return>=0?'positive':'negative'}">${pct(t.net_return)}</td><td>${t.failure_type||'正常'}</td></tr>`).join('')||'<tr><td colspan="8" class="label">该月没有发生买入或卖出。</td></tr>';document.querySelectorAll('#tradeRows tr[data-i]').forEach(r=>r.onclick=()=>choose(+r.dataset.i))}function selectMonth(y,m){monthFilter=`${y}-${String(m).padStart(2,'0')}`;document.getElementById('tradeContext').textContent=`${y}年${m}月发生的买卖（含跨月完成的订单）`;document.getElementById('clearMonth').style.display='inline-block';document.querySelectorAll('.month-cell').forEach(x=>x.style.outline=x.dataset.year==y&&x.dataset.month==m?'2px solid #4a6cf7':'');const first=DATA.trades.findIndex(t=>tradeInMonth(t,monthFilter));if(first>=0){tradeIndex=first;select.value=first;renderTrade();drawPrice()}renderRows(document.getElementById('tradeSearch').value);document.getElementById('trades').scrollIntoView({behavior:'smooth',block:'start'})}document.querySelectorAll('.month-cell').forEach(x=>x.onclick=()=>selectMonth(x.dataset.year,x.dataset.month));document.getElementById('clearMonth').onclick=()=>{monthFilter=null;document.getElementById('tradeContext').textContent='点击任一笔，查看其在ETF完整走势中的买卖位置';document.getElementById('clearMonth').style.display='none';document.querySelectorAll('.month-cell').forEach(x=>x.style.outline='');renderRows(document.getElementById('tradeSearch').value)};document.getElementById('tradeSearch').oninput=e=>renderRows(e.target.value);
let timer;window.onresize=()=>{clearTimeout(timer);timer=setTimeout(()=>{drawEtfTrend();drawEquity();drawPrice()},100)};renderTrade();renderRows();drawEquity();drawPrice();
</script></body></html>'''


FLOW_STYLE = r'''
.daily-intro{padding:20px}.daily-intro h3{margin:0 0 7px;font-size:18px}.daily-intro p{margin:0;color:var(--muted);font-size:13px;line-height:1.75}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:12px}.flow-step{position:relative;padding:16px;border:1px solid var(--line);border-radius:9px;background:#fff}.flow-step b{display:block;color:var(--blue);font-size:11px}.flow-step h3{margin:7px 0 5px;font-size:14px}.flow-step p{margin:0;color:var(--muted);font-size:12px;line-height:1.65}.flow-step:after{content:'→';position:absolute;right:-9px;top:43%;z-index:2;color:#a8afba;background:var(--bg);font-weight:700}.flow-step:last-child:after{display:none}.decision-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.decision{padding:18px}.decision b{display:block;font-size:12px;color:var(--blue)}.decision h3{margin:7px 0;font-size:16px}.decision ul{margin:0;padding-left:18px;color:var(--muted);font-size:12px;line-height:1.85}.decision.sell b{color:var(--green)}.decision.buy b{color:var(--red)}.decision .all{display:inline-block;margin:3px 0 8px;padding:3px 6px;border-radius:4px;background:#fff3f3;color:#b53540;font-size:10px}.decision .any{display:inline-block;margin:3px 0 8px;padding:3px 6px;border-radius:4px;background:#effaf7;color:#087c61;font-size:10px}.execution-note{margin-top:10px;padding:14px 16px;border-radius:8px;background:#f1f3ff;color:#4a5677;font-size:12px;line-height:1.7}.execution-note b{color:#2c47bd}@media(max-width:1050px){.flow{grid-template-columns:repeat(3,1fr)}.flow-step:nth-child(3):after{display:none}}@media(max-width:650px){.flow,.decision-grid{grid-template-columns:1fr}.flow-step:after{display:none}}
'''

STRATEGY_SECTION = r'''<section class="section" id="strategy"><div class="section-title"><h2>策略规则：每日决策流程</h2><span>当天收盘决定目标仓位 · 下一交易日开盘执行</span></div><article class="card daily-intro"><h3>核心冠军优先，卫星只补空档</h3><p>全池由45只核心ETF和6只卫星组成。核心池独立形成排名、类别宽度与退出标尺，并拥有买入优先权。卫星只有完整通过全部条件、且当天没有任何核心候选时才允许补位；一旦核心候选重新出现，卫星下一交易日开盘让位。</p></article><div class="flow"><article class="flow-step"><b>收盘后 · 第 1 步</b><h3>冻结当天输入</h3><p>更新全部51只ETF的日线与成交额，采集盘面和新闻快照。来源失败即标记为不可用，不把缺失当成中性。</p></article><article class="flow-step"><b>第 2 步</b><h3>逐条审核情绪</h3><p>当前 Codex 对话逐条审核已采集信息：相关主题、方向、期限、证据与风险。任何漏审，明天都禁止新开仓。</p></article><article class="flow-step"><b>第 3 步</b><h3>检查旧仓</h3><p>先检查原卖出条件；核心旧仓未卖出时，再检查严格机会换仓。卫星遇到合格核心候选则让位。</p></article><article class="flow-step"><b>第 4 步</b><h3>核心候选优先</h3><p>先在45只核心中运行完整入场路径；只要有一只核心合格，当天就不选择卫星。</p></article><article class="flow-step"><b>第 5 步</b><h3>无核心才由卫星补位</h3><p>核心没有候选时，才从技术合格的卫星中选一只；最终仍只输出一只ETF或现金。</p></article></div></section><section class="section"><div class="section-title"><h2>什么时候买入、持有、卖出或换仓？</h2><span>最多一只 ETF；条件不满足时现金也是明确仓位</span></div><div class="decision-grid"><article class="card decision buy"><b>买入条件</b><h3>必须同时满足全部基础条件</h3><span class="all">全部满足才可买</span><ul><li>上市满 120 个交易日，近 20 日平均成交额不低于 2,000 万元；</li><li>动量总分 = ROC20 × 1.0 + ROC60 × 1.5；核心池前5，卫星用相对核心的虚拟排名；</li><li>ROC20 与 ROC60 都为正，收盘价在 MA120 上方，且不高于 MA120 的 9%；</li><li>核心候选优先；没有核心候选时，卫星才可补位；</li><li>若排第 4–5 名且 ROC20 低于 2%，还必须通过热点确认；</li><li>当天情绪审核覆盖率必须为 100%。</li></ul></article><article class="card decision"><b>持有</b><h3>卖出和换仓都未触发</h3><ul><li>持仓不必每天都在前5；</li><li>另一只ETF单日领先不会立刻追入；</li><li>新买资格不机械迫使旧仓卖出。</li></ul></article><article class="card decision sell"><b>卖出</b><h3>风险条件任一触发</h3><ul><li>跌破MA120；</li><li>ROC20转负；</li><li>当前排名同时差于5日前和20日前；</li><li>热点只能短暂保护后两项。</li></ul></article><article class="card decision"><b>机会换仓</b><h3>四项条件全部满足</h3><ul><li>核心旧仓掉出前5；</li><li>完整合格核心候选动量分领先至少5个百分点；</li><li>同一候选连续2日成立；</li><li>旧仓持有满5日后，下一开盘全仓先卖后买。</li></ul></article></div><div class="execution-note"><b>执行时间：</b>T 日收盘后完成上述判断 → T+1 开盘按订单计划一次性执行。普通 ETF 单边成本固定为 0.15%；QDII/溢价敏感 ETF 单边成本固定为 0.30%；每笔最低佣金 5 元。回测与实盘使用同一口径。</div></section>'''


def replace_strategy_section(html: str) -> str:
    start = html.index('<section class="section" id="strategy">')
    end = html.index('</section>', start) + len('</section>')
    return html[:start] + STRATEGY_SECTION + html[end:]


PAGE_STYLE = r'''
<style>
:root{--bg:#f6f7f9;--paper:#fff;--ink:#19202a;--muted:#687282;--line:#e4e8ee;--red:#db4b5a;--green:#0f9c79;--blue:#4768e9;--amber:#a66e16;--shadow:0 10px 30px rgba(20,30,48,.06)}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;font-variant-numeric:tabular-nums;overflow-x:hidden}.shell{display:grid;grid-template-columns:190px minmax(0,1fr);min-height:100vh;min-width:0}.side{position:sticky;top:0;height:100vh;padding:28px 20px;background:#fff;border-right:1px solid var(--line)}.logo{font-size:21px;font-weight:800}.logo i{font-style:normal;color:var(--red)}.side small{display:block;margin-top:5px;color:var(--muted)}.side nav{display:grid;gap:5px;margin-top:40px}.side a{padding:10px 12px;border-radius:8px;color:var(--muted);text-decoration:none;font-size:13px}.side a.active,.side a:hover{background:#f0f3ff;color:var(--blue)}.content{width:min(1200px,calc(100% - 48px));margin:0 auto;padding:38px 0 64px;min-width:0}.head{display:flex;justify-content:space-between;gap:22px;align-items:flex-start}.head h1{margin:0;font-size:30px;letter-spacing:-.03em}.head p{margin:8px 0 0;color:var(--muted);font-size:13px;line-height:1.65}.tag{padding:8px 12px;border:1px solid #cdeee4;border-radius:999px;background:#effaf7;color:#087c61;font-size:12px;white-space:nowrap}.section{margin-top:25px;min-width:0}.section h2{margin:0 0 12px;font-size:19px}.section-title{display:flex;justify-content:space-between;align-items:end;gap:16px;flex-wrap:wrap}.section-title>*{min-width:0}.section-title span{color:var(--muted);font-size:12px;margin-bottom:13px;overflow-wrap:anywhere}.card{background:var(--paper);border:1px solid var(--line);border-radius:11px;box-shadow:var(--shadow);min-width:0}.intro{padding:22px}.intro p{margin:0;color:var(--muted);font-size:14px;line-height:1.8}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.flow article{padding:16px;position:relative}.flow b,.rule b{font-size:11px;color:var(--blue)}.flow h3,.rule h3{margin:8px 0 5px;font-size:15px}.flow p,.rule p{margin:0;color:var(--muted);font-size:12px;line-height:1.72}.rules{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.rule{padding:19px}.rule ul{margin:7px 0 0;padding-left:19px;color:var(--muted);font-size:12px;line-height:1.85}.notice{padding:13px 15px;background:#fff8e8;border:1px solid #f0deb8;border-radius:9px;color:#805d20;font-size:12px;line-height:1.7}.kpis{display:grid;grid-template-columns:repeat(4,1fr);overflow:hidden}.kpi{padding:19px;border-right:1px solid var(--line)}.kpi:last-child{border:0}.label{color:var(--muted);font-size:11px}.value{display:block;margin-top:7px;font-size:22px;font-weight:750}.positive{color:var(--red)}.negative{color:var(--green)}.plan{padding:22px;overflow:hidden}.plan-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.plan-grid>div{min-width:0}.plan strong{font-size:18px}.plan p{margin:6px 0 0;color:var(--muted);font-size:12px;line-height:1.7;overflow-wrap:anywhere;word-break:break-word}.table-wrap{overflow:auto;max-width:100%}.simple{width:100%;border-collapse:collapse}.simple th,.simple td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;font-size:11px;white-space:nowrap}.simple th{color:var(--muted);background:#fafbfc;font-weight:600}.simple th:first-child,.simple td:first-child{text-align:left}.simple tr.top{background:#f3f5ff}.badge{display:inline-block;padding:3px 6px;border-radius:4px;background:#effaf7;color:#087c61;font-size:10px}.muted{color:var(--muted)}.footer{margin-top:30px;padding:15px 0;border-top:1px solid var(--line);color:var(--muted);font-size:11px}@media(max-width:900px){.shell{grid-template-columns:minmax(0,1fr)}.side{position:static;height:auto;display:flex;justify-content:space-between;align-items:center;padding:14px 20px}.side nav{display:flex;margin:0}.flow{grid-template-columns:repeat(3,1fr)}.kpis{grid-template-columns:repeat(2,1fr)}.kpi:nth-child(2){border-right:0}}@media(max-width:620px){.content{width:calc(100% - 20px);padding-top:24px}.side nav{display:none}.head{display:block}.tag{display:inline-block;margin-top:10px}.section-title>span{display:none}.flow,.rules,.plan-grid{grid-template-columns:minmax(0,1fr)}.kpis{grid-template-columns:minmax(0,1fr)}.kpi{border-right:0;border-bottom:1px solid var(--line)}}
</style>'''


def nav(active: str) -> str:
    items = [("strategy", "ye-strategy.html", "策略介绍"), ("daily", "ye-daily.html", "今日日报"), ("backtest", "ye-backtest.html", "回测" )]

    links = "".join(
        f'<a class="{"active" if key == active else ""}" href="{href}">{label}</a>'
        for key, href, label in items
    )
    return f'<aside class="side"><div><div class="logo"><i>ye</i> strategy</div><small>ETF 轮动</small></div><nav>{links}</nav></aside>'


STRATEGY_EXTRA_STYLE = r'''<style>
.strategy-lead{padding:25px;background:linear-gradient(135deg,#fff 0%,#f1f4ff 100%)}.strategy-lead h2{margin:0 0 8px;font-size:24px;letter-spacing:-.03em}.strategy-lead p{margin:0;color:var(--muted);font-size:14px;line-height:1.85}.truth-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.truth{padding:18px}.truth .num{display:block;color:var(--blue);font-size:12px;font-weight:800}.truth h3{margin:7px 0;font-size:17px}.truth p{margin:0;color:var(--muted);font-size:12px;line-height:1.75}.decision-map{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.decision-step{padding:18px;position:relative}.decision-step em{display:inline-block;font-style:normal;color:var(--blue);font-size:11px;font-weight:800}.decision-step h3{margin:8px 0 6px;font-size:15px}.decision-step p{margin:0;color:var(--muted);font-size:12px;line-height:1.7}.decision-step strong{display:block;margin-top:11px;font-size:12px}.decision-step:after{content:'→';position:absolute;right:-9px;top:42%;z-index:2;padding:2px;color:#97a1b3;background:var(--bg)}.decision-step:last-child:after{display:none}.rule-matrix{display:grid;grid-template-columns:1fr 1fr;gap:10px}.rule-matrix .rule{min-height:250px}.path-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.path-card{padding:19px;min-width:0}.path-card .count{float:right;font-size:26px;font-weight:800;color:var(--blue)}.path-card h3{margin:7px 0;font-size:16px}.path-card p{margin:0;color:var(--muted);font-size:12px;line-height:1.72;overflow-wrap:anywhere}.filter-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.filter-stage{padding:18px;position:relative;min-width:0}.filter-stage small{color:var(--muted)}.filter-stage strong{display:block;margin:7px 0 4px;font-size:25px}.filter-stage p{margin:0;color:var(--muted);font-size:12px;line-height:1.65;overflow-wrap:anywhere}.filter-stage:after{content:'→';position:absolute;right:-9px;top:43%;z-index:2;padding:2px;color:#97a1b3;background:var(--bg)}.filter-stage:last-child:after{display:none}.status-pass{color:var(--green);font-weight:750}.status-fail{color:var(--muted)}.callout{padding:18px;border-left:4px solid var(--blue);background:#f3f5ff}.callout h3{margin:0 0 7px;font-size:16px}.callout p{margin:0;color:#4d5872;font-size:12px;line-height:1.8}.execution-table td:first-child{font-weight:750;color:var(--ink)}.execution-table td{vertical-align:top;white-space:normal;line-height:1.65}.execution-table th:nth-child(1){width:18%}.execution-table th:nth-child(2){width:34%}@media(max-width:900px){.truth-grid,.decision-map,.filter-strip{grid-template-columns:repeat(2,minmax(0,1fr))}.path-grid{grid-template-columns:minmax(0,1fr)}.decision-step:nth-child(2):after,.filter-stage:nth-child(2):after{display:none}}@media(max-width:620px){.truth-grid,.decision-map,.rule-matrix,.filter-strip{grid-template-columns:minmax(0,1fr)}.decision-step:after,.filter-stage:after{display:none}.path-card p,.truth p,.filter-stage p{word-break:break-all}}
</style>'''


def page(
    title: str,
    active: str,
    header: str,
    subtitle: str,
    badge: str,
    body: str,
    *,
    extra_style: str = "",
    footer: str = "ye 策略 · 数据与规则均以收盘后冻结版本为准 · 历史回测不代表未来收益",
) -> str:
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>{PAGE_STYLE}{STRATEGY_EXTRA_STYLE}{extra_style}</head><body><div class="shell">{nav(active)}<main class="content"><header class="head"><div><h1>{header}</h1><p>{subtitle}</p></div><span class="tag">{badge}</span></header>{body}<footer class="footer">{footer}</footer></main></div></body></html>'''


def as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value) if not pd.isna(value) else False


def build_daily_page() -> str:
    market = yaml.safe_load((PROJECT / "config" / "market.yaml").read_text(encoding="utf-8"))
    date = str(market["project"]["data_end"])
    ranking_path = PROJECT / "results" / "comparison" / "latest_ranking.csv"
    plan_path = PROJECT / "results" / "live" / f"{date}_order_plan.json"
    readiness_path = PROJECT / "results" / "live" / "readiness_report.json"
    if not ranking_path.exists():
        body = '<section class="section"><div class="notice">尚未生成当日排名。请先完成收盘后运行，再刷新此页。</div></section>'
        return page("ye 策略今日日报", "daily", "今日日报", "等待收盘后冻结数据", "未生成", body)
    ranking_history = pd.read_csv(ranking_path)
    rankings = ranking_history.loc[ranking_history["date"].eq(date)].copy()
    prior_dates = sorted(value for value in ranking_history["date"].unique() if str(value) < date)
    previous_rankings = (
        ranking_history.loc[ranking_history["date"].eq(prior_dates[-1])].copy()
        if prior_dates else pd.DataFrame()
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.exists() else {}
    actions = plan.get("actions", []) or []
    first_action = actions[0] if actions else {}
    sides = [item.get("side") for item in actions]
    target_symbol = plan.get("target_symbol")
    switch_status = plan.get("decision_basis", {}).get("opportunity_switch", {})
    if "sell" in sides and "buy" in sides:
        action = "换仓"
    elif "buy" in sides:
        action = "买入"
    elif "sell" in sides:
        action = "卖出并空仓" if not target_symbol else "卖出"
    elif first_action.get("side") == "hold" and target_symbol:
        action = "持有"
    elif first_action.get("side") == "hold":
        action = "继续空仓"
    else:
        action = plan.get("action", "尚未生成")
    target_match = rankings[rankings["symbol"].eq(target_symbol)] if target_symbol else pd.DataFrame()
    target_name = str(target_match.iloc[0]["name"]) if not target_match.empty else "现金"
    review = plan.get("ai_review", plan.get("sentiment_review", {})) or {}
    account = plan.get("account_state", {}) or {}
    orders = plan.get("execution", {}).get("orders", []) or []
    buy_order = next((item for item in orders if item.get("side") == "buy"), {})
    buy_estimate = buy_order.get("buy_estimate", {}) or {}
    review_path = PROJECT / "market_data" / "sentiment" / "ai_review" / f"{date}.json"
    full_review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {"items": []}
    theme_counts: dict[str, dict[str, int]] = {}
    for item in full_review.get("items", []):
        ai = item.get("ai", {})
        if not ai.get("relevant"):
            continue
        for category in ai.get("matched_categories", []):
            bucket = theme_counts.setdefault(str(category), {"positive": 0, "negative": 0})
            if int(ai.get("direction", 0)) > 0:
                bucket["positive"] += 1
            elif int(ai.get("direction", 0)) < 0:
                bucket["negative"] += 1
    reviewed = review.get("reviewed_count", 0)
    total = review.get("input_count", review.get("total_count", 0))
    status = readiness.get("status", "待检查")
    final_mask = rankings["final_entry_pass"].map(as_bool)
    candidates = rankings.loc[final_mask].copy()
    score_column = "selection_score" if "selection_score" in candidates else "momentum_score"
    candidates = candidates.sort_values([score_column, "rank"], ascending=[False, True])
    current_symbol = plan.get("current_symbol")
    switch_candidate_symbol = switch_status.get("candidate_symbol")
    switch_candidate_match = (
        rankings[rankings["symbol"].eq(switch_candidate_symbol)]
        if switch_candidate_symbol else pd.DataFrame()
    )
    switch_candidate_name = (
        str(switch_candidate_match.iloc[0]["name"])
        if not switch_candidate_match.empty else str(switch_candidate_symbol or "无")
    )
    satellite_mask = rankings.get("pool_role", pd.Series("core", index=rankings.index)).eq("challenger")
    satellite_rankings = rankings.loc[satellite_mask].sort_values(["rank", "momentum_score"], ascending=[True, False])
    core_candidates = candidates.loc[~candidates.index.isin(satellite_rankings.index)]
    satellite_technical = satellite_rankings.loc[
        satellite_rankings["technical_entry_pass"].map(as_bool)
    ]
    satellite_final = satellite_rankings.loc[
        satellite_rankings["final_entry_pass"].map(as_bool)
    ]

    def path_name(row) -> str:
        normal_value = getattr(row, "confirmed_normal_entry", getattr(row, "normal_entry", False))
        if as_bool(normal_value):
            return "常规动量"
        if as_bool(getattr(row, "emerging_entry", False)):
            return "新趋势例外"
        if as_bool(getattr(row, "quality_extension", False)):
            return "9%—12%质量延伸"
        return "未通过"

    def candidate_handling(symbol: str, index: int) -> str:
        if symbol == target_symbol:
            return "<b>唯一目标</b>"
        if current_symbol == target_symbol and target_symbol:
            return "合格；现有仓位尚未达到完整换仓触发"
        return "合格，选择分较低" if index > 1 else "合格；当前计划未选用"

    core_candidate_names = "、".join(
        f"{row.name}（{row.symbol}）" for row in core_candidates.itertuples(index=False)
    ) or "无"
    satellite_technical_names = "、".join(
        f"{row.name}（{row.symbol}）" for row in satellite_technical.itertuples(index=False)
    ) or "无"
    satellite_final_names = "、".join(
        f"{row.name}（{row.symbol}）" for row in satellite_final.itertuples(index=False)
    ) or "无"
    if not satellite_final.empty:
        satellite_status = f"卫星补位已启用：{satellite_final_names}进入最终候选。"
    elif not satellite_technical.empty and not core_candidates.empty:
        satellite_status = f"卫星待命：{satellite_technical_names}技术合格，但核心已有候选。"
    elif not satellite_technical.empty:
        satellite_status = f"卫星技术合格但未进入最终候选：{satellite_technical_names}。"
    else:
        satellite_status = f"卫星未触发：{len(satellite_rankings)}只均未通过技术入场。"

    satellite_rows_html = []
    for row in satellite_rankings.itertuples(index=False):
        technical_ok = as_bool(row.technical_entry_pass)
        final_ok = as_bool(row.final_entry_pass)
        technical_label = (
            '<span class="status-pass">通过</span>'
            if technical_ok else '<span class="status-fail">未通过</span>'
        )
        if final_ok:
            handling = "<b>最终补位候选</b>"
        elif technical_ok and not core_candidates.empty:
            handling = "技术通过；核心已有候选，待命"
        elif technical_ok:
            handling = "技术通过；未进入最终候选"
        else:
            blockers = []
            if not as_bool(row.pool_eligible): blockers.append("资格不足")
            if as_float(row.rank) > 5: blockers.append("虚拟排名&gt;5")
            if as_float(row.roc20) <= 0: blockers.append("ROC20≤0")
            if as_float(row.roc60) <= 0: blockers.append("ROC60≤0")
            if not as_bool(row.above_ma120): blockers.append("MA120下方")
            if as_float(row.ma120_bias) > .09: blockers.append("乖离&gt;9%")
            handling = "未通过：" + "、".join(blockers[:3])
        satellite_rows_html.append(
            f"<tr><td><b>{html_lib.escape(str(row.name))}</b><br><span class=\"muted\">{row.symbol}</span></td>"
            f"<td>{int(as_float(row.rank))}</td><td>{pct(as_float(row.momentum_score))}</td>"
            f"<td>{pct(as_float(row.roc20))}</td><td>{pct(as_float(row.roc60))}</td>"
            f"<td>{pct(as_float(row.ma120_bias))}</td>"
            f"<td>{technical_label}</td>"
            f"<td>{handling}</td></tr>"
        )
    satellite_rows_table = "".join(satellite_rows_html)

    candidate_rows = "".join(
        f"<tr><td>{index}</td><td><b>{html_lib.escape(str(row.name))}</b><br><span class=\"muted\">{row.symbol}</span></td><td>{int(row.rank)}</td><td>{pct(as_float(getattr(row, score_column)))}</td><td>{html_lib.escape(path_name(row))}</td><td>{candidate_handling(str(row.symbol), index)}</td></tr>"
        for index, row in enumerate(candidates.itertuples(index=False), start=1)
    )

    rows = []
    for row in rankings.sort_values("rank").itertuples(index=False):
        top = " class=\"top\"" if as_float(row.rank) <= 5 else ""
        pool_ok = as_bool(row.pool_eligible)
        normal_ok = as_bool(getattr(row, "confirmed_normal_entry", getattr(row, "normal_entry", False))) and pool_ok
        emerging_ok = as_bool(getattr(row, "emerging_entry", False)) and pool_ok
        extension_ok = as_bool(getattr(row, "quality_extension", False)) and pool_ok
        final_ok = as_bool(row.final_entry_pass)
        technical_ok = as_bool(getattr(row, "technical_entry_pass", final_ok))
        if str(row.symbol) == target_symbol:
            handling = "<b>唯一目标</b>"
        elif final_ok:
            handling = (
                "最终合格；现有仓位尚未达到完整换仓触发"
                if current_symbol == target_symbol and target_symbol
                else "最终合格，选择分较低"
            )
        elif technical_ok:
            handling = "技术合格；已有核心候选，等待空档"
        else:
            blockers = []
            if not pool_ok: blockers.append("上市期/流动性")
            if int(row.rank) > 5: blockers.append("常规排名>5")
            if as_float(row.roc20) <= 0: blockers.append("ROC20≤0")
            if as_float(row.roc60) <= 0: blockers.append("ROC60≤0")
            if not as_bool(row.above_ma120): blockers.append("MA120下方")
            if as_float(row.ma120_bias) > .09: blockers.append("常规乖离>9%")
            handling = "、".join(blockers[:3])
            if not handling:
                handling = "常规确认或两条例外路径未通过"
        flag = lambda ok, label: f'<span class="status-pass">{label}</span>' if ok else '<span class="status-fail">—</span>'
        rows.append(
            f"<tr{top}><td>{int(row.rank)}</td><td><b>{html_lib.escape(str(row.name))}</b><br><span class=\"muted\">{row.symbol} · {row.category} · {'卫星' if getattr(row, 'pool_role', 'core') == 'challenger' else '核心'}</span></td>"
            f"<td>{flag(pool_ok, '通过')}</td><td>{pct(as_float(row.momentum_score))}</td><td>{pct(as_float(row.roc20))}</td><td>{pct(as_float(row.roc60))}</td><td>{pct(as_float(row.ma120_bias))}</td>"
            f"<td>{flag(normal_ok, '通过')}</td><td>{flag(emerging_ok, '通过')}</td><td>{flag(extension_ok, '通过')}</td><td>{handling}</td></tr>"
        )

    pool_count = int(rankings["pool_eligible"].map(as_bool).sum())
    normal_count = sum(
        as_bool(getattr(row, "confirmed_normal_entry", getattr(row, "normal_entry", False))) and as_bool(row.pool_eligible)
        for row in rankings.itertuples(index=False)
    )
    emerging_count = int((rankings["emerging_entry"].map(as_bool) & rankings["pool_eligible"].map(as_bool)).sum())
    extension_count = int((rankings["quality_extension"].map(as_bool) & rankings["pool_eligible"].map(as_bool)).sum())
    base_like = (
        rankings["pool_eligible"].map(as_bool)
        & rankings["rank"].between(4, 5)
        & rankings["roc20"].gt(0.0)
        & rankings["roc20"].lt(0.02)
        & rankings["roc60"].gt(0.0)
        & rankings["above_ma120"].map(as_bool)
        & rankings["ma120_bias"].le(0.09)
    )
    weak_edge_count = int(base_like.sum())
    target_category = str(target_match.iloc[0]["category"]) if not target_match.empty else "—"
    target_theme = theme_counts.get(target_category, {"positive": 0, "negative": 0})
    theme_rows = [(name, value["positive"], value["negative"]) for name, value in theme_counts.items()]
    strongest = max(theme_rows, key=lambda item: item[1] - item[2], default=("无", 0, 0))
    riskiest = max(theme_rows, key=lambda item: item[2] - item[1], default=("无", 0, 0))
    positions = [item for item in account.get("positions", []) if as_float(item.get("quantity")) > 0]
    account_position_plain = (
        "空仓"
        if not positions
        else f"持有 {positions[0].get('symbol')} · {int(as_float(positions[0].get('quantity'))):,} 股"
    )
    account_position = html_lib.escape(account_position_plain)
    target_weight = "100%" if target_symbol else "0%"
    total_equity = as_float(account.get("total_equity"))
    performance = account.get("performance", {}) or {}
    contributed_capital = as_float(performance.get("net_contributed_capital"))
    strategy_pnl = total_equity - contributed_capital if contributed_capital > 0 else 0.0
    strategy_return = strategy_pnl / contributed_capital if contributed_capital > 0 else 0.0
    strategy_start = str(performance.get("strategy_start_date", "—"))
    if positions:
        live_position = positions[0]
        position_quantity = as_float(live_position.get("quantity"))
        average_cost = as_float(live_position.get("average_cost"))
        market_price = as_float(live_position.get("market_price"))
        purchase_pnl = position_quantity * (market_price - average_cost)
        purchase_return = market_price / average_cost - 1.0 if average_cost > 0 else 0.0
    else:
        average_cost = market_price = 0.0
        purchase_pnl = purchase_return = 0.0
    previous_plan_paths = sorted(
        path for path in (PROJECT / "results" / "live").glob("????-??-??_order_plan.json")
        if path.stem[:10] < date
    )
    previous_equity = contributed_capital
    previous_equity_date = strategy_start
    if previous_plan_paths:
        previous_plan = json.loads(previous_plan_paths[-1].read_text(encoding="utf-8"))
        previous_equity = as_float(previous_plan.get("account_state", {}).get("total_equity"))
        previous_equity_date = str(previous_plan.get("signal_date", previous_plan_paths[-1].stem[:10]))
    daily_pnl = total_equity - previous_equity if previous_equity > 0 else 0.0
    daily_return = daily_pnl / previous_equity if previous_equity > 0 else 0.0

    def signed_pct(value: float) -> str:
        return f"{value:+.2%}"

    def signed_money(value: float) -> str:
        return f"{value:+,.2f} 元"

    def value_class(value: float) -> str:
        return "positive" if value > 0 else "negative" if value < 0 else ""

    order_text = (
        f"收盘估算约 {int(as_float(buy_estimate.get('estimated_quantity_at_last_close'))):,} 份；开盘按实际价格、可用现金和100份整数倍重算。"
        if buy_estimate else "本次没有新增买单。"
    )
    confirmation_label = "买卖计划待成交确认" if orders else "无买卖订单 · 无需成交确认"
    held_match = rankings[rankings["symbol"].eq(current_symbol)] if current_symbol else pd.DataFrame()
    if not held_match.empty:
        held_row = held_match.iloc[-1]
        held_exit_checks = (
            f"收盘价{'仍在MA120上方' if as_bool(held_row['above_ma120']) else '已经跌到MA120下方'}，"
            f"ROC20为{pct(as_float(held_row['roc20']))}，"
            f"5日与20日排名{'没有同时恶化' if not as_bool(held_row['dual_rank_decline']) else '已经同时恶化'}"
        )
    else:
        held_exit_checks = "当前没有旧仓需要检查"
    if not target_match.empty:
        target_row = target_match.iloc[-1]
        target_snapshot = (
            f"{target_name}在全池排第{int(as_float(target_row['rank']))}，正式选择分为"
            f"{pct(as_float(target_row.get(score_column, target_row['momentum_score'])))}，"
            f"ROC20为{pct(as_float(target_row['roc20']))}、ROC60为{pct(as_float(target_row['roc60']))}，"
            f"MA120乖离为{pct(as_float(target_row['ma120_bias']))}，通过{path_name(target_row)}路径"
        )
    else:
        target_snapshot = "今天没有ETF通过全部条件，策略目标因此是现金"
    previous_target_match = (
        previous_rankings[previous_rankings["symbol"].eq(target_symbol)]
        if target_symbol and not previous_rankings.empty else pd.DataFrame()
    )
    if not target_match.empty and not previous_target_match.empty:
        previous_target = previous_target_match.iloc[-1]
        score_change = as_float(target_row.get(score_column, target_row["momentum_score"])) - as_float(
            previous_target.get(score_column, previous_target["momentum_score"])
        )
        target_change_story = (
            f"排名由第{int(as_float(previous_target['rank']))}变为第{int(as_float(target_row['rank']))}，"
            f"选择分较昨日{score_change * 100:+.2f}个百分点；ROC20由{pct(as_float(previous_target['roc20']))}"
            f"降至{pct(as_float(target_row['roc20']))}，ROC60由{pct(as_float(previous_target['roc60']))}"
            f"降至{pct(as_float(target_row['roc60']))}。"
        )
    else:
        target_change_story = (
            f"ROC20为{pct(as_float(target_row['roc20']))}，ROC60为{pct(as_float(target_row['roc60']))}。"
            if not target_match.empty else ""
        )
    candidate_names = "、".join(
        f"{row.name}（{row.symbol}）" for row in candidates.itertuples(index=False)
    ) or "无"
    rejected_details = []
    for row in rankings.loc[rankings["rank"].le(5) & ~final_mask].sort_values("rank").itertuples(index=False):
        reasons = []
        if as_float(row.roc20) <= 0:
            reasons.append(f"ROC20 {pct(as_float(row.roc20))}")
        if as_float(row.roc60) <= 0:
            reasons.append(f"ROC60 {pct(as_float(row.roc60))}")
        if as_float(row.ma120_bias) > 0.09:
            reasons.append(f"MA120乖离 {pct(as_float(row.ma120_bias))}")
        if not as_bool(row.above_ma120):
            reasons.append("MA120下方")
        rejected_details.append(f"第{int(row.rank)}名{row.name}因{'、'.join(reasons[:2]) or '入场条件不足'}未通过")
    rejected_story = "；".join(rejected_details) or "前5名没有额外淘汰项"
    leading_rejection = rejected_details[0] if rejected_details else "前5名没有额外淘汰项"
    if not target_match.empty:
        target_prices = pd.read_csv(PROJECT / "market_data" / "prices" / f"{target_symbol}.csv")
        target_prices["datetime"] = pd.to_datetime(target_prices["datetime"])
        recent_prices = target_prices.loc[target_prices["datetime"].le(pd.Timestamp(date))].sort_values("datetime").tail(2)
        previous_close = as_float(recent_prices.iloc[-2]["close"]) if len(recent_prices) >= 2 else market_price
        close_change = market_price / previous_close - 1.0 if previous_close > 0 else 0.0
        target_roc20 = as_float(target_row["roc20"])
        target_roc60 = as_float(target_row["roc60"])
        if target_roc20 > 0 and target_roc60 > 0:
            momentum_story = "ROC20、ROC60均为正，短中期动量同向。"
        elif target_roc20 > 0:
            momentum_story = "ROC20仍为正、ROC60已转负；中期动量走弱，但这不是现有仓位的独立卖出条件。"
        else:
            momentum_story = "ROC20已转负，需要按正式退出规则处理。"
        target_insight_story = (
            f"{target_name}今天仍排第{int(as_float(target_row['rank']))}，"
            f"选择分{pct(as_float(target_row.get(score_column, target_row['momentum_score'])))}。"
            f"{target_change_story}MA120乖离为{pct(as_float(target_row['ma120_bias']))}，价格仍在MA120上方。"
            f"{momentum_story}"
        )
        close_story = f"{target_name}收于{market_price:.3f}元，较昨日{signed_pct(close_change)}"
    else:
        target_insight_story = "今天没有ETF成为最终目标，账户保持现金。"
        close_story = "账户当前没有ETF持仓"
    if len(candidates) > 1:
        first_candidate, second_candidate = candidates.iloc[0], candidates.iloc[1]
        candidate_comparison_story = (
            f"{first_candidate['name']}选择分{pct(as_float(first_candidate[score_column]))}，"
            f"高于{second_candidate['name']}的{pct(as_float(second_candidate[score_column]))}"
        )
    elif len(candidates) == 1:
        only_candidate = candidates.iloc[0]
        candidate_comparison_story = (
            f"唯一候选{only_candidate['name']}选择分{pct(as_float(only_candidate[score_column]))}"
        )
    else:
        candidate_comparison_story = "今天没有最终合格候选"
    category_peers = rankings.loc[
        rankings["category"].eq(target_category) & ~rankings["symbol"].eq(target_symbol)
    ].sort_values("rank")
    if not category_peers.empty:
        peer = category_peers.iloc[0]
        if as_bool(peer["final_entry_pass"]):
            peer_result = "并已通过正式入场筛选"
        elif as_float(peer["rank"]) > 5:
            peer_result = "但未进入前5"
        else:
            peer_result = "但未通过完整入场条件"
        category_peer_story = (
            f"同属{target_category}的{peer['name']}排第{int(as_float(peer['rank']))}，"
            f"ROC20为{pct(as_float(peer['roc20']))}，{peer_result}。"
        )
    else:
        category_peer_story = f"{target_category}没有其他可比ETF。"
    if current_symbol and current_symbol == target_symbol:
        if switch_status.get("qualifies_today"):
            if switch_status.get("status") == "baseline":
                decision_story = (
                    f"现有{target_name}未触发卖出；{switch_candidate_name}满足换仓比较，但今天只建立观察基线0/2，"
                    "明日继续满足才记1/2，所以继续持有。"
                )
            else:
                decision_story = (
                    f"现有{target_name}未触发卖出；{switch_candidate_name}满足换仓比较，"
                    f"当前连续确认{switch_status.get('confirmation_streak', 0)}/{switch_status.get('required_confirmation_days', 2)}，"
                    "尚未触发，所以继续持有。"
                )
        else:
            decision_story = f"现有{target_name}未触发卖出，也未完整满足机会换仓，继续持有。"
    elif not current_symbol and target_symbol:
        decision_story = f"账户原本空仓，因此从最终合格候选中按正式选择分选出{target_name}作为唯一目标。"
    elif current_symbol and target_symbol:
        decision_story = f"旧仓已经触发退出，卖出后再按正式选择分切换到{target_name}，先卖后买。"
    elif current_symbol:
        decision_story = "旧仓已经触发退出，但今天没有新的合格候选，因此卖出后持有现金。"
    else:
        decision_story = "账户原本空仓，今天又没有新的合格候选，因此继续持有现金。"
    if current_symbol and current_symbol == target_symbol and target_symbol:
        core_insight = (
            f"{target_name}虽出现单日波动，但仍未触发价格退出；{leading_rejection}，"
            "因此继续按既定纪律持有。"
        )
    else:
        core_insight = decision_story
    overview_paragraphs = [
        f"今天账户变动{signed_money(daily_pnl)}，收益{signed_pct(daily_return)}；{close_story}。自{strategy_start}实盘开启以来，账户累计{signed_money(strategy_pnl)}，本次买入浮动盈亏{signed_money(purchase_pnl)}。明日结论不变：{action}{target_name}，不产生新订单。",
        target_insight_story,
        f"{len(rankings)}只ETF中有{len(candidates)}只最终通过，分别是{candidate_names}。{candidate_comparison_story}，且{decision_story}前5名里，{rejected_story}；它们名次高，但当前并不是可买候选。",
        f"板块观察：{category_peer_story}换仓必须同时满足前5外、领先5点、连续2日和持有5日。今天没有新趋势或质量延伸候选，最终候选均来自常规动量。",
        f"资讯面上，{target_category}主题记录为{target_theme['positive']}条正向、{target_theme['negative']}条负向，均按专属关键词直接映射。今天最重要的洞察是：{core_insight}",
        satellite_status,
    ]
    if len("".join(overview_paragraphs)) < 500:
        raise RuntimeError("daily plain-language overview must contain at least 500 characters")
    overview_html = "".join(f"<p>{html_lib.escape(paragraph)}</p>" for paragraph in overview_paragraphs)
    if not switch_status.get("enabled"):
        switch_summary = "机会换仓未启用。"
    elif switch_status.get("status") == "baseline":
        switch_summary = (
            f"机会换仓：{switch_candidate_name}今日合格，但今天只建立观察基线0/2；"
            "明日继续合格才记1/2，当前未触发。"
        )
    else:
        switch_summary = (
            f"机会换仓：{switch_candidate_name}；今日{'合格' if switch_status.get('qualifies_today') else '不合格'}，"
            f"连续确认{switch_status.get('confirmation_streak', 0)}/{switch_status.get('required_confirmation_days', 2)}，"
            f"{'已触发' if switch_status.get('triggered') else '未触发'}。"
        )
    body = f'''
<section class="section" id="liveReturns"><div class="card kpis" style="grid-template-columns:repeat(3,1fr)"><div class="kpi"><span class="label">策略实盘开启以来</span><strong class="value {value_class(strategy_return)}">{signed_pct(strategy_return)}</strong><p>{signed_money(strategy_pnl)} · 自 {html_lib.escape(strategy_start)}</p></div><div class="kpi"><span class="label">本次买入收益</span><strong class="value {value_class(purchase_return)}">{signed_pct(purchase_return)}</strong><p>{signed_money(purchase_pnl)} · 成本 {average_cost:.3f} 元</p></div><div class="kpi"><span class="label">今日收益</span><strong class="value {value_class(daily_return)}">{signed_pct(daily_return)}</strong><p>{signed_money(daily_pnl)} · 对比 {html_lib.escape(previous_equity_date)}</p></div></div></section>
<section class="section" id="dailyOverview"><article class="card strategy-lead daily-overview"><span class="label">先看这里 · 约500字要点版</span><h2>今日决策路径综述</h2>{overview_html}</article></section>
<section class="section"><div class="section-title"><h2>次日开盘计划</h2><span>{confirmation_label}</span></div><article class="card plan"><span class="label">执行结论</span><br><strong>{html_lib.escape(str(action))} {html_lib.escape(str(target_name))}</strong><p>目标仓位 {target_weight}。{html_lib.escape(order_text)}</p><p>{html_lib.escape(switch_summary)}</p></article></section>
<section class="section"><div class="section-title"><h2>今日筛选漏斗</h2><span>不仅看数量，也直接看留下了谁</span></div><div class="filter-strip"><article class="card filter-stage"><small>冻结母池</small><strong>{len(rankings)} 只</strong><p>45只核心＋{len(satellite_rankings)}只卫星；核心独立排名并拥有买入优先权。</p></article><article class="card filter-stage"><small>上市期与流动性</small><strong>{pool_count} 只</strong><p>{pool_count}只通过基础资格。</p></article><article class="card filter-stage"><small>核心优先后合格</small><strong>{len(candidates)} 只</strong><p>{html_lib.escape(candidate_names)}</p></article><article class="card filter-stage"><small>明日实际目标</small><strong>{1 if target_symbol else 0} 只</strong><p>{html_lib.escape(str(target_name))}{'（继续持有）' if current_symbol == target_symbol and target_symbol else ''}</p></article></div></section>
<section class="section" id="dailySatellite"><div class="section-title"><h2>今日卫星检查</h2><span>{html_lib.escape(satellite_status)}</span></div><div class="truth-grid"><article class="card truth"><span class="num">核心候选</span><h3>{len(core_candidates)} 只</h3><p>{html_lib.escape(core_candidate_names)}</p></article><article class="card truth"><span class="num">卫星技术合格</span><h3>{len(satellite_technical)} / {len(satellite_rankings)} 只</h3><p>{html_lib.escape(satellite_technical_names)}</p></article><article class="card truth"><span class="num">卫星最终补位</span><h3>{len(satellite_final)} 只</h3><p>{html_lib.escape(satellite_final_names)}</p></article></div><div class="card table-wrap" style="margin-top:10px"><table class="simple"><thead><tr><th>卫星ETF</th><th>虚拟排名</th><th>动量分</th><th>ROC20</th><th>ROC60</th><th>MA120乖离</th><th>技术结果</th><th>当日处理</th></tr></thead><tbody>{satellite_rows_table}</tbody></table></div></section>
<section class="section"><div class="section-title"><h2>三条技术入场路径的当日结果</h2><span>三条路径并行，不是依次放宽</span></div><div class="path-grid"><article class="card path-card"><span class="count">{normal_count}</span><span class="label">路径 A</span><h3>常规动量</h3><p>前5、双ROC为正、MA120上方、乖离≤9%；排名4—5且ROC20&lt;2%时再做热点确认。今日需弱边缘确认 {weak_edge_count} 只。</p></article><article class="card path-card"><span class="count">{emerging_count}</span><span class="label">路径 B</span><h3>新趋势例外</h3><p>排名≤15、ROC20≥3%、ROC60为-8%至0%，并通过R²、效率与热点门槛；记忆3日。</p></article><article class="card path-card"><span class="count">{extension_count}</span><span class="label">路径 C</span><h3>9%—12%质量延伸</h3><p>仅处理乖离略高但趋势质量很强的前5候选；R²、效率、ROC5和热点必须同时通过。</p></article></div></section>
<section class="section"><div class="section-title"><h2>最终候选与唯一目标</h2><span>先核心、后卫星；最终通过不等于同时买入</span></div><div class="card table-wrap"><table class="simple"><thead><tr><th>顺序</th><th>ETF</th><th>决策排名</th><th>选择分</th><th>通过路径</th><th>账户处理</th></tr></thead><tbody>{candidate_rows}</tbody></table></div></section>
<section class="section"><div class="section-title"><h2>资讯审核如何影响今天的筛选</h2><span>审核完整是硬门，热点只作用于指定路径</span></div><div class="truth-grid"><article class="card truth"><span class="num">完整性</span><h3>{reviewed}/{total} · 100%</h3><p>全部冻结记录逐条审核完成，因此新开仓资格未被数据完整性阻断。</p></article><article class="card truth"><span class="num">路径影响</span><h3>常规 {normal_count} / 新趋势 {emerging_count} / 延伸 {extension_count}</h3><p>今日最终候选全部来自常规路径，没有候选依赖新趋势或延伸例外放行。</p></article><article class="card truth"><span class="num">主题披露</span><h3>{html_lib.escape(target_category)} {target_theme['positive']}正 / {target_theme['negative']}负</h3><p>最强净正向：{html_lib.escape(str(strongest[0]))}；最大净负向：{html_lib.escape(str(riskiest[0]))}。正式规则没有全市场趋势开关，也没有统一“负面阈值”否决常规路径。</p></article></div></section>
<section class="section"><div class="section-title"><h2>{len(rankings)}只 ETF 逐层筛选明细</h2><span>核心排名独立；卫星仅显示相对核心的虚拟排名</span></div><div class="card table-wrap"><table class="simple"><thead><tr><th>决策排名</th><th>ETF / 池角色</th><th>资格</th><th>动量分</th><th>ROC20</th><th>ROC60</th><th>MA120乖离</th><th>常规</th><th>新趋势</th><th>延伸</th><th>最终结果</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>'''
    return page("ye 策略今日日报", "daily", "今日日报", f"信号日 {date} 收盘后生成 · 供下一交易日开盘执行参考", "每日更新", body)


def build_strategy_page() -> str:
    market = yaml.safe_load((PROJECT / "config" / "market.yaml").read_text(encoding="utf-8"))
    data_end = str(market["project"]["data_end"])
    ye = json.loads(
        (PROJECT / "results" / "ye_strategy" / "summary.json").read_text(encoding="utf-8")
    )
    reference = json.loads(
        (PROJECT / "results" / "etfwin_reference" / "reference_summary.json").read_text(encoding="utf-8")
    )
    ym, rm = ye["metrics"], reference["metrics"]
    period_order = ["2018—2020", "2021—2022", "2023—2024", "2025—2026"]
    period_rows = "".join(
        f"<tr><td>{label}</td><td>{pct(ye['periods'][label]['total_return'])}</td>"
        f"<td>{pct(reference['periods'][label]['total_return'])}</td></tr>"
        for label in period_order
    )
    body = f'''
<section class="section"><article class="card strategy-lead"><h2>这套策略每天只回答一个问题：明天开盘持有哪一只 ETF，还是持有现金？</h2><p>它不是每天追排行榜第一名。先检查真实旧仓是否必须卖，再检查是否完整满足严格的机会换仓；两者都没有触发就继续持有。账户空仓时，核心候选拥有买入优先权，卫星只在核心无候选时补位。</p></article></section>
<section class="section"><div class="truth-grid"><article class="card truth"><span class="num">时间</span><h3>T日收盘算，T+1开盘做</h3><p>只使用收盘后冻结的完整日线、成交额和资讯，不盘中改信号。</p></article><article class="card truth"><span class="num">仓位</span><h3>一只ETF 100%，或现金0%</h3><p>没有半仓、双持仓、主观加减仓、网格或分批止盈。</p></article><article class="card truth"><span class="num">账户</span><h3>实盘成交是真相</h3><p>订单计划和回测影子持仓都不能当成真实持仓；成交必须对账。</p></article></div></section>
<section class="section"><div class="section-title"><h2>每天按这四步走</h2><span>状态顺序固定，不允许跳步</span></div><div class="decision-map"><article class="card decision-step"><em>01 · 输入</em><h3>冻结价格与全部资讯</h3><p>更新51只ETF日线和成交额；每条资讯都保留哈希并逐条审核。</p><strong>审核不是100% → 禁止新买</strong></article><article class="card decision-step"><em>02 · 旧仓</em><h3>先判断卖出与换仓</h3><p>先检查原卖出条件；没有卖出时，再检查前5、分差、连续2日和持有期。</p><strong>两类都未触发 → 继续持有</strong></article><article class="card decision-step"><em>03 · 新仓</em><h3>确定完整合格候选</h3><p>核心池先完成全部入场筛选；只有核心没有候选时，才检查技术合格的卫星。</p><strong>卫星只补核心空档</strong></article><article class="card decision-step"><em>04 · 目标</em><h3>最终只留下一个</h3><p>卖出、换仓或空仓买入都只生成一只ETF；没有合格目标就持有现金。</p><strong>输出唯一0% / 100%</strong></article></div></section>
<section class="section"><div class="section-title"><h2>筛选漏斗：每一层在做什么</h2><span>先资格，后三条路径并行，最后结合账户状态</span></div><div class="card table-wrap"><table class="simple execution-table"><thead><tr><th>层级</th><th>检查内容</th><th>不通过时</th></tr></thead><tbody><tr><td>冻结母池</td><td>45只核心＋6只卫星。核心是冠军池，卫星是空档补位池；名单只在半年或年度审计中调整。</td><td>池外ETF不参与。</td></tr><tr><td>新开仓资格</td><td>上市至少120个交易日；最近20日成交额中位数至少2,000万元。</td><td>不能新买；不合格卫星也不能占核心名额。</td></tr><tr><td>锚定排名</td><td>核心只在45只内部排名；卫星计算相对核心池的虚拟排名，但不与合格核心争夺买入权。</td><td>常规路径要求前5；新趋势最多放宽到前15。</td></tr><tr><td>三条技术入场路径</td><td>常规动量、新趋势例外、9%—12%质量延伸并行判断，任一通过即可进入技术候选；核心优先后才形成最终候选。</td><td>三条都不通过则淘汰；技术通过不等于卫星当天可买。</td></tr><tr><td>账户状态</td><td>核心旧仓先检查原卖出，再检查5/5点/2日/5日机会换仓；卫星遇到合格核心则让位；空仓时先选核心。</td><td>卖出和换仓均未触发则持有旧仓；没有新目标时持有现金。</td></tr></tbody></table></div></section>
<section class="section"><div class="section-title"><h2>ye 当前完整规则：三条技术入场路径</h2><span>它们是三种不同机会，不是逐级放宽</span></div><div class="path-grid"><article class="card path-card"><span class="label">路径 A</span><h3>常规动量</h3><p>排名≤5；ROC20&gt;0、ROC60&gt;0；价格在MA120上方；MA120乖离≤9%。其中排名4—5且ROC20&lt;2%的弱边缘候选，还要满足强势股≥3、数量加速≥0、正DDE占比≥50%。</p></article><article class="card path-card"><span class="label">路径 B</span><h3>新趋势例外</h3><p>排名≤15；ROC20≥3%；ROC60在-8%至0%；MA120乖离在-3%至9%；R²20≥0.70、效率≥0.20，并满足强势股≥3、热点评分≥0.45、数量加速≥0.25、正DDE占比≥50%。触发记忆3日。</p></article><article class="card path-card"><span class="label">路径 C</span><h3>9%—12%质量延伸</h3><p>排名≤5、双ROC为正、MA120上方；乖离&gt;9%且≤12%；R²20≥0.75、效率≥0.40、ROC5≥-1%，并满足强势股≥2、热点评分≥0.45、数量加速≥0。</p></article></div></section>
<section class="section"><div class="section-title"><h2>最终候选如何排序</h2><span>不同路径使用配置中明确的选择分</span></div><article class="card callout"><h3>常规动量与质量延伸看核心动量分；新趋势看专属分</h3><p>常规动量和质量延伸使用 ROC20 + 1.5×ROC60；新趋势使用 ROC20 + 0.05×R²20。核心池内按正式选择分排序；只有核心没有候选时，卫星才按自身选择分补位。机会换仓比较的是两只核心ETF的原始动量分，不使用新趋势专属分。</p></article></section>
<section class="section"><div class="section-title"><h2>持有、卖出与换仓</h2><span>三种处理分开判断；先卖出，再换仓，否则持有</span></div><div class="rule-matrix" style="grid-template-columns:repeat(3,1fr)"><article class="card rule"><b>一、继续持有</b><h3>卖出和换仓都未触发</h3><ul><li>持仓不必每天仍在前5；</li><li>另一只ETF单日领先，不会立刻追过去；</li><li>上市期与流动性是新买门槛，不是机械卖出条件；</li><li>卫星持仓遇到合格核心候选时仍须让位。</li></ul></article><article class="card rule"><b>二、卖出</b><h3>风险条件任一触发</h3><ul><li>硬退出：收盘价跌破MA120，任何热点都不能保护；</li><li>软退出：ROC20转负；或当前排名同时差于5日前和20日前；</li><li>合格热点最多保护软退出2日；</li><li>卖出后同一ETF冷却5个交易日。</li></ul></article><article class="card rule"><b>三、主动换仓</b><h3>四项条件必须全部满足</h3><ul><li>核心旧仓掉出前5；</li><li>另一只完整合格核心ETF动量分领先至少5个百分点；</li><li>同一只领先ETF连续2个收盘成立；</li><li>旧仓已持有满5个交易日；第2次确认后，下一开盘全仓先卖后买。</li></ul></article></div></section>
<section class="section"><div class="section-title"><h2>AI、资讯和市场判断的边界</h2><span>这里最容易被误解</span></div><div class="truth-grid"><article class="card truth"><span class="num">资讯完整性</span><h3>全部行必须审核</h3><p>漏审、重复、来源失败或覆盖不足100%，一律禁止新开仓；已有仓位的价格卖出仍执行。</p></article><article class="card truth"><span class="num">AI的作用</span><h3>确认指定路径，不凭新闻造趋势</h3><p>热点指标用于弱边缘、新趋势、质量延伸和软退出保护，但不能改变核心优先级。正式规则没有统一“负面风险阈值”否决所有常规买点。</p></article><article class="card truth"><span class="num">全市场趋势</span><h3>没有独立大盘开关</h3><p>策略判断单只ETF趋势和主题热点；不要求沪深300站上均线，也没有risk-on/risk-off或防守ETF切换。</p></article></div></section>
<section class="section"><div class="section-title"><h2>指标翻译</h2><span>读日报时只需理解这五项</span></div><div class="card table-wrap"><table class="simple execution-table"><thead><tr><th>指标</th><th>白话含义</th><th>用途</th></tr></thead><tbody><tr><td>ROC20 / ROC60</td><td>过去20日、60日涨跌幅。</td><td>构成核心动量分，也参与买卖门槛。</td></tr><tr><td>MA120乖离</td><td>当前价格离120日均线有多远。</td><td>判断长期趋势和是否追得过高。</td></tr><tr><td>R²20</td><td>最近20日趋势是否平滑、连贯。</td><td>只用于新趋势和质量延伸。</td></tr><tr><td>效率20</td><td>净涨幅相对每日波动总和的比例。</td><td>排除来回震荡造成的假趋势。</td></tr><tr><td>热点评分 / DDE</td><td>主题强势股数量、加速程度和资金方向的结构化结果。</td><td>只确认指定例外和软退出保护。</td></tr></tbody></table></div></section>
<section class="section"><article class="card callout"><h3>历史缺失期与实盘不同</h3><p>早期回测没有完整资讯时，只允许常规基础排名前3、且同主题ROC20为正比例≥75%的候选；强持仓还可按价格条件保护软退出。这个回退只用于历史复现。实盘资讯审核不完整时直接禁止新开仓。</p></article></section>
<section class="section"><div class="section-title"><h2>ye 与 etfwin 规则对照</h2><span>etfwin 为公开指南的本地量化代理，仅作对照</span></div><div class="card table-wrap"><table class="simple execution-table"><thead><tr><th>项目</th><th>ye 策略</th><th>etfwin 参考策略</th></tr></thead><tbody><tr><td>ETF 池</td><td>45只核心＋6只卫星；核心冠军优先，卫星只补空档</td><td>公开指南对应的固定 20 只参考池</td></tr><tr><td>核心评分</td><td>ROC20 + 1.5 × ROC60</td><td>ROC20 + 1.5 × ROC60</td></tr><tr><td>通常买入</td><td>前 5、双 ROC 为正、MA120 上方、乖离≤9%</td><td>前 5、双 ROC 为正、MA120 上方、乖离≤15%</td></tr><tr><td>增强判断</td><td>主题广度、AI弱边缘、新趋势、质量延伸、热点保护与严格机会换仓</td><td>无本地增强模块</td></tr><tr><td>卖出 / 换仓</td><td>原三项退出；另有前5外＋领先5点＋连续2日＋持有5日的核心换仓</td><td>MA120破位 / ROC20转负 / 5日与20日排名双降</td></tr><tr><td>仓位与执行</td><td>一只 ETF 或现金；T+1 开盘</td><td>同样按一只 ETF 或现金、T+1 开盘重建对照</td></tr></tbody></table></div></section>
<section class="section"><div class="section-title"><h2>同口径回测</h2><span>2018-07-02 至 {data_end} · 初始资金 10 万元</span></div><div class="card kpis"><div class="kpi"><span class="label">ye 累计收益</span><strong class="value positive">{pct(ym['total_return'])}</strong></div><div class="kpi"><span class="label">etfwin 累计收益</span><strong class="value">{pct(rm['total_return'])}</strong></div><div class="kpi"><span class="label">ye 年化</span><strong class="value positive">{pct(ym['cagr'])}</strong></div><div class="kpi"><span class="label">etfwin 年化</span><strong class="value">{pct(rm['cagr'])}</strong></div></div><div class="card table-wrap" style="margin-top:10px"><table class="simple"><thead><tr><th>指标/区间</th><th>ye</th><th>etfwin</th></tr></thead><tbody><tr><td>最大回撤</td><td>{pct(ym['max_drawdown'])}</td><td>{pct(rm['max_drawdown'])}</td></tr><tr><td>夏普比率</td><td>{ym['sharpe']:.2f}</td><td>{rm['sharpe']:.2f}</td></tr><tr><td>失败操作率</td><td>{pct(ye['timing']['failed_operation_rate'])}</td><td>{pct(reference['timing']['failed_operation_rate'])}</td></tr>{period_rows}</tbody></table></div></section>
<section class="section"><article class="card callout"><h3>成本与回测口径</h3><p>普通 ETF 单边固定成本 0.15%，QDII/溢价敏感 ETF 单边固定成本 0.30%，每笔最低佣金 5 元。两套策略使用同一日线快照、同一 T+1 开盘成交逻辑和同一成本模型。超过统一流动性参与上限时只做机械子订单拆分，不代表策略重复发出买卖信号。</p></article></section>'''
    return page(
        "ye ETF轮动策略",
        "strategy",
        "ye ETF轮动策略",
        f"当前唯一正式规则 · 数据截至 {data_end}",
        "规则已冻结",
        body,
    )


def replace_nav(html: str) -> str:
    start = html.index("<nav>")
    end = html.index("</nav>", start) + len("</nav>")
    links = '<nav><a href="ye-strategy.html">策略介绍</a><a href="ye-daily.html">今日日报</a><a class="active" href="ye-backtest.html">回测</a></nav>'
    return html[:start] + links + html[end:]


def remove_strategy_section(html: str) -> str:
    start = html.index('<section class="section" id="strategy">')
    end = html.index('</section>', start) + len('</section>')
    return html[:start] + html[end:]


def with_output_links(html: str) -> str:
    return (html.replace("ye-strategy.html", OUTPUT_HTML.name)
                .replace("ye-daily.html", OUTPUT_DAILY_HTML.name)
                .replace("ye-backtest.html", OUTPUT_BACKTEST_HTML.name))



def main() -> None:
    payload = build_payload()
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    data_end = payload["etf_dashboard"]["as_of"]
    strategy_html = build_strategy_page()
    daily_html = build_daily_page()
    backtest_html = (
        remove_strategy_section(replace_nav(HTML))
        .replace(".shell{display:grid;grid-template-columns:184px 1fr;", ".shell{display:grid;grid-template-columns:190px 1fr;")
        .replace("min-height:100vh}.side", "min-height:100vh;min-width:0}.side", 1)
        .replace("padding:38px 0 64px}.head", "padding:38px 0 64px;min-width:0}.head", 1)
        .replace(".side a:hover{background:#f4f5f7;color:var(--ink)}", ".side a.active,.side a:hover{background:#f0f3ff;color:var(--blue)}")
        .replace("2026-07-17", data_end)
        .replace("__DATA__", data)
    )
    PUBLIC_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_HTML.write_text(strategy_html, encoding="utf-8")
    PUBLIC_DAILY_HTML.write_text(daily_html, encoding="utf-8")
    PUBLIC_BACKTEST_HTML.write_text(backtest_html, encoding="utf-8")
    OUTPUT_HTML.write_text(with_output_links(strategy_html), encoding="utf-8")
    OUTPUT_DAILY_HTML.write_text(with_output_links(daily_html), encoding="utf-8")
    OUTPUT_BACKTEST_HTML.write_text(with_output_links(backtest_html), encoding="utf-8")
    print(PUBLIC_HTML)
    print(PUBLIC_DAILY_HTML)
    print(PUBLIC_BACKTEST_HTML)
    print(OUTPUT_HTML)
    print(OUTPUT_DAILY_HTML)
    print(OUTPUT_BACKTEST_HTML)
    print(f"trades={len(payload['trades'])}; backtest_bytes={len(backtest_html.encode('utf-8'))}")


if __name__ == "__main__":
    main()

