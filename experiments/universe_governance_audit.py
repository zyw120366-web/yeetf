"""Reproducible non-return ETF universe governance audit.

This audit does not score historical profit and cannot promote an ETF.  It only
checks whether a satellite has enough operational evidence to be considered at
a scheduled semiannual review.  Missing fee/index/tracking metadata is a hard
block rather than being guessed from price performance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.data import load_panel, symbol_key, universe_keys
from etf_rotation.execution import entry_eligibility


OUTPUT = ROOT / "results" / "research" / "universe_governance_v1"
REVIEW_WINDOW = 252
MIN_HISTORY = 252
MIN_ELIGIBILITY_RATE = 0.90
MAX_MISSING_RATE = 0.01


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    market = load_yaml(ROOT / "config" / "market.yaml")
    formal = load_yaml(ROOT / "config" / "ye_strategy.yaml")
    panel = load_panel(market, ROOT / "market_data" / "prices")
    symbols = universe_keys(market)
    challengers = [str(s) for s in formal["enhanced_selection"]["universe_architecture"]["challenger_symbols"]]
    items = {symbol_key(item): item for item in market["universe"]}
    eligibility, listed_sessions, amount20 = entry_eligibility(panel, symbols, formal["rules"])

    rows = []
    for symbol in challengers:
        item = items[symbol]
        close = panel["close"][symbol]
        history = int(listed_sessions[symbol].iloc[-1])
        recent = close.tail(REVIEW_WINDOW)
        eligibility_rate = float(eligibility[symbol].tail(REVIEW_WINDOW).mean())
        missing_rate = float(recent.isna().mean())
        current_amount20 = float(amount20[symbol].iloc[-1])
        observable_ready = bool(
            history >= MIN_HISTORY
            and eligibility_rate >= MIN_ELIGIBILITY_RATE
            and missing_rate <= MAX_MISSING_RATE
            and current_amount20 >= float(formal["rules"]["minimum_entry_amount"])
        )
        required_metadata = ("benchmark_index", "expense_ratio", "tracking_error")
        missing_metadata = [key for key in required_metadata if item.get(key) in (None, "")]
        metadata_complete = not missing_metadata
        promotion_ready = observable_ready and metadata_complete
        rows.append({
            "symbol": symbol,
            "name": item["name"],
            "category": item["category"],
            "first_price_date": str(close.first_valid_index().date()),
            "listed_sessions": history,
            "eligibility_rate_last252": eligibility_rate,
            "missing_rate_last252": missing_rate,
            "trailing_amount20": current_amount20,
            "observable_ready": observable_ready,
            "metadata_complete": metadata_complete,
            "missing_metadata": "|".join(missing_metadata),
            "promotion_ready": promotion_ready,
            "governance_action": "eligible_for_metadata_review" if observable_ready else "remain_in_shadow_observation",
        })

    frame = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "candidates.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "governance_research_only",
        "as_of": str(panel["close"].index[-1].date()),
        "formal_strategy_unchanged": True,
        "return_or_backtest_used_for_promotion": False,
        "review_cadence": "semiannual",
        "thresholds": {
            "minimum_history_sessions": MIN_HISTORY,
            "eligibility_rate_last252": MIN_ELIGIBILITY_RATE,
            "maximum_missing_rate_last252": MAX_MISSING_RATE,
            "minimum_trailing_amount20": float(formal["rules"]["minimum_entry_amount"]),
        },
        "required_metadata": ["benchmark_index", "expense_ratio", "tracking_error"],
        "promotion_rule": "one-for-one replacement of the same exposure only; never use historical return contribution",
        "observable_ready_symbols": frame.loc[frame["observable_ready"], "symbol"].tolist(),
        "promotion_ready_symbols": frame.loc[frame["promotion_ready"], "symbol"].tolist(),
        "daily_execution_impact": "none",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
