"""Shared, read-only ranking projection; no trading or account writes."""
import pandas as pd


def ranking_for_day(panel, symbols, names, categories, ye_config, ye_bundle,
                    ye_features, ye_eligibility, listed_sessions, trailing_amount,
                    decision, sentiment, latest):
    rank = decision["entry_rank"]
    base_pass = (
        ye_eligibility
        & rank.le(int(ye_config["rules"]["entry_rank_limit"]))
        & ye_features.roc_short.gt(0.0)
        & ye_features.roc_medium.gt(0.0)
        & ye_features.above_ma
        & ye_features.ma_bias.le(float(ye_config["rules"]["max_entry_ma_bias"]))
    )
    technical_entry_pass = decision["entry_gate"] & ye_eligibility
    core_entry_available = technical_entry_pass[decision["core_symbols"]].any(axis=1)
    priority_entry_pass = technical_entry_pass.copy()
    if decision["challenger_symbols"]:
        priority_entry_pass.loc[:, decision["challenger_symbols"]] &= (
            ~core_entry_available.to_numpy()[:, None]
        )
    return pd.DataFrame({
        "date": str(latest.date()),
        "symbol": symbols,
        "name": [names[symbol] for symbol in symbols],
        "category": [categories[symbol] for symbol in symbols],
        "close": panel["close"].loc[latest, symbols].to_numpy(),
        "change_1d": panel["close"][symbols].pct_change(fill_method=None).loc[latest].to_numpy(),
        "trailing_amount_20d": trailing_amount.loc[latest, symbols].to_numpy(),
        "listed_sessions": listed_sessions.loc[latest, symbols].to_numpy(),
        "roc20": ye_features.roc_short.loc[latest].to_numpy(),
        "roc60": ye_features.roc_medium.loc[latest].to_numpy(),
        "momentum_score": ye_features.raw_score.loc[latest].to_numpy(),
        "selection_score": decision["entry_score"].loc[latest].to_numpy(),
        "ma120": ye_features.moving_average.loc[latest].to_numpy(),
        "ma120_bias": ye_features.ma_bias.loc[latest].to_numpy(),
        "above_ma120": ye_features.above_ma.loc[latest].to_numpy(),
        "rank": rank.loc[latest].to_numpy(),
        "rank_5d_ago": rank.shift(int(ye_config["rules"]["rank_change_short_days"])).loc[latest].to_numpy(),
        "rank_20d_ago": rank.shift(int(ye_config["rules"]["rank_change_long_days"])).loc[latest].to_numpy(),
        "dual_rank_decline": decision["dual_rank_decline"].loc[latest].to_numpy(),
        "pool_role": [
            "challenger" if symbol in decision["challenger_symbols"] else "core"
            for symbol in symbols
        ],
        "pool_eligible": ye_eligibility.loc[latest].to_numpy(),
        "base_entry_pass": base_pass.loc[latest].to_numpy(),
        "technical_entry_pass": technical_entry_pass.loc[latest].to_numpy(),
        "final_entry_pass": priority_entry_pass.loc[latest].to_numpy(),
        "normal_entry": decision["normal"].loc[latest].to_numpy(),
        "confirmed_normal_entry": decision["current_normal"].loc[latest].to_numpy(),
        "weak_edge_confirmed": decision["weak_edge_confirmed"].loc[latest].to_numpy(),
        "historical_fallback": decision["fallback"].loc[latest].to_numpy(),
        "emerging_entry": decision["emerging"].loc[latest].to_numpy(),
        "quality_extension": decision["quality_extension"].loc[latest].to_numpy(),
        "r2_20": decision["r2_20"].loc[latest].to_numpy(),
        "efficiency20": decision["efficiency20"].loc[latest].to_numpy(),
        "category_breadth": decision["category_breadth"].loc[latest].to_numpy(),
        "sentiment_matched_count": sentiment["matched_count"].loc[latest].to_numpy(),
        "sentiment_hot_score": sentiment["hot_score"].loc[latest].to_numpy(),
        "sentiment_count_acceleration": sentiment["count_acceleration"].loc[latest].to_numpy(),
        "sentiment_positive_dde_share": sentiment["positive_dde_share"].loc[latest].to_numpy(),
        "soft_exit_confirmation": decision["soft_exit_confirmation"].loc[latest].to_numpy(),
        "hot_exit_protection": decision["hot_exit_protection"].loc[latest].to_numpy(),
        "missing_data_soft_exit_protection": decision["missing_data_soft_exit_protection"].loc[latest].to_numpy(),
        "target_weight": ye_bundle.weights.loc[latest, symbols].to_numpy(),
    }).sort_values(["rank", "momentum_score"], ascending=[True, False], na_position="last")
