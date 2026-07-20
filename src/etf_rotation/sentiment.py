from __future__ import annotations

from pathlib import Path

import pandas as pd


SENTIMENT_FIELDS = (
    "matched_count",
    "positive_dde_share",
    "count_acceleration",
    "hot_score",
)


def load_sentiment_matrices(
    path: Path, calendar: pd.DatetimeIndex, symbols: list[str]
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Load reviewed daily sentiment features without inventing missing days."""

    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    matrices = {
        field: frame.pivot(index="date", columns="symbol", values=field).reindex(
            index=calendar, columns=symbols
        )
        for field in SENTIMENT_FIELDS
    }
    available_dates = set(frame["date"].unique())
    available = pd.Series(calendar.isin(available_dates), index=calendar, dtype=bool)
    return matrices, available


def broadcast(mask: pd.Series, symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {symbol: mask.astype(bool).to_numpy(copy=True) for symbol in symbols},
        index=mask.index,
    )
