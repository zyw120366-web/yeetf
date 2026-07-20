from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SignalBundle:
    weights: pd.DataFrame
    exposure: pd.Series
    diagnostics: pd.DataFrame
