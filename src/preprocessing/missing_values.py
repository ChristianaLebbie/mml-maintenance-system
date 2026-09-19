"""Configurable missing-value handling for the standardized CMMS schema.

Numeric PM/WO compliance metrics default to 0 when missing -- a missing
"Overdue PMs" count in this data means the metric was not tracked for that
asset, not that it should be imputed from other machines' distributions
(which would fabricate compliance history). Categorical asset attributes
default to an explicit "Unknown" label rather than the most-frequent value,
so "we don't know this asset's manufacturer" is never silently turned into
"this asset's manufacturer is the most common one".
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing.cleaning import NUMERIC_FIELDS, TEXT_FIELDS

UNKNOWN_LABEL = "Unknown"


def summarize_missing(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {col: 0.0 for col in df.columns}
    return {col: round(100 * df[col].isna().mean(), 2) for col in df.columns}


def fill_missing(df: pd.DataFrame, numeric_fill: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = df[col].fillna(numeric_fill)
    for col in TEXT_FIELDS:
        if col == "machine_id":
            continue  # never impute the identifier itself
        if col in df.columns:
            df[col] = df[col].fillna(UNKNOWN_LABEL)
    return df
