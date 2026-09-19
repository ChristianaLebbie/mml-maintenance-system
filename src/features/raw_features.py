"""Raw feature group: the PM/work-order compliance metrics as-is."""

from __future__ import annotations

import pandas as pd

RAW_FEATURE_COLUMNS = [
    "overdue_pms",
    "overdue_wos",
    "total_completed_pms",
    "total_completed_wos",
    "downtime_minutes",
    "total_cost",
]


def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in RAW_FEATURE_COLUMNS if c in df.columns]
    return df[columns].copy()


def build_extra_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Any additional numeric columns from the uploaded source that didn't
    map to one of the known canonical fields (see src/ingestion/mapper.py's
    `extra__` pass-through) are still carried through as candidate model
    features. This is what lets the system learn from a differently-shaped
    mining dataset -- e.g. a different mine's CMMS export with its own
    operational metrics -- without a code change for every new column a
    new source happens to have. Non-numeric extras (free text, IDs) are
    deliberately excluded here: they remain visible in previews/reports
    but aren't blindly one-hot encoded, which would risk an unbounded
    number of columns from arbitrary free-text fields."""
    extra_columns = [
        c
        for c in df.columns
        if c.startswith("extra__") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[extra_columns].copy()
