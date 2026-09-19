"""Outlier detection (IQR-based) for numeric compliance metrics.

Detects and reports only -- never auto-removes. An asset with unusually
high downtime or overdue-WO count is exactly the kind of record a
intervention-priority model needs to see, not noise to discard.
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing.cleaning import NUMERIC_FIELDS


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a boolean DataFrame (same index as df) flagging IQR outliers
    per column. NaNs are never flagged."""
    columns = columns or [c for c in NUMERIC_FIELDS if c in df.columns]
    flags = pd.DataFrame(index=df.index)
    for col in columns:
        series = df[col]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags[col] = series.notna() & ((series < lower) | (series > upper))
    return flags


def summarize_outliers(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, int]:
    flags = detect_outliers_iqr(df, columns)
    return {col: int(flags[col].sum()) for col in flags.columns}
