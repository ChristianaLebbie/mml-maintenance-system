"""Generic, dataset-agnostic validation helpers used during ingestion and
by the Phase 4 mapping suggester: detecting likely ID/date columns,
missingness, and duplicate rows. These make no assumptions about a
specific source's column names.
"""

from __future__ import annotations

import re

import pandas as pd

_ID_NAME_PATTERN = re.compile(r"\b(id|identifier|asset|machine|equipment)\b", re.IGNORECASE)
_DATE_NAME_PATTERN = re.compile(
    r"(date|timestamp|time|scheduled|completed|created|updated)", re.IGNORECASE
)


def detect_datetime_columns(df: pd.DataFrame, sample_size: int = 200) -> list[str]:
    """Return columns that are either already datetime-typed, have a
    date-like name, or whose values largely parse as dates."""
    found: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append(col)
            continue
        sample = series.dropna().head(sample_size)
        # A numeric column's name can still match date-ish words (e.g.
        # "Total Completed PMs" contains "completed") -- actual dtype
        # evidence must be checked before trusting the name pattern.
        if sample.empty or pd.api.types.is_numeric_dtype(sample):
            continue
        if _DATE_NAME_PATTERN.search(str(col)):
            found.append(col)
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.8:
            found.append(col)
    return found


def detect_id_like_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that plausibly identify a machine/asset: an
    ID-shaped name, or high-cardinality-but-repeating string values."""
    found: list[str] = []
    n = len(df)
    if n == 0:
        return found
    for col in df.columns:
        if _ID_NAME_PATTERN.search(str(col)):
            found.append(col)
            continue
        series = df[col]
        if series.dtype == object:
            nunique = series.nunique(dropna=True)
            if 1 < nunique < n and (series.value_counts().iloc[0] / n) < 0.9:
                found.append(col)
    return found


def compute_missingness(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {col: 0.0 for col in df.columns}
    return {col: round(100 * df[col].isna().mean(), 2) for col in df.columns}


def count_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    return [f"Missing required column: {c}" for c in missing]
