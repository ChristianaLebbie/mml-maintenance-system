"""Generic cleaning of a standardized DataFrame: whitespace/casing
normalization for text fields, numeric coercion for compliance metrics,
and duplicate-row reporting (never silently dropped without being counted
first, per the project's "report before removing" rule)."""

from __future__ import annotations

import pandas as pd

NUMERIC_FIELDS = [
    "overdue_pms",
    "overdue_wos",
    "total_completed_pms",
    "total_completed_wos",
    "downtime_minutes",
    "total_cost",
]
TEXT_FIELDS = ["machine_id", "equipment_category", "manufacturer", "criticality"]
DATE_FIELDS = ["last_completed_pm", "last_completed_wo"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TEXT_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: v.strip() if isinstance(v, str) else v
            )
            df[col] = df[col].replace({"": None, "nan": None, "None": None})

    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in DATE_FIELDS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    return df


def count_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (deduplicated_df, n_dropped). Callers must report n_dropped
    before discarding it -- never remove duplicates silently."""
    n_dropped = count_duplicate_rows(df)
    return df.drop_duplicates().reset_index(drop=True), n_dropped
