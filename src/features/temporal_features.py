"""Temporal feature group, adapted for a cross-sectional CMMS snapshot:
recency since each asset's last completed PM/work order, relative to a
reference date (the snapshot's own max date by default). This is not the
rolling-window/lag style temporal features a repeated-observation sensor
time series would use (see lag_features.py) -- there is only one row per
machine here, so "temporal" signal means "how long ago", not "how has this
changed over recent observations"."""

from __future__ import annotations

import datetime

import pandas as pd

RECENCY_SOURCE_COLUMNS = {
    "last_completed_pm": "days_since_last_completed_pm",
    "last_completed_wo": "days_since_last_completed_wo",
}


def build_temporal_features(
    df: pd.DataFrame, reference_date: datetime.datetime | None = None
) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    date_cols = [c for c in RECENCY_SOURCE_COLUMNS if c in df.columns and df[c].notna().any()]
    if not date_cols:
        return features

    if reference_date is None:
        reference_date = pd.concat([df[c] for c in date_cols]).max()

    for source_col, feature_name in RECENCY_SOURCE_COLUMNS.items():
        if source_col not in df.columns:
            continue
        parsed = pd.to_datetime(df[source_col], errors="coerce", format="mixed")
        features[feature_name] = (reference_date - parsed).dt.days

    return features


def build_trend_features(
    df: pd.DataFrame,
    sensor_columns: list[str] | None = None,
    group_col: str = "machine_id",
    time_col: str = "timestamp",
    ewma_span: int = 6,
) -> pd.DataFrame:
    """Trend features for genuine time-series data: first difference,
    percentage change, and EWMA, computed per machine in chronological
    order (never across machine boundaries, never using future values).
    A no-op when `time_col`/`group_col` aren't present -- e.g. the
    cross-sectional CMMS snapshot, which has no per-machine time series to
    compute a trend from (use build_temporal_features for that data
    instead)."""
    if time_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame(index=df.index)

    sensor_columns = sensor_columns or [
        c for c in ("voltage", "rotation", "pressure", "vibration") if c in df.columns
    ]
    if not sensor_columns:
        return pd.DataFrame(index=df.index)

    ordered = df.sort_values([group_col, time_col])
    grouped = ordered.groupby(group_col, sort=False)

    features = pd.DataFrame(index=ordered.index)
    for col in sensor_columns:
        features[f"{col}_diff_1"] = grouped[col].diff(1)
        features[f"{col}_pct_change_1"] = grouped[col].pct_change(1)
        features[f"{col}_ewma_{ewma_span}"] = grouped[col].transform(
            lambda s: s.ewm(span=ewma_span, adjust=False).mean()
        )

    return features.reindex(df.index)
