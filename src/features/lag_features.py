"""Lag and rolling-window feature group.

Only meaningful for genuine time-series data: repeated, chronologically-
ordered observations per machine (e.g. the synthetic sensor-telemetry demo
dataset, docs/synthetic_demo_track.md). The real CMMS export is a single
point-in-time snapshot (~1 row per machine -- see dataset feasibility
reports), so this stays an honest no-op for it: fabricating lag columns
from a single observation per machine would manufacture false signal.

Leakage rule enforced here: every lag/rolling value at row t is computed
using only rows at or before t for the SAME machine -- grouping by
`group_col` and sorting by `time_col` before shifting/rolling guarantees
no cross-machine contamination and no use of future values.
"""

from __future__ import annotations

import pandas as pd


def build_lag_features(
    df: pd.DataFrame,
    sensor_columns: list[str] | None = None,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    group_col: str = "machine_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    if time_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame(index=df.index)

    sensor_columns = sensor_columns or [
        c for c in ("voltage", "rotation", "pressure", "vibration") if c in df.columns
    ]
    if not sensor_columns:
        return pd.DataFrame(index=df.index)

    lags = lags if lags is not None else [1, 3, 6]
    rolling_windows = rolling_windows if rolling_windows is not None else [3, 6, 12, 24]

    ordered = df.sort_values([group_col, time_col])
    grouped = ordered.groupby(group_col, sort=False)

    features = pd.DataFrame(index=ordered.index)
    for col in sensor_columns:
        for lag in lags:
            features[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
        for window in rolling_windows:
            rolling = grouped[col].rolling(window=window, min_periods=1)
            features[f"{col}_rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            features[f"{col}_rolling_std_{window}"] = rolling.std().reset_index(level=0, drop=True)

    return features.reindex(df.index)
