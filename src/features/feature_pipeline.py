"""Combines the active feature groups into one feature matrix.

Categorical encoding follows the same fit-on-train/apply-everywhere
lifecycle as src/preprocessing/scaling.py (via sklearn's OneHotEncoder with
handle_unknown="ignore") so a category seen only in validation/test never
leaks structure into training and never crashes inference.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.features.event_features import build_event_features
from src.features.frequency_features import build_frequency_features
from src.features.lag_features import build_lag_features
from src.features.raw_features import build_extra_numeric_features, build_raw_features
from src.features.statistical_features import build_statistical_features
from src.features.temporal_features import build_temporal_features

NUMERIC_GROUP_BUILDERS = {
    "raw": build_raw_features,
    "temporal": build_temporal_features,
    "statistical": build_statistical_features,
    "events": build_event_features,
    "frequency": build_frequency_features,
    "lags": build_lag_features,
    "extra": build_extra_numeric_features,
}

CATEGORICAL_COLUMNS = ["equipment_category", "manufacturer", "criticality"]

DEFAULT_GROUPS = ["raw", "temporal", "statistical", "extra"]


def build_numeric_features(df: pd.DataFrame, groups: list[str] | None = None) -> pd.DataFrame:
    groups = groups or DEFAULT_GROUPS
    unknown = set(groups) - set(NUMERIC_GROUP_BUILDERS)
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}")

    parts = [NUMERIC_GROUP_BUILDERS[group](df) for group in groups]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def fit_categorical_encoder(
    df: pd.DataFrame, columns: list[str] | None = None
) -> OneHotEncoder:
    columns = columns or [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df[columns].astype(str))
    return encoder


def apply_categorical_encoder(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    # sklearn records the fitted column order/names as `feature_names_in_`
    # whenever fit() is called with a DataFrame -- reuse it so apply()
    # always selects the same columns fit() was trained on.
    columns = list(encoder.feature_names_in_)
    encoded = encoder.transform(df[columns].astype(str))
    names = encoder.get_feature_names_out(columns)
    return pd.DataFrame(encoded, columns=names, index=df.index)


def save_encoder(encoder: OneHotEncoder, path: str | Path) -> None:
    joblib.dump(encoder, path)


def load_encoder(path: str | Path) -> OneHotEncoder:
    return joblib.load(path)


def build_features(
    df: pd.DataFrame,
    groups: list[str] | None = None,
    categorical_encoder: OneHotEncoder | None = None,
) -> pd.DataFrame:
    """Combine numeric feature groups with (optionally) already-fitted
    categorical encoding. Pass `categorical_encoder=None` to skip
    categorical features entirely (e.g. for a quick numeric-only baseline)."""
    numeric = build_numeric_features(df, groups)
    if categorical_encoder is None:
        return numeric
    categorical = apply_categorical_encoder(df, categorical_encoder)
    return pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
