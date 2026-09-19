"""Feature scaling. Fit only on training data; the fitted scaler is
serialized so inference uses the exact same transform (see
models/preprocessors/ and src/inference)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def fit_scaler(df: pd.DataFrame, columns: list[str], method: str = "standard"):
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    scaler.fit(df[columns])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[columns] = scaler.transform(df[columns])
    return df


def save_scaler(scaler, path: str | Path) -> None:
    joblib.dump(scaler, path)


def load_scaler(path: str | Path):
    return joblib.load(path)
