"""Tests for the LSTM/GRU sequence model, built for the synthetic
sensor-telemetry demonstration dataset (docs/synthetic_demo_track.md).
Uses a tiny synthetic time series and minimal epochs -- this validates
correctness of the windowing and training/prediction plumbing, not model
quality (that's assessed with the real generated dataset, see
scripts/train_sequence_model.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.sequence_model import (
    build_lstm_model,
    build_sequences,
    predict_sequence_model,
    train_sequence_model,
)


@pytest.fixture()
def tiny_time_series():
    n_per_machine = 10
    machines = ["M1", "M2"]
    rows = []
    for m in machines:
        for i in range(n_per_machine):
            rows.append(
                {
                    "machine_id": m,
                    "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i),
                    "sensor_a": float(i) + (0 if m == "M1" else 100),
                    "sensor_b": float(i) * 2 + (0 if m == "M1" else 100),
                }
            )
    df = pd.DataFrame(rows)
    labels = pd.Series((df["sensor_a"] % 2 == 0).astype(int).values, index=df.index)
    return df, labels


def test_build_sequences_shape(tiny_time_series):
    df, labels = tiny_time_series
    X, y, idx = build_sequences(df, labels, sensor_columns=["sensor_a", "sensor_b"], sequence_length=4)

    # each machine has 10 rows -> 10 - 4 + 1 = 7 full-history windows
    assert X.shape == (14, 4, 2)
    assert y.shape == (14,)
    assert len(idx) == 14


def test_build_sequences_never_crosses_machine_boundary(tiny_time_series):
    df, labels = tiny_time_series
    X, y, idx = build_sequences(df, labels, sensor_columns=["sensor_a"], sequence_length=4)

    # M2's sensor_a values are offset by +100 -- no window should mix
    # M1 (small values) and M2 (values >= 100) together.
    for window in X:
        values = window[:, 0]
        assert (values >= 100).all() or (values < 100).all()


def test_build_sequences_empty_when_no_machine_has_enough_history():
    df = pd.DataFrame(
        {
            "machine_id": ["A", "A"],
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "sensor_a": [1.0, 2.0],
        }
    )
    labels = pd.Series([0, 0], index=df.index)
    X, y, idx = build_sequences(df, labels, sensor_columns=["sensor_a"], sequence_length=5)
    assert X.shape == (0, 5, 1)
    assert len(y) == 0


def test_build_lstm_model_output_shape():
    model = build_lstm_model(sequence_length=4, n_features=2, overrides={"hidden_units": 4, "dropout": 0.0})
    dummy = np.random.default_rng(0).normal(size=(3, 4, 2))
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (3, 1)
    assert (preds >= 0).all() and (preds <= 1).all()


def test_train_and_predict_sequence_model_runs_end_to_end(tiny_time_series):
    df, labels = tiny_time_series
    X, y, _ = build_sequences(df, labels, sensor_columns=["sensor_a", "sensor_b"], sequence_length=4)

    split = len(X) // 2
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = train_sequence_model(
        X_train,
        y_train,
        X_val,
        y_val,
        overrides={"hidden_units": 4, "dropout": 0.0, "epochs": 2, "batch_size": 4, "early_stopping_patience": 1},
    )
    probabilities = predict_sequence_model(model, X_val)
    assert probabilities.shape == (len(X_val),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
