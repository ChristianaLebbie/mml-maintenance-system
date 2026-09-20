# cSpell:words CMMS crossentropy
"""LSTM/GRU sequence model (section 27) -- only meaningful for genuine
per-machine time series. Not used for the real CMMS export (a single
snapshot per machine has no sequence to model); built for the synthetic
sensor-telemetry demonstration dataset, see docs/synthetic_demo_track.md.

Sequences contain only past-and-current observations relative to their
label timestamp -- `build_sequences()` windows end at t using
[t-sequence_length+1, t], and the label at t is whatever the caller passed
in (e.g. from construct_failure_within_horizon_label, which is allowed to
look forward -- that's the label, not a feature). No future sensor reading
is ever included in X.

Uses X/y (capitalized) for feature/label arrays throughout, matching
standard scikit-learn/Keras ML convention rather than PEP8 variable
naming -- intentional, not a style oversight.
"""
# pylint: disable=invalid-name

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from config.settings import MODEL_CONFIG, RANDOM_SEED

# `from tensorflow import keras` trips static checkers (Pylint E0611 /
# Pylance reportMissingModuleSource) because TensorFlow loads `keras` as a
# dynamic submodule that isn't visible to static analysis -- a well-known,
# harmless false alarm (the code runs fine either way). Accessing it as an
# attribute of the already-imported `tf` module, instead of importing it by
# name directly, sidesteps that specific check.
keras = tf.keras

tf.random.set_seed(RANDOM_SEED)


def build_sequences(
    df: pd.DataFrame,
    labels: pd.Series,
    sensor_columns: list[str],
    sequence_length: int,
    group_col: str = "machine_id",
    time_col: str = "timestamp",
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Returns (X, y, source_index) where X has shape
    (n_sequences, sequence_length, n_features). Only full-history windows
    are included (no zero-padding) -- a machine's first `sequence_length-1`
    rows produce no sequence."""
    ordered = df.sort_values([group_col, time_col])
    sensor_values = ordered[sensor_columns].to_numpy(dtype=float)
    ordered_labels = labels.loc[ordered.index].to_numpy()

    X_list, y_list, idx_list = [], [], []
    for _, group in ordered.groupby(group_col, sort=False):
        positions = ordered.index.get_indexer(group.index)
        values = sensor_values[positions]
        group_labels = ordered_labels[positions]
        n = len(group)
        for end in range(sequence_length - 1, n):
            start = end - sequence_length + 1
            X_list.append(values[start : end + 1])
            y_list.append(group_labels[end])
            idx_list.append(group.index[end])

    if not X_list:
        return (
            np.empty((0, sequence_length, len(sensor_columns))),
            np.empty((0,)),
            pd.Index([]),
        )

    return np.stack(X_list), np.array(y_list), pd.Index(idx_list)


def build_lstm_model(
    sequence_length: int, n_features: int, overrides: dict | None = None
) -> keras.Model:
    """Build and compile an untrained LSTM/GRU binary classifier for
    sequences of shape (sequence_length, n_features)."""
    cfg = {**MODEL_CONFIG["sequence_model"], **(overrides or {})}
    layer_cls = keras.layers.LSTM if cfg.get("type", "LSTM") == "LSTM" else keras.layers.GRU

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, n_features)),
            layer_cls(cfg["hidden_units"]),
            keras.layers.Dropout(cfg["dropout"]),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_sequence_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    overrides: dict | None = None,
) -> keras.Model:
    """Train an LSTM/GRU model on (X_train, y_train), early-stopping on
    (X_val, y_val), and return the fitted model."""
    cfg = {**MODEL_CONFIG["sequence_model"], **(overrides or {})}
    model = build_lstm_model(X_train.shape[1], X_train.shape[2], overrides)

    class_weight = None
    positive_rate = y_train.mean()
    if 0 < positive_rate < 1:
        class_weight = {0: 1.0, 1: (1 - positive_rate) / positive_rate}

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg["early_stopping_patience"], restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=0,
    )
    return model


def predict_sequence_model(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Return per-sequence predicted probabilities for X."""
    return model.predict(X, verbose=0).ravel()
