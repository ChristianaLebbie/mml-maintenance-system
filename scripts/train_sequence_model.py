"""Train and evaluate the LSTM sequence model on the SYNTHETIC
sensor-telemetry demonstration dataset (see
data/raw/synthetic/README.md and docs/synthetic_demo_track.md -- this is
fabricated demonstration data, never the real Data Repo).

Also trains a flat Random Forest on the same chronological split (using
lag/rolling/trend features flattened per row) as a "general model" point
of comparison against the sequence model, per section 52.

Usage:
    python scripts/train_sequence_model.py [horizon_hours] [sequence_length]

Example:
    python scripts/train_sequence_model.py 24 24
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import PATHS  # noqa: E402
from src.features.lag_features import build_lag_features  # noqa: E402
from src.features.temporal_features import build_trend_features  # noqa: E402
from src.labels.failure_labels import construct_failure_within_horizon_label  # noqa: E402
from src.training.evaluation import evaluate_classifier  # noqa: E402
from src.training.random_forest import train_random_forest  # noqa: E402
from src.training.sequence_model import (  # noqa: E402
    build_sequences,
    predict_sequence_model,
    train_sequence_model,
)
from src.training.validation import chronological_split  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.train_sequence_model")

SENSOR_COLUMNS = ["voltage", "rotation", "pressure", "vibration"]


def main() -> None:
    horizon_hours = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    sequence_length = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    data_path = PATHS["raw_excel"].parent / "synthetic" / "synthetic_sensor_telemetry.csv"
    if not data_path.exists():
        print(f"No synthetic dataset found at {data_path}. Run scripts/generate_synthetic_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"failure_event": "failure_label"})
    logger.info(f"Loaded synthetic dataset: {len(df)} rows, {df['machine_id'].nunique()} machines")

    labels = construct_failure_within_horizon_label(df, horizon_hours=horizon_hours)
    logger.info(f"Positive labels (failure within {horizon_hours}h): {labels.sum()} / {len(labels)}")

    split = chronological_split(df, time_col="timestamp", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # --- LSTM sequence model ---
    X, y, seq_index = build_sequences(df, labels, SENSOR_COLUMNS, sequence_length)
    train_mask = seq_index.isin(split.train_idx)
    val_mask = seq_index.isin(split.val_idx)
    test_mask = seq_index.isin(split.test_idx)

    # Override epochs/batch_size for this demo run: ~50k training sequences
    # makes the config default (epochs=50, batch_size=32) slow on CPU for a
    # demonstration script; early stopping still applies.
    lstm_model = train_sequence_model(
        X[train_mask],
        y[train_mask],
        X[val_mask],
        y[val_mask],
        overrides={"epochs": 15, "batch_size": 128},
    )
    lstm_proba = predict_sequence_model(lstm_model, X[test_mask])
    lstm_preds = (lstm_proba >= 0.5).astype(int)
    lstm_result = evaluate_classifier(y[test_mask], lstm_preds, lstm_proba)

    # --- Flat Random Forest baseline (general model) on lag/rolling/trend features ---
    lag_feats = build_lag_features(df, sensor_columns=SENSOR_COLUMNS)
    trend_feats = build_trend_features(df, sensor_columns=SENSOR_COLUMNS)
    flat_features = pd.concat([lag_feats, trend_feats], axis=1).fillna(0)

    rf_model = train_random_forest(flat_features.loc[split.train_idx], labels.loc[split.train_idx])
    rf_proba = rf_model.predict_proba(flat_features.loc[split.test_idx])[:, 1]
    rf_preds = (rf_proba >= 0.5).astype(int)
    rf_result = evaluate_classifier(labels.loc[split.test_idx].values, rf_preds, rf_proba)

    print(f"\n{'Model':25s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s} {'ROC-AUC':>8s} {'PR-AUC':>7s}")
    for name, result in [("random_forest (flat)", rf_result), ("lstm (sequence)", lstm_result)]:
        print(
            f"{name:25s} {result.accuracy:9.3f} {result.precision:10.3f} "
            f"{result.recall:8.3f} {result.f1:6.3f} {result.roc_auc or 0:8.3f} {result.pr_auc or 0:7.3f}"
        )

    out_dir = PATHS["reports_experiments"] / "synthetic_sequence_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "random_forest_flat_metrics.json").write_text(
        json.dumps(rf_result.__dict__, indent=2), encoding="utf-8"
    )
    (out_dir / "lstm_metrics.json").write_text(json.dumps(lstm_result.__dict__, indent=2), encoding="utf-8")

    models_dir = PATHS["models_trained"]
    models_dir.mkdir(parents=True, exist_ok=True)
    lstm_model.save(models_dir / "lstm_synthetic_demo.keras")

    print(f"\nMetrics saved to: {out_dir}")
    print(f"LSTM model saved to: {models_dir / 'lstm_synthetic_demo.keras'}")
    print(
        "\nReminder: this run used the SYNTHETIC demonstration dataset "
        "(data/raw/synthetic/), not the real Data Repo."
    )


if __name__ == "__main__":
    main()
