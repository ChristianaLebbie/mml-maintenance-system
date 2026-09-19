"""Inference pipeline: standardized input row(s) -> validate -> features ->
predict -> intervention priority -> SHAP explanation -> structured
PredictionResult.

Must use the exact same feature construction as training (section 67).
The final `X = X[expected_columns]` reindex below is what actually
guarantees this: `expected_columns` is the exact feature set frozen in the
model's metadata at training time, so even if this function's own
label-formula-column exclusion (via resolve_priority_weights, mirroring
src.services.model_service.prepare_training_data) resolves slightly
differently against a small inference batch, any stray column is dropped
by the reindex and any expected-but-missing column is filled with 0.0.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.explainability.shap_service import build_explainer, explain_instance
from src.features.feature_pipeline import apply_categorical_encoder, build_numeric_features
from src.inference.model_loader import LoadedModel
from src.inference.priority_engine import classify_intervention_priority
from src.labels.failure_labels import resolve_priority_weights
from src.services.model_service import NUMERIC_FEATURE_GROUPS


@dataclass
class PredictionResult:
    machine_id: str
    prediction_timestamp: datetime.datetime
    failure_probability: float
    predicted_class: int
    intervention_priority: str
    model_version: str
    top_factors: list[dict] = field(default_factory=list)


def validate_input(df: pd.DataFrame) -> list[str]:
    problems = []
    if "machine_id" not in df.columns:
        problems.append("Missing required field: machine_id")
    return problems


def build_inference_features(df: pd.DataFrame, loaded: LoadedModel) -> pd.DataFrame:
    numeric = build_numeric_features(df, groups=NUMERIC_FEATURE_GROUPS)
    numeric = numeric.drop(columns=list(resolve_priority_weights(df)), errors="ignore")
    categorical = apply_categorical_encoder(df, loaded.encoder)
    X = pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)

    expected_columns = loaded.metadata.get("feature_columns")
    if expected_columns:
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[expected_columns]
    return X.fillna(0)


def predict(
    df: pd.DataFrame,
    loaded: LoadedModel,
    background: pd.DataFrame | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[PredictionResult]:
    problems = validate_input(df)
    if problems:
        raise ValueError(f"Invalid inference input: {problems}")

    X = build_inference_features(df, loaded)
    probabilities = loaded.model.predict_proba(X)[:, 1]

    watch_threshold = loaded.metadata.get("watch_threshold", 0.5)
    high_threshold = loaded.metadata.get("high_threshold", 0.8)

    explainer = None
    if background is not None and len(background) > 0:
        background_X = build_inference_features(background, loaded)
        explainer = build_explainer(loaded.model, background=background_X)

    results = []
    now = datetime.datetime.utcnow()
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if progress_callback is not None:
            progress_callback(f"Scoring machine {i + 1}/{total}...", i / total)
        probability = float(probabilities[i])
        intervention_priority = classify_intervention_priority(probability, watch_threshold, high_threshold)

        top_factors = []
        if explainer is not None:
            explanations = explain_instance(explainer, X.iloc[[i]])
            top_factors = [
                {
                    "feature": e.feature_name,
                    "value": e.feature_value,
                    "shap_value": e.shap_value,
                    "rank": e.importance_rank,
                }
                for e in explanations[:5]
            ]

        results.append(
            PredictionResult(
                machine_id=str(row["machine_id"]),
                prediction_timestamp=now,
                failure_probability=probability,
                predicted_class=int(probability >= 0.5),
                intervention_priority=intervention_priority,
                model_version=loaded.version,
                top_factors=top_factors,
            )
        )
    if progress_callback is not None:
        progress_callback("Done", 1.0)
    return results
