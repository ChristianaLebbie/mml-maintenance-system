"""Alert-generation logic: decide whether a prediction warrants an alert,
and what it should say. Persistence (Phase 13 database wiring) happens in
src/services/prediction_service.py, which calls this after saving a
Prediction row."""

from __future__ import annotations

from src.inference.predictor import PredictionResult

ALERTABLE_PRIORITY_LEVELS = {"Watch", "High Priority"}


def should_alert(prediction: PredictionResult) -> bool:
    return prediction.intervention_priority in ALERTABLE_PRIORITY_LEVELS


def build_alert_message(prediction: PredictionResult) -> str:
    top_factor = prediction.top_factors[0]["feature"] if prediction.top_factors else None
    base = (
        f"Machine {prediction.machine_id} classified as {prediction.intervention_priority} "
        f"(failure probability {prediction.failure_probability:.2f}, model {prediction.model_version})."
    )
    if top_factor:
        base += f" Top contributing factor: {top_factor}."
    return base
