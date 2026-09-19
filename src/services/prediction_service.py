"""Ties the inference pipeline to persistence: run predictions, save
Prediction + PredictionExplanation rows, and create an Alert when
warranted. This is what the Streamlit "Run Prediction" page calls."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from database.database import get_session
from database.models import Alert, Machine, Prediction, PredictionExplanation
from src.alerts.alert_service import build_alert_message, should_alert
from src.inference.model_loader import LoadedModel
from src.inference.predictor import PredictionResult, predict
from src.utils.logger import get_logger

logger = get_logger("services.prediction_service")


def _get_or_create_machine(session, machine_id: str) -> Machine:
    machine = session.query(Machine).filter_by(machine_identifier=machine_id).first()
    if machine is None:
        machine = Machine(machine_identifier=machine_id)
        session.add(machine)
        session.flush()
    return machine


def run_and_save_predictions(
    df: pd.DataFrame,
    loaded: LoadedModel,
    background: pd.DataFrame | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[PredictionResult]:
    results = predict(df, loaded, background=background, progress_callback=progress_callback)

    with get_session() as session:
        for result in results:
            machine = _get_or_create_machine(session, result.machine_id)

            prediction = Prediction(
                machine_id=machine.id,
                model_version_id=loaded.model_version_id,
                prediction_timestamp=result.prediction_timestamp,
                failure_probability=result.failure_probability,
                predicted_class=result.predicted_class,
                intervention_priority=result.intervention_priority,
            )
            session.add(prediction)
            session.flush()

            for factor in result.top_factors:
                session.add(
                    PredictionExplanation(
                        prediction_id=prediction.id,
                        feature_name=factor["feature"],
                        feature_value=factor["value"],
                        shap_value=factor["shap_value"],
                        importance_rank=factor["rank"],
                    )
                )

            if should_alert(result):
                session.add(
                    Alert(
                        prediction_id=prediction.id,
                        machine_id=machine.id,
                        intervention_priority=result.intervention_priority,
                        message=build_alert_message(result),
                    )
                )

    logger.info(f"Saved {len(results)} predictions ({sum(should_alert(r) for r in results)} alerts).")
    return results
