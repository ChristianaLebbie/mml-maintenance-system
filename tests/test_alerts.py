"""Phase 13 tests: alert-generation decision logic and message content."""

from __future__ import annotations

import datetime

from src.alerts.alert_service import build_alert_message, should_alert
from src.inference.predictor import PredictionResult


def _make_result(intervention_priority: str, top_factors=None) -> PredictionResult:
    return PredictionResult(
        machine_id="4212-CR-001",
        prediction_timestamp=datetime.datetime.utcnow(),
        failure_probability=0.85,
        predicted_class=1,
        intervention_priority=intervention_priority,
        model_version="1.0.0",
        top_factors=top_factors or [],
    )


def test_should_alert_true_for_high_priority():
    assert should_alert(_make_result("High Priority")) is True


def test_should_alert_true_for_watch():
    assert should_alert(_make_result("Watch")) is True


def test_should_alert_false_for_normal():
    assert should_alert(_make_result("Normal")) is False


def test_build_alert_message_includes_machine_and_priority():
    message = build_alert_message(_make_result("High Priority"))
    assert "4212-CR-001" in message
    assert "High Priority" in message


def test_build_alert_message_includes_top_factor_when_present():
    factors = [{"feature": "total_completed_pms", "value": 2, "shap_value": 0.3, "rank": 1}]
    message = build_alert_message(_make_result("High Priority", top_factors=factors))
    assert "total_completed_pms" in message


def test_build_alert_message_handles_no_factors_gracefully():
    message = build_alert_message(_make_result("Watch", top_factors=[]))
    assert "Watch" in message
