"""Phase 15 tests: Power BI CSV export functions, against an isolated
in-memory database (never the real project database)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.database import Base
from database.models import Alert, Machine, ModelVersion, Prediction
from export_powerbi_data import (
    export_active_model_metadata,
    export_alerts,
    export_equipment_health,
    export_predictions,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)
    s = TestSession()
    machine = Machine(machine_identifier="4212-CR-001")
    s.add(machine)
    s.commit()

    model_version = ModelVersion(version="1.0.0", model_name="rf", model_type="random_forest", active=True)
    s.add(model_version)
    s.commit()

    prediction = Prediction(
        machine_id=machine.id,
        model_version_id=model_version.id,
        failure_probability=0.9,
        predicted_class=1,
        intervention_priority="High Priority",
    )
    s.add(prediction)
    s.commit()

    alert = Alert(
        prediction_id=prediction.id, machine_id=machine.id, intervention_priority="High Priority", message="test alert"
    )
    s.add(alert)
    s.commit()

    try:
        yield s
    finally:
        s.close()


def test_export_equipment_health_writes_expected_columns(session, tmp_path):
    export_equipment_health(session, tmp_path)
    df = pd.read_csv(tmp_path / "equipment_health.csv")
    assert list(df.columns) == ["machine", "timestamp", "probability", "intervention_priority"]
    assert df.iloc[0]["machine"] == "4212-CR-001"
    assert df.iloc[0]["intervention_priority"] == "High Priority"


def test_export_predictions_includes_model_version(session, tmp_path):
    export_predictions(session, tmp_path)
    df = pd.read_csv(tmp_path / "predictions.csv")
    assert df.iloc[0]["failure_probability"] == pytest.approx(0.9)


def test_export_alerts_includes_message(session, tmp_path):
    export_alerts(session, tmp_path)
    df = pd.read_csv(tmp_path / "alerts.csv")
    assert df.iloc[0]["message"] == "test alert"


def test_export_active_model_metadata_only_includes_active(session, tmp_path):
    export_active_model_metadata(session, tmp_path)
    df = pd.read_csv(tmp_path / "active_models.csv")
    assert len(df) == 1
    assert bool(df.iloc[0]["active"]) is True


def test_exports_handle_empty_database_gracefully(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    EmptySession = sessionmaker(bind=engine, expire_on_commit=False)
    s = EmptySession()
    try:
        export_equipment_health(s, tmp_path)
        export_predictions(s, tmp_path)
        export_alerts(s, tmp_path)
        export_active_model_metadata(s, tmp_path)
    finally:
        s.close()

    for name in ["equipment_health", "predictions", "alerts", "active_models"]:
        assert (tmp_path / f"{name}.csv").exists()
