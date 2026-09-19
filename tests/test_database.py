"""Phase 2 tests: schema creation and basic CRUD across the core tables.

Uses an in-memory SQLite database (via a fixture) so tests never touch the
real project database file.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.database import Base
from database.models import (
    Alert,
    Dataset,
    Machine,
    MaintenanceEvent,
    ModelVersion,
    Prediction,
    PredictionExplanation,
)
from database.repositories.dataset_repository import DatasetRepository
from database.repositories.machine_repository import MachineRepository


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    s = TestSession()
    try:
        yield s
    finally:
        s.close()


def test_create_dataset(session):
    ds = Dataset(name="test-dataset", source_type="excel", source_filename="test.xlsx")
    session.add(ds)
    session.commit()
    assert ds.id is not None
    assert ds.status == "Imported"


def test_machine_equipment_category_is_nullable(session):
    ds = Dataset(name="ds", source_type="excel", source_filename="f.xlsx")
    session.add(ds)
    session.commit()

    m = Machine(machine_identifier="4212-CR-001", dataset_id=ds.id)
    session.add(m)
    session.commit()
    assert m.equipment_category is None


def test_prediction_and_explanation_relationship(session):
    ds = Dataset(name="ds", source_type="excel", source_filename="f.xlsx")
    session.add(ds)
    session.commit()

    machine = Machine(machine_identifier="4212-CR-001", dataset_id=ds.id)
    model_version = ModelVersion(
        version="1.0.0", model_name="rf", model_type="random_forest", dataset_id=ds.id
    )
    session.add_all([machine, model_version])
    session.commit()

    prediction = Prediction(
        machine_id=machine.id,
        model_version_id=model_version.id,
        failure_probability=0.73,
        predicted_class=1,
        intervention_priority="High Priority",
    )
    session.add(prediction)
    session.commit()

    explanation = PredictionExplanation(
        prediction_id=prediction.id,
        feature_name="overdue_pms",
        feature_value=3.0,
        shap_value=0.21,
        importance_rank=1,
    )
    session.add(explanation)
    session.commit()

    fetched = session.get(Prediction, prediction.id)
    assert len(fetched.explanations) == 1
    assert fetched.explanations[0].feature_name == "overdue_pms"


def test_alert_lifecycle_fields(session):
    ds = Dataset(name="ds", source_type="excel", source_filename="f.xlsx")
    machine = Machine(machine_identifier="4212-CR-001")
    session.add_all([ds, machine])
    session.commit()

    model_version = ModelVersion(
        version="1.0.0", model_name="rf", model_type="random_forest", dataset_id=ds.id
    )
    session.add(model_version)
    session.commit()

    prediction = Prediction(
        machine_id=machine.id,
        model_version_id=model_version.id,
        failure_probability=0.9,
        predicted_class=1,
        intervention_priority="High Priority",
    )
    session.add(prediction)
    session.commit()

    alert = Alert(
        prediction_id=prediction.id,
        machine_id=machine.id,
        intervention_priority="High Priority",
        message="Overdue PM threshold exceeded",
    )
    session.add(alert)
    session.commit()

    assert alert.status == "New"
    assert alert.acknowledged_at is None

    alert.status = "Acknowledged"
    alert.acknowledged_at = datetime.datetime.utcnow()
    session.commit()

    refreshed = session.get(Alert, alert.id)
    assert refreshed.status == "Acknowledged"
    assert refreshed.acknowledged_at is not None


def test_maintenance_event_foreign_keys(session):
    ds = Dataset(name="ds", source_type="excel", source_filename="f.xlsx")
    machine = Machine(machine_identifier="4212-CR-001")
    session.add_all([ds, machine])
    session.commit()

    event = MaintenanceEvent(
        machine_id=machine.id,
        event_timestamp=datetime.datetime.utcnow(),
        maintenance_type="Preventive",
        dataset_id=ds.id,
    )
    session.add(event)
    session.commit()

    assert event.id is not None
    assert event.machine.machine_identifier == "4212-CR-001"


def test_dataset_repository_crud(session):
    repo = DatasetRepository(session)
    ds = repo.create(name="ds1", source_type="excel", source_filename="f.xlsx")
    assert ds.id is not None

    fetched = repo.get(ds.id)
    assert fetched.name == "ds1"

    updated = repo.update_status(ds.id, "Validated")
    assert updated.status == "Validated"

    assert repo.delete(ds.id) is True
    assert repo.get(ds.id) is None


def test_machine_repository_crud(session):
    dataset_repo = DatasetRepository(session)
    ds = dataset_repo.create(name="ds1", source_type="excel", source_filename="f.xlsx")

    machine_repo = MachineRepository(session)
    m = machine_repo.create(machine_identifier="4212-CR-001", dataset_id=ds.id)

    assert machine_repo.get_by_identifier("4212-CR-001").id == m.id
    assert len(machine_repo.list(dataset_id=ds.id)) == 1
    assert machine_repo.delete(m.id) is True
    assert machine_repo.get(m.id) is None
