"""Tests for the higher-level services that wire ingestion/mapping/
training to the database and filesystem: src/services/dataset_service.py
and src/services/model_service.py -- in particular `train_and_activate`,
the one-call flow behind the Import Dataset page's "Train Models" button.

Uses an in-memory SQLite database and a tmp_path filesystem monkeypatched
into place so these tests never touch the real project database file or
write real model/report artifacts into the project tree.
"""

from __future__ import annotations

import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.services.dataset_service as dataset_service
import src.services.model_service as model_service
from database.database import Base
from database.models import (
    Alert,
    Dataset,
    ExperimentRun,
    Machine,
    MaintenanceEvent,
    ModelVersion,
    Prediction,
    PredictionExplanation,
)


@pytest.fixture()
def isolated_db(monkeypatch):
    """Point model_service.get_session and dataset_service.get_session at
    a throwaway in-memory database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def fake_get_session():
        session = TestSession()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(model_service, "get_session", fake_get_session)
    monkeypatch.setattr(dataset_service, "get_session", fake_get_session)
    return TestSession


@pytest.fixture()
def isolated_paths(monkeypatch, tmp_path):
    """Redirect every filesystem path model_service/dataset_service write
    to under a pytest tmp_path, so tests never write real artifacts into
    (or delete real files from) the project's data/models/reports trees."""
    fake_paths = {
        "models_trained": tmp_path / "models" / "trained",
        "models_preprocessors": tmp_path / "models" / "preprocessors",
        "models_metadata": tmp_path / "models" / "metadata",
        "reports_experiments": tmp_path / "reports" / "experiments",
        "processed": tmp_path / "data" / "processed",
        "reports_feasibility": tmp_path / "reports" / "dataset_feasibility",
    }
    monkeypatch.setattr(model_service, "PATHS", fake_paths)
    monkeypatch.setattr(dataset_service, "PATHS", fake_paths)
    return fake_paths


@pytest.fixture()
def isolated_mapping_config(monkeypatch, tmp_path):
    """Redirect the dataset-mapping YAML to a tmp file so delete_dataset's
    call to delete_mapping never touches the real config/datasets.yaml."""
    import src.ingestion.mapper as mapper

    fake_config_path = tmp_path / "datasets.yaml"
    monkeypatch.setattr(mapper, "CONFIG_PATH", fake_config_path)
    return fake_config_path


@pytest.fixture()
def tiny_cmms_df():
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "machine_id": [f"M-{i}" for i in range(n)],
            "equipment_category": rng.choice(["Pump", "Crusher", "Motor"], size=n),
            "manufacturer": rng.choice(["KSB", "WEG"], size=n),
            "criticality": rng.choice(["High", "Low"], size=n),
            "overdue_pms": rng.integers(0, 10, size=n),
            "overdue_wos": rng.integers(0, 5, size=n),
            "total_completed_pms": rng.integers(0, 50, size=n),
            "total_completed_wos": rng.integers(0, 10, size=n),
            "downtime_minutes": rng.uniform(0, 500, size=n),
            "total_cost": rng.uniform(0, 5000, size=n),
            "last_completed_pm": pd.NaT,
            "last_completed_wo": pd.NaT,
            "source_dataset": "unit-test.csv",
        }
    )


def _seed_dataset(session_factory, name: str) -> None:
    session = session_factory()
    session.add(Dataset(name=name, source_type="csv", source_filename=f"{name}.csv"))
    session.commit()
    session.close()


def test_train_and_activate_registers_one_experiment_run_per_model(
    isolated_db, isolated_paths, tiny_cmms_df
):
    _seed_dataset(isolated_db, "unit_test_ds")

    result = model_service.train_and_activate("unit_test_ds", version="test-v1", df=tiny_cmms_df)

    assert result.best_model_name in {"logistic_regression", "decision_tree", "random_forest", "xgboost"}
    assert result.selection_report_path.exists()

    session = isolated_db()
    try:
        runs = session.query(ExperimentRun).all()
        assert len(runs) == 4
        assert {r.model for r in runs} == {
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "xgboost",
        }
    finally:
        session.close()


def test_train_and_activate_registers_exactly_one_active_model_version(
    isolated_db, isolated_paths, tiny_cmms_df
):
    _seed_dataset(isolated_db, "unit_test_ds")
    result = model_service.train_and_activate("unit_test_ds", version="test-v1", df=tiny_cmms_df)

    session = isolated_db()
    try:
        active = session.query(ModelVersion).filter_by(active=True).all()
        assert len(active) == 1
        assert active[0].model_type == result.best_model_name
        assert active[0].id == result.model_version_id
    finally:
        session.close()


def test_train_and_activate_deactivates_previous_version_on_retrain(
    isolated_db, isolated_paths, tiny_cmms_df
):
    _seed_dataset(isolated_db, "unit_test_ds")
    first = model_service.train_and_activate("unit_test_ds", version="v1", df=tiny_cmms_df)
    second = model_service.train_and_activate("unit_test_ds", version="v2", df=tiny_cmms_df)

    session = isolated_db()
    try:
        active = session.query(ModelVersion).filter_by(active=True).all()
        assert len(active) == 1
        assert active[0].id == second.model_version_id
        assert active[0].id != first.model_version_id

        first_version_row = session.query(ModelVersion).filter_by(id=first.model_version_id).one()
        assert first_version_row.active is False
    finally:
        session.close()


def test_train_and_activate_writes_model_artifacts_under_isolated_paths(
    isolated_db, isolated_paths, tiny_cmms_df
):
    _seed_dataset(isolated_db, "unit_test_ds")
    model_service.train_and_activate("unit_test_ds", version="test-v1", df=tiny_cmms_df)

    trained_files = list(isolated_paths["models_trained"].glob("*_test-v1.joblib"))
    assert len(trained_files) == 4


def test_train_and_activate_reports_progress_from_zero_to_one(
    isolated_db, isolated_paths, tiny_cmms_df
):
    _seed_dataset(isolated_db, "unit_test_ds")
    calls: list[tuple[str, float]] = []

    model_service.train_and_activate(
        "unit_test_ds",
        version="test-v1",
        df=tiny_cmms_df,
        progress_callback=lambda msg, frac: calls.append((msg, frac)),
    )

    assert len(calls) > 1
    fractions = [f for _, f in calls]
    assert fractions[0] < fractions[-1]
    assert fractions[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in fractions)
    # Fractions must never regress -- a progress bar that jumps backward
    # looks broken to a user watching it.
    assert fractions == sorted(fractions)


class TestDeleteDataset:
    """`delete_dataset` is destructive and irreversible, so its cascade
    correctness across every dependent table plus filesystem artifacts is
    tested in depth -- this is the highest-risk piece of the delete
    feature (the UI confirmation gating is a separate, page-level
    concern)."""

    @pytest.fixture()
    def seeded_dataset_with_everything(
        self, isolated_db, isolated_paths, isolated_mapping_config, tiny_cmms_df
    ):
        """Set up a dataset with: a Dataset row, a saved column mapping, a
        processed parquet file, a feasibility report, a trained+activated
        model (via train_and_activate, which also creates ExperimentRun
        rows and model artifact files), and a Prediction +
        PredictionExplanation + Alert simulating a completed "Run
        Prediction". Everything a real dataset could have accumulated."""
        import src.ingestion.mapper as mapper
        from src.feasibility.analyzer import analyze
        from src.feasibility.report_generator import save_report, save_report_json

        dataset_name = "full_ds"
        _seed_dataset(isolated_db, dataset_name)
        mapper.save_mapping(dataset_name, "csv", f"{dataset_name}.csv", {"machine_id": "Asset ID"})

        processed_path = isolated_paths["processed"] / f"{dataset_name}.parquet"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        tiny_cmms_df.to_parquet(processed_path, index=False)

        report = analyze(tiny_cmms_df, source_dataset=f"{dataset_name}.csv")
        save_report(report, output_dir=isolated_paths["reports_feasibility"])
        save_report_json(report, output_dir=isolated_paths["reports_feasibility"])

        train_result = model_service.train_and_activate(
            dataset_name, version="v1", df=tiny_cmms_df
        )

        session = isolated_db()
        try:
            dataset = session.query(Dataset).filter_by(name=dataset_name).one()
            machine = Machine(machine_identifier="M-0", dataset_id=dataset.id)
            session.add(machine)
            session.flush()

            prediction = Prediction(
                machine_id=machine.id,
                model_version_id=train_result.model_version_id,
                failure_probability=0.9,
                predicted_class=1,
                intervention_priority="High Priority",
            )
            session.add(prediction)
            session.flush()

            session.add(
                PredictionExplanation(
                    prediction_id=prediction.id,
                    feature_name="overdue_pms",
                    feature_value=5.0,
                    shap_value=0.4,
                    importance_rank=1,
                )
            )
            session.add(
                Alert(
                    prediction_id=prediction.id,
                    machine_id=machine.id,
                    intervention_priority="High Priority",
                    message="test alert",
                )
            )
            session.add(
                MaintenanceEvent(
                    machine_id=machine.id,
                    event_timestamp=datetime.datetime.utcnow(),
                    dataset_id=dataset.id,
                )
            )
            session.commit()
        finally:
            session.close()

        return dataset_name

    def test_delete_dataset_removes_all_database_rows(
        self, isolated_db, isolated_paths, seeded_dataset_with_everything
    ):
        dataset_name = seeded_dataset_with_everything
        dataset_service.delete_dataset(dataset_name)

        session = isolated_db()
        try:
            assert session.query(Dataset).filter_by(name=dataset_name).count() == 0
            assert session.query(Machine).filter_by(machine_identifier="M-0").count() == 0
            assert session.query(Prediction).count() == 0
            assert session.query(PredictionExplanation).count() == 0
            assert session.query(Alert).count() == 0
            assert session.query(ModelVersion).filter(
                ModelVersion.model_name.in_(
                    ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
                )
            ).count() == 0
            assert session.query(ExperimentRun).filter_by(dataset=dataset_name).count() == 0
            assert session.query(MaintenanceEvent).count() == 0
        finally:
            session.close()

    def test_delete_dataset_removes_filesystem_artifacts(
        self, isolated_paths, seeded_dataset_with_everything
    ):
        dataset_name = seeded_dataset_with_everything
        summary = dataset_service.delete_dataset(dataset_name)

        assert not (isolated_paths["processed"] / f"{dataset_name}.parquet").exists()
        assert not list(isolated_paths["reports_feasibility"].glob(f"{dataset_name}*"))
        assert not (isolated_paths["reports_experiments"] / f"{dataset_name}_v1").exists()
        assert not list(isolated_paths["models_trained"].glob("*_v1.joblib"))
        assert summary.files_failed == []
        assert len(summary.files_deleted) > 0

    def test_delete_dataset_removes_saved_mapping(
        self, isolated_mapping_config, seeded_dataset_with_everything
    ):
        import src.ingestion.mapper as mapper

        dataset_name = seeded_dataset_with_everything
        assert mapper.load_mapping(dataset_name) is not None

        dataset_service.delete_dataset(dataset_name)

        assert mapper.load_mapping(dataset_name) is None

    def test_delete_dataset_reports_correct_counts(
        self, isolated_paths, seeded_dataset_with_everything
    ):
        dataset_name = seeded_dataset_with_everything
        summary = dataset_service.delete_dataset(dataset_name)

        assert summary.dataset_rows_deleted == 1
        assert summary.machines_deleted == 1
        assert summary.predictions_deleted == 1
        assert summary.prediction_explanations_deleted == 1
        assert summary.alerts_deleted == 1
        assert summary.model_versions_deleted == 1
        assert summary.maintenance_events_deleted == 1

    def test_delete_dataset_does_not_touch_other_datasets(
        self, isolated_db, isolated_paths, isolated_mapping_config, seeded_dataset_with_everything, tiny_cmms_df
    ):
        _seed_dataset(isolated_db, "other_ds")
        other_processed = isolated_paths["processed"] / "other_ds.parquet"
        tiny_cmms_df.to_parquet(other_processed, index=False)

        dataset_service.delete_dataset(seeded_dataset_with_everything)

        session = isolated_db()
        try:
            assert session.query(Dataset).filter_by(name="other_ds").count() == 1
        finally:
            session.close()
        assert other_processed.exists()

    def test_delete_dataset_raises_for_unknown_dataset(self, isolated_db, isolated_paths):
        with pytest.raises(ValueError):
            dataset_service.delete_dataset("does_not_exist")


class TestImportAndPrepareProgress:
    def test_import_and_prepare_reports_progress_from_zero_to_one(
        self, isolated_db, isolated_paths, isolated_mapping_config, tmp_path
    ):
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame(
            {
                "Asset ID": ["A1", "A2", "A3"],
                "Overdue PMs": [1, 0, 5],
            }
        ).to_csv(csv_path, index=False)

        calls: list[tuple[str, float]] = []
        dataset_service.import_and_prepare(
            csv_path,
            "progress_test_ds",
            progress_callback=lambda msg, frac: calls.append((msg, frac)),
        )

        assert len(calls) > 1
        fractions = [f for _, f in calls]
        assert fractions[0] > 0.0
        assert fractions[-1] == 1.0
        assert fractions == sorted(fractions)
        assert all(isinstance(msg, str) and msg for msg, _ in calls)

    def test_import_and_prepare_assigns_first_non_null_equipment_category(
        self, isolated_db, isolated_paths, isolated_mapping_config, tmp_path
    ):
        # Regression check for the groupby-based machine registration
        # (replaced an O(n_machines * n_rows) per-machine boolean mask):
        # a machine's category must still resolve to its first non-null
        # value even when some of its rows have a missing category.
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame(
            {
                "Asset ID": ["A1", "A1", "A2"],
                "Equipment Category": [None, "Pump", "Crusher"],
                "Overdue PMs": [1, 1, 0],
            }
        ).to_csv(csv_path, index=False)

        result = dataset_service.import_and_prepare(csv_path, "category_test_ds")

        session = isolated_db()
        try:
            machines = {
                m.machine_identifier: m.equipment_category
                for m in session.query(Machine).filter_by(dataset_id=result.dataset_id)
            }
        finally:
            session.close()

        assert machines["A1"] == "Pump"
        assert machines["A2"] == "Crusher"
