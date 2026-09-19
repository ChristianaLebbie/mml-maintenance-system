"""ORM models for the predictive-maintenance decision-support database.

Field names follow the project's original sensor-telemetry design (e.g.
`failure_probability`, `prediction_horizon`) for schema stability, but the
confirmed primary task is CMMS-based intervention-priority classification
(see README "Primary ML task"), not time-horizon failure prediction:
- `failure_probability` holds the model's intervention-priority probability.
- `prediction_horizon` is nullable and unused unless/until a future dataset
  supports genuine time-horizon prediction.
"""

from __future__ import annotations

import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    imported_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    machine_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_date: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    end_date: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    sampling_interval: Mapped[str | None] = mapped_column(String(100), nullable=True)
    mapping_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    quality_summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="Imported", nullable=False)

    machines: Mapped[list["Machine"]] = relationship(back_populates="dataset")
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="dataset")
    maintenance_events: Mapped[list["MaintenanceEvent"]] = relationship(back_populates="dataset")


class Machine(Base):
    __tablename__ = "machines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_identifier: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    machine_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Nullable: only populated when a dataset's own metadata verifies a
    # category (e.g. "Crusher", "Conveyor") -- never fabricated.
    equipment_category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    age: Mapped[float | None] = mapped_column(Float, nullable=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    dataset: Mapped[Dataset | None] = relationship(back_populates="machines")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="machine")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="machine")
    maintenance_events: Mapped[list["MaintenanceEvent"]] = relationship(back_populates="machine")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    prediction_horizon: Mapped[int | None] = mapped_column(Integer, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    preprocessor_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    dataset: Mapped[Dataset | None] = relationship(back_populates="model_versions")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[int] = mapped_column(ForeignKey("machines.id"), nullable=False)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)
    prediction_timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    intervention_priority: Mapped[str] = mapped_column(String(50), nullable=False)
    prediction_horizon: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    machine: Mapped[Machine] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion] = relationship(back_populates="predictions")
    explanations: Mapped[list["PredictionExplanation"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(back_populates="prediction")


class PredictionExplanation(Base):
    __tablename__ = "prediction_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    shap_value: Mapped[float] = mapped_column(Float, nullable=False)
    importance_rank: Mapped[int] = mapped_column(Integer, nullable=False)

    prediction: Mapped[Prediction] = relationship(back_populates="explanations")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False)
    machine_id: Mapped[int] = mapped_column(ForeignKey("machines.id"), nullable=False)
    intervention_priority: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="New", nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    acknowledged_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    closed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)

    prediction: Mapped[Prediction] = relationship(back_populates="alerts")
    machine: Mapped[Machine] = relationship(back_populates="alerts")


class MaintenanceEvent(Base):
    __tablename__ = "maintenance_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[int] = mapped_column(ForeignKey("machines.id"), nullable=False)
    event_timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    maintenance_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    component: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)

    machine: Mapped[Machine] = relationship(back_populates="maintenance_events")
    dataset: Mapped[Dataset | None] = relationship(back_populates="maintenance_events")


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    dataset: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    feature_group: Mapped[str | None] = mapped_column(String(100), nullable=True)
    validation_strategy: Mapped[str | None] = mapped_column(String(100), nullable=True)
    parameters_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )


class User(Base):
    """A named account for someone allowed to sign in to the app (see
    src/services/auth_service.py). Passwords are never stored in plain
    text -- only a bcrypt hash. `role` is informational (shown in the UI)
    rather than enforced as a real permission system; every signed-in
    user currently has the same access, which is appropriate for a small
    team using one shared decision-support tool."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="Member", nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    last_login_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
