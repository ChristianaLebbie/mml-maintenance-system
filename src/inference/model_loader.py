"""Loads the active, already-trained model + its preprocessing artifacts
for inference. Training happens only in scripts/train_models.py --
Streamlit and other inference callers only ever load serialized artifacts
here (section 36: separate training from inference)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from database.database import get_session
from database.models import ModelVersion


@dataclass
class LoadedModel:
    model: object
    encoder: object
    metadata: dict
    model_version_id: int
    version: str
    model_type: str


def get_active_model_version(dataset_id: int | None = None) -> ModelVersion | None:
    with get_session() as session:
        query = session.query(ModelVersion).filter_by(active=True)
        if dataset_id is not None:
            query = query.filter_by(dataset_id=dataset_id)
        row = query.order_by(ModelVersion.created_at.desc()).first()
        if row is None:
            return None
        session.expunge(row)
        return row


def load_active_model(dataset_id: int | None = None) -> LoadedModel:
    model_version = get_active_model_version(dataset_id)
    if model_version is None:
        raise RuntimeError(
            "No active model version found. Run scripts/train_models.py and "
            "scripts/select_model.py first."
        )

    model = joblib.load(_artifact_path_for(model_version))
    encoder = joblib.load(_preprocessor_path_for(model_version))

    metadata_path = _metadata_path_for(model_version)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    return LoadedModel(
        model=model,
        encoder=encoder,
        metadata=metadata,
        model_version_id=model_version.id,
        version=model_version.version,
        model_type=model_version.model_type,
    )


def _metadata_path_for(model_version: ModelVersion) -> Path:
    from config.settings import PATHS

    return PATHS["models_metadata"] / f"{model_version.model_type}_{model_version.version}.json"


def _artifact_path_for(model_version: ModelVersion) -> Path:
    # Rebuilt from the current PATHS config + the naming convention
    # save_model_artifacts() uses (src/services/model_service.py), rather
    # than trusting the absolute path stored in the database's
    # `artifact_path` column: that column can hold a path from a
    # different machine or a different checkout location (e.g. a Windows
    # absolute path in a database committed alongside code deployed to a
    # Linux host), which would silently fail to resolve there even though
    # the actual .joblib file the training run produced is sitting right
    # where this convention expects it.
    from config.settings import PATHS

    return PATHS["models_trained"] / f"{model_version.model_type}_{model_version.version}.joblib"


def _preprocessor_path_for(model_version: ModelVersion) -> Path:
    from config.settings import PATHS

    return PATHS["models_preprocessors"] / f"encoder_{model_version.version}.joblib"
