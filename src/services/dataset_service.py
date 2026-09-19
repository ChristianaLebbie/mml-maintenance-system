"""Dataset import/preparation service: ingest -> map -> preprocess ->
feasibility report -> persist (processed file + database rows). This is
what scripts/prepare_dataset.py and the future Import Dataset Streamlit
page both call, so the two entry points never diverge in behavior.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from sqlalchemy import or_

from config.settings import PATHS
from database.database import get_session
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
from src.feasibility.analyzer import FeasibilityReport, analyze
from src.feasibility.report_generator import save_report, save_report_json
from src.ingestion.base_source import DataSource
from src.ingestion.csv_loader import CSVDataSource
from src.ingestion.excel_loader import ExcelDataSource
from src.ingestion.mapper import (
    apply_mapping,
    delete_mapping,
    load_mapping,
    save_mapping,
    suggest_mapping,
)
from src.preprocessing.pipeline import PreprocessingReport, run_preprocessing
from src.utils.logger import get_logger

logger = get_logger("services.dataset_service")

ProgressCallback = Callable[[str, float], None]


def _report_progress(callback: ProgressCallback | None, message: str, fraction: float) -> None:
    if callback is not None:
        callback(message, fraction)


@dataclass
class PreparedDataset:
    dataset_id: int
    dataset_name: str
    processed_path: Path
    feasibility_report: FeasibilityReport
    feasibility_report_path: Path
    preprocessing_report: PreprocessingReport
    machine_count: int


def _build_source(file_path: Path, sheet_name: str | None) -> DataSource:
    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return ExcelDataSource(file_path, sheet_name=sheet_name)
    if suffix == ".csv":
        return CSVDataSource(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def import_and_prepare(
    file_path: str | Path,
    dataset_name: str,
    sheet_name: str | None = None,
    mapping_override: dict | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PreparedDataset:
    file_path = Path(file_path)
    source = _build_source(file_path, sheet_name)

    _report_progress(progress_callback, "Validating file...", 0.05)
    problems = source.validate()
    if problems:
        raise ValueError(f"Dataset validation failed: {problems}")

    _report_progress(progress_callback, "Loading file...", 0.15)
    df = source.load()
    logger.info(f"Loaded {file_path.name} ({len(df)} rows, {len(df.columns)} columns)")

    _report_progress(progress_callback, "Resolving column mapping...", 0.25)
    if mapping_override is not None:
        mapping = mapping_override
    else:
        saved = load_mapping(dataset_name)
        mapping = saved["mapping"] if saved else suggest_mapping(list(df.columns))

    source_type = "csv" if file_path.suffix.lower() == ".csv" else "excel"
    save_mapping(dataset_name, source_type, file_path.name, mapping)

    _report_progress(progress_callback, "Applying mapping...", 0.35)
    standardized = apply_mapping(df, mapping, source_dataset=file_path.name)
    standardized = standardized.dropna(subset=["machine_id"]).reset_index(drop=True)

    # Resolve each machine's equipment_category from the PRE-preprocessing
    # data: run_preprocessing (below) fills missing categories with the
    # literal string "Unknown" (see src/preprocessing/missing_values.py),
    # so deriving this after preprocessing would let an "Unknown" from one
    # row of a machine shadow a genuine category found on another row of
    # that same machine. Real missingness here means NaN, not yet a
    # sentinel string.
    category_by_machine: pd.Series = (
        standardized.groupby("machine_id")["equipment_category"].apply(
            lambda s: next((v for v in s if pd.notna(v)), None)
        )
        if "equipment_category" in standardized.columns
        else pd.Series(dtype=object)
    )

    _report_progress(
        progress_callback, "Preprocessing (cleaning, deduplication, missing values)...", 0.50
    )
    processed, preprocessing_report = run_preprocessing(standardized)

    _report_progress(progress_callback, "Analyzing dataset feasibility...", 0.65)
    feasibility_report = analyze(processed, source_dataset=file_path.name)
    feasibility_path = save_report(feasibility_report)
    save_report_json(feasibility_report)

    _report_progress(progress_callback, "Saving processed dataset...", 0.80)
    processed_dir = PATHS["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"{dataset_name}.parquet"
    processed.to_parquet(processed_path, index=False)

    _report_progress(progress_callback, "Registering machines in database...", 0.90)
    with get_session() as session:
        dataset = Dataset(
            name=dataset_name,
            source_type=source_type,
            source_filename=file_path.name,
            row_count=len(processed),
            column_count=len(processed.columns),
            machine_count=processed["machine_id"].nunique(),
            status="Processed",
        )
        session.add(dataset)
        session.flush()

        # One row per unique machine_id, using the pre-preprocessing
        # category_by_machine computed above -- a vectorized groupby
        # instead of a per-machine boolean mask (which was
        # O(n_machines * n_rows), a real bottleneck once a dataset has
        # thousands of machines).
        valid = processed.dropna(subset=["machine_id"])

        machine_count = 0
        for machine_id in valid["machine_id"].unique():
            session.add(
                Machine(
                    machine_identifier=str(machine_id),
                    equipment_category=category_by_machine.get(machine_id)
                    if len(category_by_machine)
                    else None,
                    dataset_id=dataset.id,
                )
            )
            machine_count += 1
        session.flush()
        dataset_id = dataset.id

    _report_progress(progress_callback, "Done", 1.0)
    logger.info(
        f"Prepared dataset '{dataset_name}': {len(processed)} rows, {machine_count} machines"
    )

    return PreparedDataset(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        processed_path=processed_path,
        feasibility_report=feasibility_report,
        feasibility_report_path=feasibility_path,
        preprocessing_report=preprocessing_report,
        machine_count=machine_count,
    )


def load_processed_dataset(dataset_name: str) -> pd.DataFrame:
    path = PATHS["processed"] / f"{dataset_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed dataset found at {path}")
    return pd.read_parquet(path)


@dataclass
class DatasetDeletionSummary:
    dataset_name: str
    dataset_rows_deleted: int
    machines_deleted: int
    predictions_deleted: int
    prediction_explanations_deleted: int
    alerts_deleted: int
    model_versions_deleted: int
    experiment_runs_deleted: int
    maintenance_events_deleted: int
    files_deleted: list[str] = field(default_factory=list)
    files_failed: list[str] = field(default_factory=list)


def delete_dataset(dataset_name: str) -> DatasetDeletionSummary:
    """Permanently delete a dataset and everything derived from it:
    database rows (machines, predictions, explanations, alerts, model
    versions, experiment runs, maintenance events) and filesystem
    artifacts (processed Parquet file, feasibility reports, model/
    preprocessor/metadata files, experiment report directories), plus its
    saved column mapping in config/datasets.yaml.

    Irreversible. Callers (the Dataset Feasibility page's "Danger Zone")
    must obtain an explicit, hard-to-misclick confirmation from the user
    before calling this -- this function itself performs no confirmation.

    Deletes in child-to-parent order using bulk `.delete()` calls (which
    bypass ORM relationship cascades), so the order below is what actually
    keeps the database consistent -- not the `cascade=` settings on the
    model relationships.
    """
    files_deleted: list[str] = []
    files_failed: list[str] = []

    def _remove_file(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
                files_deleted.append(str(path))
        except OSError as exc:
            logger.warning(f"Could not delete {path}: {exc}")
            files_failed.append(str(path))

    def _remove_dir(path: Path) -> None:
        try:
            if path.exists():
                shutil.rmtree(path)
                files_deleted.append(str(path))
        except OSError as exc:
            logger.warning(f"Could not delete directory {path}: {exc}")
            files_failed.append(str(path))

    with get_session() as session:
        datasets = session.query(Dataset).filter_by(name=dataset_name).all()
        if not datasets:
            raise ValueError(f"No dataset named '{dataset_name}' found.")

        dataset_ids = [d.id for d in datasets]
        source_filenames = {d.source_filename for d in datasets if d.source_filename}

        machine_ids = [
            row.id for row in session.query(Machine.id).filter(Machine.dataset_id.in_(dataset_ids))
        ]
        model_version_ids = [
            row.id
            for row in session.query(ModelVersion.id).filter(
                ModelVersion.dataset_id.in_(dataset_ids)
            )
        ]

        prediction_ids = [
            row.id
            for row in session.query(Prediction.id).filter(
                or_(
                    Prediction.machine_id.in_(machine_ids),
                    Prediction.model_version_id.in_(model_version_ids),
                )
            )
        ]

        n_explanations = (
            session.query(PredictionExplanation)
            .filter(PredictionExplanation.prediction_id.in_(prediction_ids))
            .delete(synchronize_session=False)
        )
        n_alerts = (
            session.query(Alert)
            .filter(Alert.prediction_id.in_(prediction_ids))
            .delete(synchronize_session=False)
        )
        n_predictions = (
            session.query(Prediction)
            .filter(Prediction.id.in_(prediction_ids))
            .delete(synchronize_session=False)
        )
        n_maintenance_events = (
            session.query(MaintenanceEvent)
            .filter(MaintenanceEvent.dataset_id.in_(dataset_ids))
            .delete(synchronize_session=False)
        )
        n_experiment_runs = (
            session.query(ExperimentRun)
            .filter(ExperimentRun.dataset == dataset_name)
            .delete(synchronize_session=False)
        )
        n_model_versions = (
            session.query(ModelVersion)
            .filter(ModelVersion.id.in_(model_version_ids))
            .delete(synchronize_session=False)
        )
        n_machines = (
            session.query(Machine)
            .filter(Machine.id.in_(machine_ids))
            .delete(synchronize_session=False)
        )
        n_datasets = (
            session.query(Dataset).filter(Dataset.id.in_(dataset_ids)).delete(synchronize_session=False)
        )

    # Filesystem cleanup happens after the DB transaction commits, and is
    # best-effort: a locked/already-missing file is logged, not fatal.
    _remove_file(PATHS["processed"] / f"{dataset_name}.parquet")

    for source_filename in source_filenames:
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in source_filename)
        _remove_file(PATHS["reports_feasibility"] / f"{safe_name}_feasibility.md")
        _remove_file(PATHS["reports_feasibility"] / f"{safe_name}_feasibility.json")

    for experiment_dir in PATHS["reports_experiments"].glob(f"{dataset_name}_*"):
        if experiment_dir.is_dir():
            _remove_dir(experiment_dir)

    # Every candidate model trained for this dataset gets its own
    # artifact + metadata file (see save_model_artifacts), but only the
    # SELECTED candidate is ever registered as a ModelVersion DB row --
    # so the metadata files' own `dataset_name` field (not the
    # ModelVersion table) is the authoritative list of what to remove
    # here. Using ModelVersion rows alone would silently leave the
    # non-selected candidates' .joblib/.json files behind.
    models_metadata_dir = PATHS["models_metadata"]
    if models_metadata_dir.exists():
        for metadata_path in models_metadata_dir.glob("*.json"):
            try:
                content = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning(f"Could not read {metadata_path}: {exc}")
                continue
            if content.get("dataset_name") != dataset_name:
                continue
            if content.get("model_artifact_path"):
                _remove_file(Path(content["model_artifact_path"]))
            if content.get("preprocessor_path"):
                _remove_file(Path(content["preprocessor_path"]))
            _remove_file(metadata_path)

    delete_mapping(dataset_name)

    logger.info(
        f"Deleted dataset '{dataset_name}': {n_datasets} dataset row(s), {n_machines} machines, "
        f"{n_model_versions} model versions, {len(files_deleted)} files"
    )

    return DatasetDeletionSummary(
        dataset_name=dataset_name,
        dataset_rows_deleted=n_datasets,
        machines_deleted=n_machines,
        predictions_deleted=n_predictions,
        prediction_explanations_deleted=n_explanations,
        alerts_deleted=n_alerts,
        model_versions_deleted=n_model_versions,
        experiment_runs_deleted=n_experiment_runs,
        maintenance_events_deleted=n_maintenance_events,
        files_deleted=files_deleted,
        files_failed=files_failed,
    )
