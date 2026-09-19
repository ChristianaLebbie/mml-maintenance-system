"""End-to-end model training/evaluation/selection over a processed CMMS
dataset. Wraps the well-tested pieces in src/labels, src/features, and
src/training into one reusable workflow used by scripts/train_models.py,
scripts/select_model.py, and (eventually) the researcher-facing Streamlit
pages.

Feature/label separation: the model's feature set deliberately excludes
overdue_pms/overdue_wos/downtime_minutes because those are exactly what the
intervention-priority label is built from -- see project memory
"project-modeling-design-decisions" for why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd

from config.settings import APPLICATION_CONFIG, PATHS
from database.database import get_session
from database.models import Dataset, ExperimentRun, ModelVersion
from src.explainability.shap_service import build_explainer, global_importance
from src.features.feature_pipeline import (
    apply_categorical_encoder,
    build_numeric_features,
    fit_categorical_encoder,
)
from src.labels.failure_labels import (
    PriorityThresholds,
    apply_priority_thresholds,
    compute_priority_score,
    fit_priority_thresholds,
    resolve_priority_weights,
)
from src.training.baselines import train_decision_tree, train_logistic_regression
from src.training.evaluation import EvaluationResult, evaluate_classifier
from src.training.random_forest import train_random_forest
from src.training.validation import DataSplit, stratified_split
from src.training.xgboost_model import train_xgboost
from src.utils.logger import get_logger

logger = get_logger("services.model_service")

ProgressCallback = Callable[[str, float], None]


def _report_progress(callback: ProgressCallback | None, message: str, fraction: float) -> None:
    if callback is not None:
        callback(message, fraction)


NUMERIC_FEATURE_GROUPS = ["raw", "temporal", "extra"]

MODEL_TRAINERS = {
    "logistic_regression": train_logistic_regression,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
}


@dataclass
class PreparedTrainingData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    split: DataSplit
    thresholds: PriorityThresholds
    encoder: object
    feature_columns: list[str]


@dataclass
class TrainingRunResult:
    models: dict[str, object]
    evaluations: dict[str, EvaluationResult]
    best_model_name: str
    shap_importance: pd.Series
    prepared: PreparedTrainingData


def prepare_training_data(
    df: pd.DataFrame,
    feature_groups: list[str] | None = None,
    split_fn: Callable[[pd.Series], DataSplit] | None = None,
) -> PreparedTrainingData:
    """`split_fn`, if given, replaces the default stratified_split call
    below -- pass a functools.partial binding whatever extra arguments
    the alternate split needs (e.g. machine_holdout_split's machine_ids),
    so it matches the `labels -> DataSplit` signature stratified_split
    already has. Used by scripts/run_advanced_ablation.py to compare
    split strategies without duplicating this function's feature/label
    construction. Default (None) is byte-for-byte the previous behavior."""
    feature_groups = feature_groups or NUMERIC_FEATURE_GROUPS

    validation_cfg = APPLICATION_CONFIG.get("validation", {})
    if not resolve_priority_weights(df):
        raise ValueError(
            "This dataset has no populated, varying PM/work-order compliance "
            "metrics (overdue counts, downtime, completion history) or other "
            "numeric operational columns -- intervention-priority "
            "classification needs at least one such signal to build a "
            "meaningful priority score. Check the feasibility report's "
            "limitations before training."
        )
    all_scores = compute_priority_score(df)
    provisional_binary = (all_scores > all_scores.median()).astype(int)
    if split_fn is not None:
        split = split_fn(provisional_binary)
    else:
        split = stratified_split(
            provisional_binary,
            train_ratio=validation_cfg.get("train_ratio", 0.70),
            val_ratio=validation_cfg.get("validation_ratio", 0.15),
            test_ratio=validation_cfg.get("test_ratio", 0.15),
        )

    train_scores = compute_priority_score(df.loc[split.train_idx])
    thresholds = fit_priority_thresholds(train_scores, watch_percentile=70, high_percentile=90)

    all_labels = apply_priority_thresholds(all_scores, thresholds)
    y = (all_labels == "High Priority").astype(int)

    # Whichever columns actually fed the priority score (the known CMMS
    # fields, or -- for a differently-shaped dataset -- the extra__
    # fallback columns from resolve_priority_weights) must never also be
    # model features: that would let the model trivially "predict" a label
    # from its own formula inputs rather than learn a genuine relationship.
    label_formula_columns = list(resolve_priority_weights(df).keys())
    numeric = build_numeric_features(df, groups=feature_groups)
    numeric = numeric.drop(columns=label_formula_columns, errors="ignore")

    encoder = fit_categorical_encoder(df.loc[split.train_idx])

    def build_X(idx) -> pd.DataFrame:
        num_part = numeric.loc[idx].fillna(0).reset_index(drop=True)
        cat_part = apply_categorical_encoder(df.loc[idx], encoder).reset_index(drop=True)
        return pd.concat([num_part, cat_part], axis=1)

    X_train, X_val, X_test = build_X(split.train_idx), build_X(split.val_idx), build_X(split.test_idx)

    return PreparedTrainingData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y.loc[split.train_idx].reset_index(drop=True),
        y_val=y.loc[split.val_idx].reset_index(drop=True),
        y_test=y.loc[split.test_idx].reset_index(drop=True),
        split=split,
        thresholds=thresholds,
        encoder=encoder,
        feature_columns=list(X_train.columns),
    )


def train_all_models(
    prepared: PreparedTrainingData, progress_callback: ProgressCallback | None = None
) -> dict[str, object]:
    models = {}
    total = len(MODEL_TRAINERS)
    for i, (name, trainer) in enumerate(MODEL_TRAINERS.items()):
        _report_progress(progress_callback, f"Training {name} ({i + 1}/{total})...", i / total)
        logger.info(f"Training {name}...")
        models[name] = trainer(prepared.X_train, prepared.y_train)
    _report_progress(progress_callback, "Model training complete", 1.0)
    return models


def evaluate_all_models(
    models: dict[str, object], prepared: PreparedTrainingData
) -> dict[str, EvaluationResult]:
    results = {}
    for name, model in models.items():
        proba = model.predict_proba(prepared.X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        results[name] = evaluate_classifier(prepared.y_test.values, preds, proba)
    return results


def select_best_model(evaluations: dict[str, EvaluationResult]) -> str:
    """Rank primarily by PR-AUC (appropriate for an imbalanced minority
    class), tie-broken by recall -- never by accuracy alone (section 33)."""

    def sort_key(name: str):
        result = evaluations[name]
        return (result.pr_auc or 0.0, result.recall)

    return max(evaluations, key=sort_key)


def run_full_training(
    df: pd.DataFrame,
    feature_groups: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrainingRunResult:
    _report_progress(progress_callback, "Preparing features and intervention-priority labels...", 0.05)
    prepared = prepare_training_data(df, feature_groups)

    def _model_progress(message: str, fraction: float) -> None:
        # Model training occupies the 10%-70% band of this function's
        # overall progress.
        _report_progress(progress_callback, message, 0.10 + fraction * 0.60)

    models = train_all_models(prepared, progress_callback=_model_progress)

    _report_progress(progress_callback, "Evaluating candidates...", 0.75)
    evaluations = evaluate_all_models(models, prepared)
    best_name = select_best_model(evaluations)

    rf_model = models.get("random_forest")
    shap_importance = pd.Series(dtype=float)
    if rf_model is not None:
        _report_progress(progress_callback, "Computing SHAP feature importance...", 0.85)
        background = prepared.X_train.sample(min(100, len(prepared.X_train)), random_state=42)
        explainer = build_explainer(rf_model, background=background)
        shap_importance = global_importance(explainer, prepared.X_test)

    _report_progress(progress_callback, "Training run complete", 1.0)
    return TrainingRunResult(
        models=models,
        evaluations=evaluations,
        best_model_name=best_name,
        shap_importance=shap_importance,
        prepared=prepared,
    )


def save_model_artifacts(
    model_name: str,
    model: object,
    prepared: PreparedTrainingData,
    evaluation: EvaluationResult,
    dataset_name: str,
    version: str,
) -> dict[str, Path]:
    trained_dir = PATHS["models_trained"]
    preproc_dir = PATHS["models_preprocessors"]
    metadata_dir = PATHS["models_metadata"]
    for d in (trained_dir, preproc_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_path = trained_dir / f"{model_name}_{version}.joblib"
    encoder_path = preproc_dir / f"encoder_{version}.joblib"
    metadata_path = metadata_dir / f"{model_name}_{version}.json"

    joblib.dump(model, model_path)
    joblib.dump(prepared.encoder, encoder_path)

    metadata = {
        "version": version,
        "model_type": model_name,
        "dataset_name": dataset_name,
        "feature_columns": prepared.feature_columns,
        "watch_threshold": prepared.thresholds.watch_threshold,
        "high_threshold": prepared.thresholds.high_threshold,
        "metrics": evaluation.__dict__,
        "model_artifact_path": str(model_path),
        "preprocessor_path": str(encoder_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"model": model_path, "encoder": encoder_path, "metadata": metadata_path}


@dataclass
class TrainAndActivateResult:
    dataset_name: str
    version: str
    evaluations: dict[str, EvaluationResult]
    best_model_name: str
    model_version_id: int | None
    selection_report_path: Path
    shap_importance: pd.Series


def _write_selection_report(
    dataset_name: str,
    version: str,
    evaluations: dict[str, EvaluationResult],
    chosen: str,
    experiments_dir: Path,
) -> Path:
    """Same rationale/format as scripts/select_model.py's report, built
    from in-memory evaluations rather than re-loaded JSON -- used by the
    one-call `train_and_activate` (e.g. a Streamlit "Train" button)."""
    lines = [
        f"# Model Selection Report: {dataset_name} (version {version})",
        "",
        "## Candidates evaluated",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, result in evaluations.items():
        marker = " **(selected)**" if name == chosen else ""
        lines.append(
            f"| {name}{marker} | {result.accuracy:.3f} | {result.precision:.3f} | "
            f"{result.recall:.3f} | {result.f1:.3f} | {result.roc_auc or 0:.3f} | "
            f"{result.pr_auc or 0:.3f} |"
        )

    lines += [
        "",
        "## Selection rationale",
        (
            f"**{chosen}** was selected because it has the highest PR-AUC "
            f"({evaluations[chosen].pr_auc or 0:.3f}) among candidates, which is "
            "the appropriate primary metric given the high-priority class is "
            "the minority class. Recall is used as the tie-breaker. Accuracy was "
            "not used as the deciding factor (section 33)."
        ),
    ]

    report_path = experiments_dir / "model_selection_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def train_and_activate(
    dataset_name: str,
    version: str,
    feature_groups: list[str] | None = None,
    df: pd.DataFrame | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrainAndActivateResult:
    """One-call training flow for the UI (e.g. a Streamlit "Train Models"
    button): run_full_training -> save every candidate's artifacts ->
    record an ExperimentRun per candidate -> register the selected model
    as the active ModelVersion for this dataset (deactivating any
    previously active version). Equivalent to running
    scripts/train_models.py followed by scripts/select_model.py, in one
    call, using the same underlying service functions.

    `version` should be unique per training run for this dataset --
    ExperimentRun.experiment_id is `{dataset_name}_{model}_{version}` and
    is enforced unique in the database, so re-running with an
    already-used version raises an IntegrityError (mirroring
    scripts/train_models.py's existing behavior; callers -- e.g. the
    Import Dataset page -- should default `version` to a fresh timestamp).
    """
    if df is None:
        from src.services.dataset_service import load_processed_dataset

        df = load_processed_dataset(dataset_name)

    def _training_progress(message: str, fraction: float) -> None:
        # run_full_training occupies the 0%-80% band of this function's
        # overall progress; saving/activating fills the rest.
        _report_progress(progress_callback, message, fraction * 0.80)

    run_result = run_full_training(df, feature_groups, progress_callback=_training_progress)

    experiments_dir = PATHS["reports_experiments"] / f"{dataset_name}_{version}"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, dict[str, Path]] = {}
    model_version_id: int | None = None
    chosen = run_result.best_model_name

    with get_session() as session:
        dataset = session.query(Dataset).filter_by(name=dataset_name).first()
        dataset_id = dataset.id if dataset else None

        model_names = list(run_result.models.items())
        for i, (name, model) in enumerate(model_names):
            _report_progress(
                progress_callback,
                f"Saving {name} artifacts ({i + 1}/{len(model_names)})...",
                0.80 + (i / len(model_names)) * 0.15,
            )
            evaluation = run_result.evaluations[name]
            paths = save_model_artifacts(
                name, model, run_result.prepared, evaluation, dataset_name, version
            )
            artifact_paths[name] = paths
            session.add(
                ExperimentRun(
                    experiment_id=f"{dataset_name}_{name}_{version}",
                    dataset=dataset_name,
                    model=name,
                    feature_group="+".join(feature_groups or NUMERIC_FEATURE_GROUPS),
                    validation_strategy="stratified_random",
                    metrics_json=json.dumps(evaluation.__dict__),
                    parameters_json=json.dumps({"version": version}),
                )
            )
            (experiments_dir / f"{name}_metrics.json").write_text(
                json.dumps(evaluation.__dict__, indent=2), encoding="utf-8"
            )

        chosen_paths = artifact_paths[chosen]
        chosen_eval = run_result.evaluations[chosen]

        session.query(ModelVersion).filter_by(dataset_id=dataset_id, active=True).update(
            {"active": False}
        )
        model_version = ModelVersion(
            version=version,
            model_name=chosen,
            model_type=chosen,
            dataset_id=dataset_id,
            threshold=0.5,
            metrics_json=json.dumps(chosen_eval.__dict__),
            artifact_path=str(chosen_paths["model"]),
            preprocessor_path=str(chosen_paths["encoder"]),
            active=True,
        )
        session.add(model_version)
        session.flush()
        model_version_id = model_version.id

    _report_progress(progress_callback, "Selecting and activating best model...", 0.97)
    selection_report_path = _write_selection_report(
        dataset_name, version, run_result.evaluations, chosen, experiments_dir
    )

    _report_progress(progress_callback, "Done", 1.0)
    logger.info(f"Trained and activated '{chosen}' for dataset '{dataset_name}' v{version}")

    return TrainAndActivateResult(
        dataset_name=dataset_name,
        version=version,
        evaluations=run_result.evaluations,
        best_model_name=chosen,
        model_version_id=model_version_id,
        selection_report_path=selection_report_path,
        shap_importance=run_result.shap_importance,
    )