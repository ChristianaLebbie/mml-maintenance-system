"""Train baseline + advanced models on a prepared dataset, evaluate each on
a held-out test split, and serialize every candidate's artifacts (final
model selection happens separately in scripts/select_model.py so the
comparison stays auditable).

Usage:
    python scripts/train_models.py <dataset_name> [version]

Example:
    python scripts/train_models.py pm_spot_check 1.0.0
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS  # noqa: E402
from database.database import get_session, init_db  # noqa: E402
from database.models import ExperimentRun  # noqa: E402
from src.services.dataset_service import load_processed_dataset  # noqa: E402
from src.services.model_service import (  # noqa: E402
    evaluate_all_models,
    prepare_training_data,
    save_model_artifacts,
    train_all_models,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.train_models")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_name = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    init_db()
    df = load_processed_dataset(dataset_name)
    logger.info(f"Loaded processed dataset '{dataset_name}': {len(df)} rows")

    prepared = prepare_training_data(df)
    models = train_all_models(prepared)
    evaluations = evaluate_all_models(models, prepared)

    print(f"\n{'Model':22s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s} {'ROC-AUC':>8s} {'PR-AUC':>7s}")
    for name, result in evaluations.items():
        print(
            f"{name:22s} {result.accuracy:9.3f} {result.precision:10.3f} "
            f"{result.recall:8.3f} {result.f1:6.3f} {result.roc_auc or 0:8.3f} {result.pr_auc or 0:7.3f}"
        )

    experiments_dir = PATHS["reports_experiments"] / f"{dataset_name}_{version}"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        for name, model in models.items():
            evaluation = evaluations[name]
            paths = save_model_artifacts(name, model, prepared, evaluation, dataset_name, version)
            session.add(
                ExperimentRun(
                    experiment_id=f"{dataset_name}_{name}_{version}",
                    dataset=dataset_name,
                    model=name,
                    feature_group="raw+temporal",
                    validation_strategy="stratified_random",
                    metrics_json=json.dumps(evaluation.__dict__),
                    parameters_json=json.dumps({"version": version}),
                )
            )
            (experiments_dir / f"{name}_metrics.json").write_text(
                json.dumps(evaluation.__dict__, indent=2), encoding="utf-8"
            )
            logger.info(f"Saved {name} artifacts: {paths['model']}")

    print(f"\nExperiment artifacts saved under: {experiments_dir}")
    print(f"Model artifacts saved under: {PATHS['models_trained']}")


if __name__ == "__main__":
    main()
