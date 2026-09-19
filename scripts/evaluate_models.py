"""Re-evaluate a saved model artifact against a freshly-prepared test split
and render confusion-matrix/ROC/PR-curve figures. Separate from
train_models.py so evaluation-artifact generation (Phase 16 system
testing) can be re-run without retraining anything.

Usage:
    python scripts/evaluate_models.py <dataset_name> <model_type> <version>

Example:
    python scripts/evaluate_models.py pm_spot_check xgboost 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402

from config.settings import PATHS  # noqa: E402
from database.database import init_db  # noqa: E402
from src.services.dataset_service import load_processed_dataset  # noqa: E402
from src.services.model_service import prepare_training_data  # noqa: E402
from src.training.evaluation import evaluate_classifier  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
import joblib  # noqa: E402

logger = get_logger("scripts.evaluate_models")


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    dataset_name, model_type, version = sys.argv[1], sys.argv[2], sys.argv[3]
    init_db()

    df = load_processed_dataset(dataset_name)
    prepared = prepare_training_data(df)

    model_path = PATHS["models_trained"] / f"{model_type}_{version}.joblib"
    if not model_path.exists():
        print(f"No model artifact found at {model_path}. Run scripts/train_models.py first.")
        sys.exit(1)
    model = joblib.load(model_path)

    proba = model.predict_proba(prepared.X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    result = evaluate_classifier(prepared.y_test.values, preds, proba)

    print(f"Accuracy:  {result.accuracy:.3f}")
    print(f"Precision: {result.precision:.3f}")
    print(f"Recall:    {result.recall:.3f}")
    print(f"F1:        {result.f1:.3f}")
    print(f"ROC-AUC:   {result.roc_auc}")
    print(f"PR-AUC:    {result.pr_auc}")

    figures_dir = PATHS["reports_figures"] / f"{dataset_name}_{model_type}_{version}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(prepared.y_test, preds, ax=ax)
    fig.savefig(figures_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(prepared.y_test, proba, ax=ax)
    fig.savefig(figures_dir / "roc_curve.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(prepared.y_test, proba, ax=ax)
    fig.savefig(figures_dir / "pr_curve.png", bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved evaluation figures to {figures_dir}")
    print(f"\nFigures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
