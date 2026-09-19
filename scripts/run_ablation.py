"""Feature-group ablation: compare Random Forest performance using
different numeric feature-group combinations, to see whether recency
(`temporal`) actually adds value on top of raw PM/WO activity counts.

IMPORTANT: the `statistical` feature group (src/features/statistical_features.py)
is deliberately EXCLUDED from this ablation. Its ratio features
(pm_overdue_ratio, wo_overdue_ratio, downtime_per_completed_wo) are derived
directly from overdue_pms/overdue_wos/downtime_minutes -- the same columns
the intervention-priority label is built from (src/labels/failure_labels.py). Including them
would leak label-defining signal back into the model through a different
column name, silently defeating the label/feature separation documented in
project memory "project-modeling-design-decisions". Only `raw` and
`temporal` are safe to ablate for this dataset.

Usage:
    python scripts/run_ablation.py <dataset_name> [version]
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS  # noqa: E402
from database.database import init_db  # noqa: E402
from src.services.dataset_service import load_processed_dataset  # noqa: E402
from src.services.model_service import (  # noqa: E402
    evaluate_all_models,
    prepare_training_data,
)
from src.training.random_forest import train_random_forest  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.run_ablation")

SAFE_GROUP_COMBINATIONS = [
    ["raw"],
    ["temporal"],
    ["raw", "temporal"],
]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_name = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    init_db()
    df = load_processed_dataset(dataset_name)

    results = {}
    for groups in SAFE_GROUP_COMBINATIONS:
        label = "+".join(groups)
        prepared = prepare_training_data(df, feature_groups=groups)
        model = train_random_forest(prepared.X_train, prepared.y_train)
        evaluation = evaluate_all_models({"random_forest": model}, prepared)["random_forest"]
        results[label] = evaluation.__dict__
        logger.info(f"Ablation [{label}]: F1={evaluation.f1:.3f} PR-AUC={evaluation.pr_auc:.3f}")

    print(f"\n{'Feature groups':20s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s} {'ROC-AUC':>8s} {'PR-AUC':>7s}")
    for label, r in results.items():
        print(
            f"{label:20s} {r['accuracy']:9.3f} {r['precision']:10.3f} "
            f"{r['recall']:8.3f} {r['f1']:6.3f} {r['roc_auc'] or 0:8.3f} {r['pr_auc'] or 0:7.3f}"
        )

    out_dir = PATHS["reports_experiments"] / f"{dataset_name}_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAblation results saved to: {out_path}")
    print(
        "\nNote: the 'statistical' feature group was excluded from this "
        "ablation -- see this script's module docstring for why."
    )


if __name__ == "__main__":
    main()
