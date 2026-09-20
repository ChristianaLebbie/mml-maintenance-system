"""Machine-holdout and class-imbalance-strategy ablations -- the two
follow-up comparisons flagged in docs/system_requirements.md (#17),
alongside the existing feature-group ablation in scripts/run_ablation.py.

Part 1 -- Split strategy: compares the production stratified_split
(row-level -- the same machine_id can appear on multiple rows of
PM Spot Check.xlsx, since it's ~one row per PM-task per asset; see
docs/dataset_mapping.md) against a machine-holdout split (every row for a
given machine falls entirely within one split). This checks whether
row-level splitting is inflating reported test performance through a
machine appearing in both training and test.

Part 2 -- Imbalance strategy: compares "no adjustment" against the
production default (class_weight="balanced" for Random Forest; the
auto-computed scale_pos_weight for XGBoost -- see
src/training/xgboost_model.py) on the SAME production split, to quantify
-- not just assert -- why the current default was chosen.

Usage:
    python scripts/run_advanced_ablation.py <dataset_name> [version]
"""

from __future__ import annotations

import datetime
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# These imports must come after the sys.path.insert() above (this is a
# standalone script, not a package -- it needs the project root on
# sys.path before it can import project-local modules), which is why
# they're not at the very top of the file; both checkers are told that's
# intentional here rather than an oversight.
# pylint: disable=wrong-import-position
from config.settings import PATHS  # noqa: E402
from database.database import init_db  # noqa: E402
from src.services.dataset_service import load_processed_dataset  # noqa: E402
from src.services.model_service import (  # noqa: E402
    evaluate_all_models,
    prepare_training_data,
)
from src.training.random_forest import train_random_forest  # noqa: E402
from src.training.validation import machine_holdout_split  # noqa: E402
from src.training.xgboost_model import train_xgboost  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.run_advanced_ablation")


def _class_balance(y) -> dict:
    return {"n": len(y), "positive_rate": round(float(y.mean()), 4)}


def run_split_strategy_ablation(df, out_dir: Path) -> dict:
    """XGBoost only (the selected production model), since this ablation
    is about whether the SPLIT leaks information, not about comparing
    models."""
    results = {}

    prepared_default = prepare_training_data(df)
    model_default = train_xgboost(prepared_default.X_train, prepared_default.y_train)
    eval_default = evaluate_all_models({"xgboost": model_default}, prepared_default)[
        "xgboost"
    ]
    results["stratified_row_level (production default)"] = {
        **eval_default.__dict__,
        "train_balance": _class_balance(prepared_default.y_train),
        "test_balance": _class_balance(prepared_default.y_test),
    }

    holdout_fn = partial(machine_holdout_split, machine_ids=df["machine_id"])
    prepared_holdout = prepare_training_data(df, split_fn=holdout_fn)
    model_holdout = train_xgboost(prepared_holdout.X_train, prepared_holdout.y_train)
    eval_holdout = evaluate_all_models({"xgboost": model_holdout}, prepared_holdout)[
        "xgboost"
    ]
    results["machine_holdout"] = {
        **eval_holdout.__dict__,
        "train_balance": _class_balance(prepared_holdout.y_train),
        "test_balance": _class_balance(prepared_holdout.y_test),
    }

    print("\n=== Split Strategy Ablation (XGBoost) ===")
    print(
        f"{'Split':40s} {'F1':>6s} {'PR-AUC':>7s} {'ROC-AUC':>8s} "
        f"{'Test n':>7s} {'Test pos-rate':>13s}"
    )
    for label, r in results.items():
        print(
            f"{label:40s} {r['f1']:6.3f} {r['pr_auc'] or 0:7.3f} {r['roc_auc'] or 0:8.3f} "
            f"{r['test_balance']['n']:7d} {r['test_balance']['positive_rate']:13.3f}"
        )
    print(
        "\nNote: machine_holdout is not label-stratified, so a different "
        "test positive-rate than the production split is expected -- "
        "compare F1/PR-AUC/ROC-AUC trends, not raw counts."
    )

    out_path = out_dir / "split_strategy_ablation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to: {out_path}")
    return results


def run_imbalance_strategy_ablation(df, out_dir: Path) -> dict:
    """Both Random Forest and XGBoost, on the SAME production
    (stratified) split for both arms of each model -- so the only thing
    that varies is the imbalance-handling strategy, not the data split."""
    prepared = prepare_training_data(df)

    models = {
        "random_forest_no_weighting": train_random_forest(
            prepared.X_train, prepared.y_train, overrides={"class_weight": None}
        ),
        "random_forest_balanced (production default)": train_random_forest(
            prepared.X_train, prepared.y_train
        ),
        "xgboost_no_weighting": train_xgboost(
            prepared.X_train, prepared.y_train, overrides={"scale_pos_weight": 1.0}
        ),
        "xgboost_auto_scale_pos_weight (production default)": train_xgboost(
            prepared.X_train, prepared.y_train
        ),
    }
    evaluations = evaluate_all_models(models, prepared)
    results = {name: ev.__dict__ for name, ev in evaluations.items()}

    print("\n=== Imbalance Strategy Ablation ===")
    print(
        f"{'Strategy':52s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s} {'PR-AUC':>7s}"
    )
    for name, r in results.items():
        print(
            f"{name:52s} {r['precision']:10.3f} {r['recall']:8.3f} "
            f"{r['f1']:6.3f} {r['pr_auc'] or 0:7.3f}"
        )

    out_path = out_dir / "imbalance_strategy_ablation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to: {out_path}")
    return results


def main() -> None:
    """Entry point: run both advanced ablations for a dataset and save
    their results under reports/experiments/."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_name = sys.argv[1]
    version = (
        sys.argv[2]
        if len(sys.argv) > 2
        else datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    )

    init_db()
    df = load_processed_dataset(dataset_name)

    out_dir = (
        PATHS["reports_experiments"] / f"{dataset_name}_{version}_advanced_ablation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_split_strategy_ablation(df, out_dir)
    run_imbalance_strategy_ablation(df, out_dir)

    print(f"\nAll advanced ablation results saved under: {out_dir}")


if __name__ == "__main__":
    main()
