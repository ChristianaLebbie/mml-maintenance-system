"""Select the best trained model version for a dataset from its saved
metadata, write a model_selection_report.md explaining the choice, and
register it as the active ModelVersion in the database (deactivating any
previously active version for that dataset).

Selection policy (section 33): rank by PR-AUC first (appropriate for an
imbalanced minority class), tie-broken by recall. Accuracy alone is never
the deciding factor.

Usage:
    python scripts/select_model.py <dataset_name> <version>

Example:
    python scripts/select_model.py pm_spot_check 1.0.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS  # noqa: E402
from database.database import get_session, init_db  # noqa: E402
from database.models import Dataset, ModelVersion  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.select_model")


def _load_candidates(dataset_name: str, version: str) -> dict[str, dict]:
    metadata_dir = PATHS["models_metadata"]
    candidates = {}
    for path in metadata_dir.glob(f"*_{version}.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("dataset_name") == dataset_name:
            candidates[data["model_type"]] = data
    return candidates


def _select(candidates: dict[str, dict]) -> str:
    def sort_key(name: str):
        metrics = candidates[name]["metrics"]
        return (metrics.get("pr_auc") or 0.0, metrics.get("recall") or 0.0)

    return max(candidates, key=sort_key)


def _write_report(dataset_name: str, version: str, candidates: dict[str, dict], chosen: str) -> Path:
    lines = [
        f"# Model Selection Report: {dataset_name} (version {version})",
        "",
        "## Candidates evaluated",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, data in candidates.items():
        m = data["metrics"]
        marker = " **(selected)**" if name == chosen else ""
        lines.append(
            f"| {name}{marker} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc'] or 0:.3f} | {m['pr_auc'] or 0:.3f} |"
        )

    lines += [
        "",
        "## Selection rationale",
        (
            f"**{chosen}** was selected because it has the highest PR-AUC "
            f"({candidates[chosen]['metrics']['pr_auc']:.3f}) among candidates, "
            "which is the appropriate primary metric given the high-priority "
            "class is the minority class. Recall is used as the tie-breaker. Accuracy "
            "was not used as the deciding factor (section 33)."
        ),
        "",
        "## Known limitation",
        (
            "No LSTM/GRU sequence model was evaluated: the confirmed dataset "
            "is a single point-in-time CMMS snapshot with no per-machine "
            "sequential observations, so a sequence model has no valid input "
            "structure here (see project memory "
            "'project-modeling-design-decisions')."
        ),
    ]

    report_path = PATHS["reports_experiments"] / f"{dataset_name}_{version}" / "model_selection_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    dataset_name, version = sys.argv[1], sys.argv[2]
    init_db()

    candidates = _load_candidates(dataset_name, version)
    if not candidates:
        print(f"No candidate model metadata found for dataset='{dataset_name}' version='{version}'")
        sys.exit(1)

    chosen = _select(candidates)
    chosen_data = candidates[chosen]
    report_path = _write_report(dataset_name, version, candidates, chosen)

    with get_session() as session:
        dataset = session.query(Dataset).filter_by(name=dataset_name).first()
        dataset_id = dataset.id if dataset else None

        session.query(ModelVersion).filter_by(dataset_id=dataset_id, active=True).update(
            {"active": False}
        )
        model_version = ModelVersion(
            version=version,
            model_name=chosen,
            model_type=chosen,
            dataset_id=dataset_id,
            threshold=0.5,
            metrics_json=json.dumps(chosen_data["metrics"]),
            artifact_path=chosen_data["model_artifact_path"],
            preprocessor_path=chosen_data["preprocessor_path"],
            active=True,
        )
        session.add(model_version)

    print(f"Selected model: {chosen}")
    print(f"Report: {report_path}")
    print(f"Registered as active ModelVersion for dataset '{dataset_name}', version '{version}'")


if __name__ == "__main__":
    main()
