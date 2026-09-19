"""Export database contents to CSV files under data/exports/powerbi/ for
Power BI to connect to directly (Power BI is not embedded in Streamlit --
section 54). Safe to run with an empty database: each export writes just a
header row rather than failing.

Usage:
    python scripts/export_powerbi_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import PATHS  # noqa: E402
from database.database import get_session, init_db  # noqa: E402
from database.models import Alert, Machine, ModelVersion, Prediction  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.export_powerbi_data")


def export_equipment_health(session, out_dir: Path) -> None:
    rows = (
        session.query(Prediction, Machine.machine_identifier)
        .join(Machine, Prediction.machine_id == Machine.id)
        .all()
    )
    df = pd.DataFrame(
        [
            {
                "machine": machine_id,
                "timestamp": p.prediction_timestamp,
                "probability": p.failure_probability,
                "intervention_priority": p.intervention_priority,
            }
            for p, machine_id in rows
        ]
    )
    df.to_csv(out_dir / "equipment_health.csv", index=False)


def export_predictions(session, out_dir: Path) -> None:
    rows = (
        session.query(Prediction, Machine.machine_identifier)
        .join(Machine, Prediction.machine_id == Machine.id)
        .all()
    )
    df = pd.DataFrame(
        [
            {
                "prediction_id": p.id,
                "machine": machine_id,
                "prediction_timestamp": p.prediction_timestamp,
                "failure_probability": p.failure_probability,
                "predicted_class": p.predicted_class,
                "intervention_priority": p.intervention_priority,
                "model_version_id": p.model_version_id,
                "created_at": p.created_at,
            }
            for p, machine_id in rows
        ]
    )
    df.to_csv(out_dir / "predictions.csv", index=False)


def export_alerts(session, out_dir: Path) -> None:
    rows = (
        session.query(Alert, Machine.machine_identifier)
        .join(Machine, Alert.machine_id == Machine.id)
        .all()
    )
    df = pd.DataFrame(
        [
            {
                "alert_id": a.id,
                "machine": machine_id,
                "timestamp": a.created_at,
                "intervention_priority": a.intervention_priority,
                "status": a.status,
                "message": a.message,
            }
            for a, machine_id in rows
        ]
    )
    df.to_csv(out_dir / "alerts.csv", index=False)


def export_model_performance(out_dir: Path) -> None:
    experiments_root = PATHS["reports_experiments"]
    records = []
    if experiments_root.exists():
        for experiment_dir in experiments_root.iterdir():
            if not experiment_dir.is_dir():
                continue
            for metrics_file in experiment_dir.glob("*_metrics.json"):
                model_name = metrics_file.stem.replace("_metrics", "")
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                    records.append(
                        {
                            "experiment": experiment_dir.name,
                            "model": model_name,
                            "metric": metric_name,
                            "value": metrics.get(metric_name),
                        }
                    )
    pd.DataFrame(records).to_csv(out_dir / "model_performance.csv", index=False)


def export_active_model_metadata(session, out_dir: Path) -> None:
    active = session.query(ModelVersion).filter_by(active=True).all()
    df = pd.DataFrame(
        [
            {
                "model_version_id": m.id,
                "version": m.version,
                "model_type": m.model_type,
                "dataset_id": m.dataset_id,
                "threshold": m.threshold,
                "created_at": m.created_at,
                "active": m.active,
            }
            for m in active
        ]
    )
    df.to_csv(out_dir / "active_models.csv", index=False)


def main() -> None:
    init_db()
    out_dir = PATHS["powerbi_export"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        export_equipment_health(session, out_dir)
        export_predictions(session, out_dir)
        export_alerts(session, out_dir)
        export_active_model_metadata(session, out_dir)

    export_model_performance(out_dir)

    logger.info(f"Power BI exports written to {out_dir}")
    print(f"Power BI exports written to {out_dir}")
    for f in sorted(out_dir.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
