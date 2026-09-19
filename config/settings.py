"""Central configuration loader.

Loads config/application.yaml and config/model_config.yaml once and
exposes them as module-level objects, plus resolved absolute paths.
Nothing in the rest of the codebase should read YAML files directly —
import from here instead, so configuration stays in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


APPLICATION_CONFIG: dict[str, Any] = _load_yaml("application.yaml")
MODEL_CONFIG: dict[str, Any] = _load_yaml("model_config.yaml")


def _resolve(relative: str) -> Path:
    return PROJECT_ROOT / relative


# Resolved paths used throughout the app.
PATHS = {
    "raw_excel": _resolve(APPLICATION_CONFIG["data"]["raw_excel_dir"]),
    "raw_csv": _resolve(APPLICATION_CONFIG["data"]["raw_csv_dir"]),
    "raw_azure": _resolve(APPLICATION_CONFIG["data"]["raw_azure_dir"]),
    "raw_phm": _resolve(APPLICATION_CONFIG["data"]["raw_phm_dir"]),
    "interim": _resolve(APPLICATION_CONFIG["data"]["interim_dir"]),
    "processed": _resolve(APPLICATION_CONFIG["data"]["processed_dir"]),
    "uploads": _resolve(APPLICATION_CONFIG["data"]["uploads_dir"]),
    "powerbi_export": _resolve(APPLICATION_CONFIG["data"]["powerbi_export_dir"]),
    "models_trained": _resolve("models/trained"),
    "models_preprocessors": _resolve("models/preprocessors"),
    "models_explainers": _resolve("models/explainers"),
    "models_metadata": _resolve("models/metadata"),
    "reports_feasibility": _resolve("reports/dataset_feasibility"),
    "reports_experiments": _resolve("reports/experiments"),
    "reports_figures": _resolve("reports/figures"),
    "reports_metrics": _resolve("reports/metrics"),
    "reports_system_testing": _resolve("reports/system_testing"),
    "logs": _resolve(APPLICATION_CONFIG["logging"]["dir"]),
}


def get_database_url() -> str:
    """Resolve the database URL from the environment, falling back to config."""
    env_var = APPLICATION_CONFIG["database"]["url_env_var"]
    default = APPLICATION_CONFIG["database"]["default_url"]
    return os.environ.get(env_var, default)


APP_NAME: str = APPLICATION_CONFIG["application"]["name"]
PREDICTION_HORIZON_HOURS: int = APPLICATION_CONFIG["prediction"]["horizon_hours"]
SUPPORTED_EXTENSIONS: list[str] = APPLICATION_CONFIG["data"]["supported_extensions"]
RANDOM_SEED: int = MODEL_CONFIG["random_seed"]
