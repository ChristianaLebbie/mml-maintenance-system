"""Render a FeasibilityReport as Markdown and save it under
reports/dataset_feasibility/, for both the thesis write-up and the
Streamlit Dataset Feasibility page."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from config.settings import PATHS
from src.feasibility.analyzer import FeasibilityReport


def to_markdown(report: FeasibilityReport) -> str:
    lines = [
        f"# Dataset Feasibility Report: {report.source_dataset}",
        "",
        "## Dataset Structure",
        f"- Rows: {report.row_count}",
        f"- Columns: {report.column_count}",
        f"- Unique machines: {report.machine_count}",
        f"- Mean records per machine: {report.records_per_machine_mean}",
        f"- Duplicate rows: {report.duplicate_row_count}",
        "",
        "## Temporal Properties",
    ]
    if report.date_fields_present:
        for field_name in report.date_fields_present:
            date_range = report.date_range.get(field_name)
            if date_range:
                lines.append(f"- `{field_name}`: {date_range[0]} to {date_range[1]}")
    else:
        lines.append("- No populated date fields found.")
    lines.append(
        f"- Single point-in-time snapshot: {'Yes' if report.is_single_snapshot else 'No'}"
    )

    lines += ["", "## Data Quality (missingness by column)"]
    for col, pct in sorted(report.missingness_pct.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{col}`: {pct}% missing")

    lines += ["", "## Target Feasibility", f"- Failure label present: {'Yes' if report.has_failure_label else 'No'}"]
    lines += ["", "### Supported modelling options"]
    if report.supported_tasks:
        for task in report.supported_tasks:
            lines.append(f"- {task}")
    else:
        lines.append("- None identified from this dataset alone.")

    lines += ["", "### Limitations"]
    if report.limitations:
        for limitation in report.limitations:
            lines.append(f"- {limitation}")
    else:
        lines.append("- None identified.")

    return "\n".join(lines) + "\n"


def _safe_name(source_dataset: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in source_dataset)


def save_report(report: FeasibilityReport, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or PATHS["reports_feasibility"]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_safe_name(report.source_dataset)}_feasibility.md"
    out_path.write_text(to_markdown(report), encoding="utf-8")
    return out_path


def save_report_json(report: FeasibilityReport, output_dir: Path | None = None) -> Path:
    """Sidecar JSON of the full FeasibilityReport, so the Dataset
    Feasibility page can build KPI cards/charts directly from the report's
    own fields instead of re-parsing the Markdown or recomputing from the
    processed dataset."""
    output_dir = output_dir or PATHS["reports_feasibility"]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_safe_name(report.source_dataset)}_feasibility.json"
    out_path.write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")
    return out_path


def load_report_json(source_dataset: str, output_dir: Path | None = None) -> dict | None:
    output_dir = output_dir or PATHS["reports_feasibility"]
    path = output_dir / f"{_safe_name(source_dataset)}_feasibility.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
