"""Dataset feasibility analysis over a standardized DataFrame (the output
of src.ingestion.mapper.apply_mapping).

Reports what the data actually contains -- never assumes a target exists.
Per the project's data-integrity rule, a supervised failure/intervention-
priority label is only reported as feasible when the data demonstrably
supports one; the confirmed default for CMMS-style exports (no
failure_label, single point-in-time snapshot) is a cross-sectional
intervention-priority framing, not a fabricated label -- see README
"Primary ML task".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.labels.failure_labels import _has_signal

CMMS_COMPLIANCE_FIELDS = [
    "overdue_pms",
    "overdue_wos",
    "total_completed_pms",
    "total_completed_wos",
    "downtime_minutes",
    "total_cost",
]


@dataclass
class FeasibilityReport:
    source_dataset: str
    row_count: int
    column_count: int
    machine_count: int
    records_per_machine_mean: float
    duplicate_row_count: int
    missingness_pct: dict[str, float]
    date_fields_present: list[str]
    date_range: dict[str, tuple[str, str]]
    is_single_snapshot: bool
    has_failure_label: bool
    supported_tasks: list[str]
    limitations: list[str] = field(default_factory=list)


def analyze(df: pd.DataFrame, source_dataset: str) -> FeasibilityReport:
    row_count = len(df)
    column_count = len(df.columns)

    machine_series = df["machine_id"].dropna() if "machine_id" in df.columns else pd.Series(dtype=object)
    machine_count = int(machine_series.nunique())
    records_per_machine_mean = round(row_count / machine_count, 2) if machine_count else 0.0

    duplicate_row_count = int(df.duplicated().sum())
    missingness_pct = {
        col: round(100 * df[col].isna().mean(), 2) for col in df.columns
    }

    date_fields = [
        c
        for c in ("last_completed_pm", "last_completed_wo")
        if c in df.columns and df[c].notna().any()
    ]
    date_range: dict[str, tuple[str, str]] = {}
    for c in date_fields:
        parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
        parsed = parsed.dropna()
        if not parsed.empty:
            date_range[c] = (str(parsed.min().date()), str(parsed.max().date()))

    # A CMMS PM/WO snapshot has ~one row per machine (the current state of
    # each asset's PM schedule), not repeated observations over time for
    # the same machine -- that is what makes it a single point-in-time
    # cross-section rather than a time series.
    is_single_snapshot = bool(machine_count > 0 and records_per_machine_mean < 1.5)

    has_failure_label = bool(
        "failure_label" in df.columns and df["failure_label"].notna().any()
    )

    supported_tasks: list[str] = []
    limitations: list[str] = []

    if has_failure_label:
        supported_tasks.append("binary_failure_classification")
    else:
        limitations.append(
            "No populated failure_label column -- binary/multi-class failure "
            "classification is not supported without fabricating a label."
        )

    # Require actual variance, not just non-null: preprocessing zero-fills
    # missing numeric columns upstream (src/preprocessing/missing_values.py),
    # so a compliance field that was 100% missing looks "populated" here
    # but is constant 0 -- that produces a zero-variance priority score
    # downstream, not a real signal. Mirrors
    # src/labels/failure_labels.py's resolve_priority_weights/_has_signal.
    compliance_present = [c for c in CMMS_COMPLIANCE_FIELDS if c in df.columns and _has_signal(df[c])]
    extra_numeric_present = [
        c
        for c in df.columns
        if c.startswith("extra__") and pd.api.types.is_numeric_dtype(df[c]) and _has_signal(df[c])
    ]
    has_priority_signal = bool(compliance_present or extra_numeric_present)
    if has_priority_signal and machine_count > 0:
        supported_tasks.append("intervention_priority_classification")
    else:
        limitations.append(
            "No populated PM/work-order compliance metrics (overdue counts, "
            "downtime, completion history) and no other numeric operational "
            "columns -- intervention-priority classification is not "
            "supported for this dataset."
        )

    if is_single_snapshot:
        limitations.append(
            "Data is a single point-in-time snapshot (~1 row per machine) -- "
            "time-series / near-term failure-horizon prediction is not "
            "supported unless multiple dated snapshots of the same machines "
            "are supplied."
        )
    elif date_fields:
        supported_tasks.append("time_series_analysis")

    if machine_count > 1 and has_priority_signal:
        supported_tasks.append("anomaly_detection_across_machines")

    return FeasibilityReport(
        source_dataset=source_dataset,
        row_count=row_count,
        column_count=column_count,
        machine_count=machine_count,
        records_per_machine_mean=records_per_machine_mean,
        duplicate_row_count=duplicate_row_count,
        missingness_pct=missingness_pct,
        date_fields_present=date_fields,
        date_range=date_range,
        is_single_snapshot=is_single_snapshot,
        has_failure_label=has_failure_label,
        supported_tasks=supported_tasks,
        limitations=limitations,
    )
