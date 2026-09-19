"""Column mapping: source workbook columns -> the standardized internal
schema (src/ingestion/schemas.py).

Mappings are never assumed -- `suggest_mapping()` proposes candidates from
the columns actually present in an uploaded file (matched against known
aliases), the researcher/engineer confirms or corrects them (e.g. in the
Import Dataset page), and `save_mapping()` persists the confirmed mapping
to config/datasets.yaml so it can be reused for future imports of the same
kind of file.

Field set reflects the confirmed primary task (CMMS-based intervention-priority
classification, see README) -- asset attributes and PM/work-order
compliance metrics -- rather than sensor-telemetry columns.

To accept a differently-shaped mining dataset (a different mine's CMMS
export, or any file with extra operational metrics) without code changes,
`apply_mapping()` never silently drops a source column just because it
doesn't match one of the known canonical fields above: every unmapped
column is carried through under an `extra__<name>` column so it still
appears in previews, feasibility reports, and (if numeric) as a candidate
model feature -- see src/features/raw_features.py:build_extra_numeric_features
and src/labels/failure_labels.py:compute_priority_score's fallback weighting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "datasets.yaml"

# name -> (required, [case-insensitive substring aliases])
INTERNAL_FIELDS: dict[str, dict[str, Any]] = {
    "machine_id": {
        "required": True,
        "aliases": ["asset id", "asset name", "machine id", "equipment id", "asset_id"],
    },
    "equipment_category": {
        "required": False,
        "aliases": ["equipment category", "category", "asset type"],
    },
    "manufacturer": {"required": False, "aliases": ["manufacturer", "make"]},
    "criticality": {
        "required": False,
        "aliases": ["criticality", "asset criticality index", "criticality classification"],
    },
    "overdue_pms": {"required": False, "aliases": ["overdue pms"]},
    "overdue_wos": {"required": False, "aliases": ["overdue wos"]},
    "total_completed_pms": {"required": False, "aliases": ["total completed pms"]},
    "total_completed_wos": {"required": False, "aliases": ["total completed wos"]},
    "downtime_minutes": {"required": False, "aliases": ["downtime", "total time spent"]},
    "total_cost": {"required": False, "aliases": ["total cost", "cost per work order"]},
    "last_completed_pm": {"required": False, "aliases": ["last completed pm"]},
    "last_completed_wo": {"required": False, "aliases": ["last completed wo"]},
    "failure_label": {"required": False, "aliases": ["failure", "breakdown"]},
    # Sensor-telemetry fields: only populated by the synthetic demonstration
    # track (see docs/synthetic_demo_track.md) -- the real CMMS export has
    # no columns matching these, so they stay None/missing for it.
    "timestamp": {"required": False, "aliases": ["timestamp", "reading time", "reading_time"]},
    "voltage": {"required": False, "aliases": ["voltage"]},
    "rotation": {"required": False, "aliases": ["rotation", "rpm"]},
    "pressure": {"required": False, "aliases": ["pressure"]},
    "vibration": {"required": False, "aliases": ["vibration"]},
    # Peer-adjusted anomaly / pattern-discovery columns: produced by an
    # offline analysis notebook over a Limble CMMS export (see
    # PM_Spot_Check_Analysis_Ready.xlsx, "PROCESS_PLANT" sheet) and only
    # present on datasets that were run through that analysis first --
    # they stay None/missing for every other import. Named here (rather
    # than left as generic extra__ pass-through columns) so they get
    # clean labels in previews/feasibility reports and can be shown on
    # the Peer-Adjusted Analysis page (src/analysis/peer_adjusted.py).
    # Deliberately NOT added to src/features/raw_features.py's
    # RAW_FEATURE_COLUMNS or picked up by the extra__ numeric-feature
    # pass-through: this is Component II (unsupervised peer-adjusted
    # anomaly detection) output, a separate analytical layer from the
    # Component III intervention-priority classifier, and mixing the two
    # would blur that distinction and silently change the trained
    # model's feature set.
    "hierarchy_level": {"required": False, "aliases": ["hierarchy level"]},
    "pm_setup_count": {"required": False, "aliases": ["pm setup count"], "numeric": True},
    "no_setup_zero_overdue": {"required": False, "aliases": ["no setup zero overdue"]},
    "expected_overdue_wos_peer_adjusted": {
        "required": False,
        "aliases": ["expected overdue wos peer adjusted"],
        "numeric": True,
    },
    "peer_adjusted_residual": {
        "required": False,
        "aliases": ["peer adjusted residual"],
        "numeric": True,
    },
    "global_anomaly_score": {
        "required": False,
        "aliases": ["global anomaly score"],
        "numeric": True,
    },
    "rank_raw_overdue": {"required": False, "aliases": ["rank raw overdue"], "numeric": True},
    "rank_peer_adjusted": {
        "required": False,
        "aliases": ["rank peer adjusted"],
        "numeric": True,
    },
    "rank_global_anomaly": {
        "required": False,
        "aliases": ["rank global anomaly"],
        "numeric": True,
    },
    "maintenance_pattern_cluster": {
        "required": False,
        "aliases": ["maintenance pattern cluster"],
    },
}


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Best-effort auto-suggestion: for each internal field, pick the source
    column whose name contains one of its aliases. None when nothing matches
    -- the caller (UI) must let the user fill it in manually."""
    normalized = {c: c.strip().lower().replace("_", " ") for c in columns}
    mapping: dict[str, str | None] = {}
    used: set[str] = set()

    for field, spec in INTERNAL_FIELDS.items():
        match = None
        for col, norm in normalized.items():
            if col in used:
                continue
            if any(alias in norm for alias in spec["aliases"]):
                match = col
                break
        mapping[field] = match
        if match:
            used.add(match)

    return mapping


def validate_mapping(mapping: dict[str, str | None], columns: list[str]) -> list[str]:
    problems: list[str] = []
    for field, spec in INTERNAL_FIELDS.items():
        source_col = mapping.get(field)
        if source_col is None:
            if spec["required"]:
                problems.append(f"Required field '{field}' is not mapped to any column.")
            continue
        if source_col not in columns:
            problems.append(
                f"Field '{field}' is mapped to '{source_col}', which is not a column "
                "in this dataset."
            )
    return problems


def _sanitize_extra_name(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in str(name).strip().lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def _make_arrow_safe(series: pd.Series) -> pd.Series:
    """An `extra__` pass-through column can legitimately mix Python types
    within one Excel column -- e.g. a "Serial Number"/"Model Number" field
    where most values were entered as text but a few look like bare
    numbers, so pandas reads the whole column as `object` with a mix of
    `str` and `int`/`float` cells. That mix crashes outright when the
    processed dataset is later written to Parquet (pyarrow infers a single
    type per column and can't reconcile str with int/float). Any
    genuinely mixed-type object column is therefore normalized to a
    single string dtype here -- real missing values stay NaN rather than
    becoming the literal text "nan" -- which only affects free-text/ID
    pass-through columns: `build_extra_numeric_features`
    (src/features/raw_features.py) already excludes non-numeric-dtype
    extras from model features, so a column that was excluded before this
    normalization stays excluded after it."""
    if series.dtype != object:
        return series
    non_null_types = {type(v) for v in series.dropna()}
    if len(non_null_types) <= 1:
        return series
    return series.map(lambda v: v if pd.isna(v) else str(v))


def apply_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    source_dataset: str,
    include_extra: bool = True,
) -> pd.DataFrame:
    """Return a new DataFrame with canonical column names, per `mapping`.
    Unmapped optional fields become all-null columns; `source_dataset` is
    stamped onto every row.

    When `include_extra` (default True), every source column NOT used by
    the mapping is also carried through as `extra__<sanitized_name>` --
    this is what lets a differently-shaped mining dataset be accepted
    without losing data or requiring a code change for every new column a
    new source happens to have.
    """
    problems = validate_mapping(mapping, list(df.columns))
    required_missing = [p for p in problems if "not mapped" in p]
    if required_missing:
        raise ValueError("; ".join(required_missing))

    standardized = pd.DataFrame(index=df.index)
    for field, spec in INTERNAL_FIELDS.items():
        source_col = mapping.get(field)
        column = df[source_col] if source_col else None
        # A handful of fields (the peer-adjusted-analysis columns) can
        # arrive from Excel as a mix of real numbers and a text sentinel
        # (e.g. "Not Applicable (...)") for rows the analysis doesn't
        # apply to. Coercing here -- not just at display time -- matters:
        # a mixed numeric/text object column fails outright when the
        # processed dataset is later written to Parquet
        # (src/services/dataset_service.py), so this must happen before
        # that, not just in the display layer (src/analysis/peer_adjusted.py).
        # errors="coerce" turns the sentinel into NaN rather than
        # fabricating a value for it.
        if spec.get("numeric") and column is not None:
            column = pd.to_numeric(column, errors="coerce")
        standardized[field] = column
    standardized["source_dataset"] = source_dataset

    if include_extra:
        mapped_columns = {v for v in mapping.values() if v}
        used_extra_names: set[str] = set()
        for col in df.columns:
            if col in mapped_columns:
                continue
            extra_name = f"extra__{_sanitize_extra_name(col)}"
            # Guard against two source columns sanitizing to the same name
            # (e.g. "Cost (USD)" and "Cost USD") -- never silently overwrite.
            base_name = extra_name
            suffix = 2
            while extra_name in used_extra_names:
                extra_name = f"{base_name}_{suffix}"
                suffix += 1
            used_extra_names.add(extra_name)
            standardized[extra_name] = _make_arrow_safe(df[col])

    return standardized


def save_mapping(
    dataset_name: str, source_type: str, source_filename: str, mapping: dict[str, str | None]
) -> None:
    config = _load_config()
    config.setdefault("datasets", {})[dataset_name] = {
        "source_type": source_type,
        "source_filename": source_filename,
        "mapping": mapping,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def load_mapping(dataset_name: str) -> dict[str, Any] | None:
    config = _load_config()
    return config.get("datasets", {}).get(dataset_name)


def delete_mapping(dataset_name: str) -> bool:
    """Remove a dataset's saved mapping from config/datasets.yaml, if
    present -- used when a dataset is deleted (see
    src.services.dataset_service.delete_dataset). Returns True if a
    mapping was actually removed."""
    config = _load_config()
    datasets = config.get("datasets", {})
    if dataset_name not in datasets:
        return False
    del datasets[dataset_name]
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return True


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"datasets": {}}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"datasets": {}}
