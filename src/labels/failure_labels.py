"""Intervention-priority label construction.

The confirmed primary task (see README "Primary ML task") is CMMS-based
intervention-priority classification, not failure-within-horizon prediction --
the data has no populated failure/breakdown log (see project memory /
dataset feasibility reports), so a "failure" ground truth is never
fabricated here.

Instead, a composite priority score is derived from real PM/work-order
compliance metrics (overdue PMs, overdue WOs, accumulated downtime), and
priority-level thresholds are *fit* on a training split's own score
distribution and *applied* unchanged to validation/test data -- mirroring
src/preprocessing/scaling.py's fit/apply split so thresholds are never
invented ahead of time (section 32's rule) and never leak test-set
information into the boundaries.

This module also implements `construct_failure_within_horizon_label()`,
the project's ORIGINAL near-term failure-classification label design
(section 16-17), for use with the synthetic sensor-telemetry demonstration
dataset (docs/synthetic_demo_track.md), which has genuine timestamped
failure events. It is never used against the real CMMS export.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"overdue_pms": 1.0, "overdue_wos": 1.0, "downtime_minutes": 1.0}


def _has_signal(series: pd.Series) -> bool:
    """True only if a column carries real, discriminating information --
    not just non-null. By the time this runs, preprocessing may have
    already zero-filled missing numeric values (see
    src/preprocessing/missing_values.py), so a column that was 100%
    missing looks non-null but is constant 0 across every row.
    `.notna().any()` alone can't tell that apart from a genuinely
    populated column, and treating it as real signal produces a priority
    score with zero variance -- every row lands on the same side of the
    priority threshold, which crashes classifier training with a
    single-class-label error. Filling missing values as 0 (matching
    compute_priority_score's own convention) and then checking for >1
    distinct value catches both the all-missing case and a
    genuinely-constant column, while still counting a column with some
    missing and some real values (missing-as-0 vs. a populated value is
    itself a distinction)."""
    if not series.notna().any():
        return False
    return series.fillna(0.0).nunique(dropna=True) > 1


def resolve_priority_weights(df: pd.DataFrame) -> dict[str, float]:
    known = {
        col: weight
        for col, weight in DEFAULT_WEIGHTS.items()
        if col in df.columns and _has_signal(df[col])
    }
    if known:
        return known
    # Generalize to a differently-shaped mining dataset: if none of the
    # known CMMS compliance columns are populated, fall back to any
    # numeric "extra__" columns actually present (see
    # src/ingestion/mapper.py's pass-through), equally weighted. This is
    # still a real, auditable score derived from whatever operational
    # metrics the uploaded data does have -- never a fabricated label.
    extra_numeric = [
        c
        for c in df.columns
        if c.startswith("extra__") and pd.api.types.is_numeric_dtype(df[c]) and _has_signal(df[c])
    ]
    return {c: 1.0 for c in extra_numeric}


def compute_priority_score(
    df: pd.DataFrame, weights: dict[str, float] | None = None
) -> pd.Series:
    """Composite priority score: each configured column is min-max
    normalized to [0, 1] (using this DataFrame's own range) and combined by
    weighted sum. Missing values are treated as 0 (no signal), matching
    src/preprocessing/missing_values.py's fill convention.

    When the known CMMS compliance columns (overdue PMs/WOs, downtime) are
    absent, falls back to any numeric `extra__` columns instead -- see
    `resolve_priority_weights` -- so a differently-shaped mining dataset
    still gets a real priority score rather than an all-zero one. Callers
    that need to know which columns fed the score (e.g. to exclude them
    from model features and avoid circularity) should call
    `resolve_priority_weights(df)` themselves rather than re-deriving it."""
    weights = weights or resolve_priority_weights(df)
    score = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for col, weight in weights.items():
        if col not in df.columns:
            continue
        values = df[col].fillna(0.0).astype(float)
        value_range = values.max() - values.min()
        normalized = (values - values.min()) / value_range if value_range > 0 else values * 0.0
        score += weight * normalized
        total_weight += weight
    if total_weight > 0:
        score /= total_weight
    return score


@dataclass
class PriorityThresholds:
    watch_threshold: float
    high_threshold: float


def fit_priority_thresholds(
    train_scores: pd.Series, watch_percentile: float = 70.0, high_percentile: float = 90.0
) -> PriorityThresholds:
    """Determine thresholds from the training split's own score
    distribution -- never hardcoded, per section 32."""
    if watch_percentile >= high_percentile:
        raise ValueError("watch_percentile must be less than high_percentile")
    watch = float(np.percentile(train_scores, watch_percentile))
    high = float(np.percentile(train_scores, high_percentile))
    return PriorityThresholds(watch_threshold=watch, high_threshold=high)


def apply_priority_thresholds(scores: pd.Series, thresholds: PriorityThresholds) -> pd.Series:
    def label(score: float) -> str:
        if score >= thresholds.high_threshold:
            return "High Priority"
        if score >= thresholds.watch_threshold:
            return "Watch"
        return "Normal"

    return scores.apply(label)


def construct_failure_within_horizon_label(
    df: pd.DataFrame,
    horizon_hours: int,
    group_col: str = "machine_id",
    time_col: str = "timestamp",
    failure_event_col: str = "failure_label",
) -> pd.Series:
    """The ORIGINAL near-term failure-classification label (section 16-17
    of the project design): 1 if a failure event occurs within
    (t, t + horizon_hours] for that machine, else 0. Only meaningful for
    genuine time-series data with real, timestamped failure events -- the
    real CMMS export has neither (see compute_priority_score above for what
    that dataset supports instead). Used for the synthetic sensor-telemetry
    demonstration dataset; see docs/synthetic_demo_track.md.

    Never uses a sensor value recorded after t as a FEATURE -- this
    function only constructs the LABEL, which is deliberately allowed to
    look forward in time (that's what "failure within the next H hours"
    means); src/features/lag_features.py and temporal_features.py never
    look forward.
    """
    ordered = df.sort_values([group_col, time_col]).copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce", format="mixed")
    horizon = np.timedelta64(horizon_hours, "h")

    labels = pd.Series(0, index=ordered.index, dtype=int)
    for _, group in ordered.groupby(group_col, sort=False):
        failure_times = group.loc[group[failure_event_col] == 1, time_col].to_numpy()
        if len(failure_times) == 0:
            continue
        t = group[time_col].to_numpy()[:, None]  # (n_rows, 1)
        f = failure_times[None, :]  # (1, n_failures)
        in_window = ((f > t) & (f <= t + horizon)).any(axis=1)
        labels.loc[group.index[in_window]] = 1

    return labels.reindex(df.index)
