"""Peer-adjusted anomaly / pattern-discovery view (Component II of the
thesis framework: unsupervised pattern discovery and peer-adjusted anomaly
detection -- a separate analytical layer from the Component III
intervention-priority classifier in src/labels, src/features, and
src/training).

These columns are produced by an offline analysis over a Limble CMMS
export (see docs/dataset_mapping.md and
PM_Spot_Check_Analysis_Ready.xlsx's "PROCESS_PLANT" sheet): each asset is
compared against its peers (same equipment category / hierarchy grouping)
to get a peer-adjusted expected overdue-WO count, a residual and global
anomaly score, and three rank orderings, plus a coarse activity-pattern
cluster label. Only datasets that were run through that analysis carry
these columns -- most imports won't, and this module (and the page built
on it) must degrade gracefully when they're absent.

Deliberately kept independent of src/features/raw_features.py and the
extra__ numeric pass-through (src/ingestion/mapper.py) so these columns
are never silently swept into the intervention-priority classifier's
feature set -- see the comment there.
"""

from __future__ import annotations

import pandas as pd

# Coarse structural/text labels -- shown as-is, never coerced to numeric.
CATEGORICAL_FIELDS = ["hierarchy_level", "maintenance_pattern_cluster"]

# A boolean flag -- kept as True/False rather than coerced to 0/1, so it
# stays readable in a display table.
FLAG_FIELDS = ["no_setup_zero_overdue"]

# Peer-adjusted scores and rank orderings. Some of these arrive from Excel
# as a mixed column (a real float/int for most rows, plus a text sentinel
# such as "Not Applicable (area/system node or infrastructure item)" for
# rows the peer-adjustment doesn't apply to) -- pd.to_numeric(errors=
# "coerce") turns that sentinel into a proper NaN instead of leaving the
# whole column as non-numeric object dtype, without ever fabricating a
# value for those rows.
NUMERIC_SCORE_FIELDS = [
    "pm_setup_count",
    "expected_overdue_wos_peer_adjusted",
    "peer_adjusted_residual",
    "global_anomaly_score",
    "rank_raw_overdue",
    "rank_peer_adjusted",
    "rank_global_anomaly",
]

ALL_FIELDS = CATEGORICAL_FIELDS + FLAG_FIELDS + NUMERIC_SCORE_FIELDS


def has_peer_adjusted_data(df: pd.DataFrame) -> bool:
    """True if this (standardized/processed) DataFrame carries at least
    one populated peer-adjusted-analysis column -- i.e. it was imported
    from a source that included the Component II analysis output."""
    return any(field in df.columns and df[field].notna().any() for field in ALL_FIELDS)


def build_peer_adjusted_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready DataFrame of machine_id plus whichever
    peer-adjusted-analysis columns are present in `df` -- never raises for
    a dataset that has none of them (returns just machine_id, or an empty
    frame if machine_id itself is missing); callers should check
    `has_peer_adjusted_data` first to decide whether this view is worth
    showing at all."""
    out = pd.DataFrame({"machine_id": df["machine_id"]}) if "machine_id" in df.columns else pd.DataFrame()

    for field in ALL_FIELDS:
        if field not in df.columns:
            continue
        if field in NUMERIC_SCORE_FIELDS:
            out[field] = pd.to_numeric(df[field], errors="coerce")
        else:
            out[field] = df[field]

    return out
