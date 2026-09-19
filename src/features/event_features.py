"""Event feature group (e.g. recent-error-count, time-since-maintenance-
event).

Not applicable yet: this requires a populated `maintenance_events` log
keyed by machine and timestamp. The current CMMS export's closest proxies
(last_completed_pm / last_completed_wo) are already used by
temporal_features.py. If a genuine event log is imported later (e.g. from
Work Labor List-derived work orders once decoded per-machine across the
full plant -- see project memory), implement real event-count/recency
features here against that table rather than approximating from the
snapshot fields.
"""

from __future__ import annotations

import pandas as pd


def build_event_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(index=df.index)
