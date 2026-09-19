"""Train/validation/test split strategy.

The master project design calls for chronological splitting to prevent
time-series leakage. That assumes repeated, time-ordered observations of
the same entities. The confirmed data for this project is a single
point-in-time CMMS snapshot (~1 row per machine -- see dataset feasibility
reports and project memory), so there is no temporal order between rows to
respect, and an arbitrary "chronological" split by row order would be
meaningless. A stratified random split (preserving the intervention-priority
label distribution across splits) is the methodologically correct choice for
this cross-sectional dataset instead.

If a future dataset has genuine repeated per-machine observations over
time, add a chronological split function here and switch to it for that
dataset -- do not silently reuse this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class DataSplit:
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index


def chronological_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DataSplit:
    """For genuine time-series data (repeated per-machine observations
    over time, e.g. the synthetic sensor-telemetry demo dataset): sort by
    time and cut, so training only ever sees earlier observations than
    validation/test. This is what the master project design originally
    called for; it was not used for the real CMMS snapshot data because
    that data has no time order between rows to respect (see
    stratified_split below)."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    ordered = df.sort_values(time_col).index
    n = len(ordered)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return DataSplit(
        train_idx=ordered[:train_end],
        val_idx=ordered[train_end:val_end],
        test_idx=ordered[val_end:],
    )


def stratified_split(
    labels: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> DataSplit:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    train_idx, temp_idx = train_test_split(
        labels.index,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_state,
    )
    remaining_labels = labels.loc[temp_idx]
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val_ratio,
        stratify=remaining_labels,
        random_state=random_state,
    )
    return DataSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def machine_holdout_split(
    labels: pd.Series,
    machine_ids: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> DataSplit:
    """Group-aware split: every row belonging to a given machine_id falls
    entirely within one split (train, val, or test), never spanning two.
    PM Spot Check.xlsx has approximately one row per PM-task per asset
    (see docs/dataset_mapping.md), so the same machine_id can appear on
    multiple rows. stratified_split (above) splits by row, not by
    machine, so a machine could currently appear in both training and the
    held-out test set -- this function exists to test whether that
    matters. Not label-stratified (grouping takes priority over class
    balance), so the resulting positive-rate per split should be checked
    and reported alongside any comparison, not assumed equal to
    stratified_split's -- see scripts/run_advanced_ablation.py.

    `labels` and `machine_ids` must share the same index.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_pos, temp_pos = next(gss1.split(labels, labels, groups=machine_ids))

    temp_labels = labels.iloc[temp_pos]
    temp_groups = machine_ids.iloc[temp_pos]
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val_ratio, random_state=random_state)
    val_rel, test_rel = next(gss2.split(temp_labels, temp_labels, groups=temp_groups))

    return DataSplit(
        train_idx=labels.index[train_pos],
        val_idx=temp_labels.index[val_rel],
        test_idx=temp_labels.index[test_rel],
    )