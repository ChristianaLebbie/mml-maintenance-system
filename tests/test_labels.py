"""Phase 7 tests: intervention-priority score computation and threshold
fit/apply, verifying thresholds come only from the training split (no
leakage from a separately-scored test/validation set)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.labels.failure_labels import (
    PriorityThresholds,
    apply_priority_thresholds,
    compute_priority_score,
    construct_failure_within_horizon_label,
    fit_priority_thresholds,
    resolve_priority_weights,
)


def test_compute_priority_score_higher_for_worse_compliance():
    df = pd.DataFrame(
        {
            "overdue_pms": [0, 5],
            "overdue_wos": [0, 3],
            "downtime_minutes": [0.0, 500.0],
        }
    )
    scores = compute_priority_score(df)
    assert scores.iloc[1] > scores.iloc[0]


def test_compute_priority_score_treats_missing_as_zero_signal():
    df = pd.DataFrame({"overdue_pms": [None, 5], "overdue_wos": [0, 0], "downtime_minutes": [0, 0]})
    scores = compute_priority_score(df)
    assert scores.iloc[0] < scores.iloc[1]


def test_compute_priority_score_constant_column_does_not_error():
    df = pd.DataFrame({"overdue_pms": [3, 3, 3], "overdue_wos": [0, 0, 0], "downtime_minutes": [0, 0, 0]})
    scores = compute_priority_score(df)
    assert (scores == 0.0).all()


def test_resolve_priority_weights_prefers_known_cmms_fields():
    df = pd.DataFrame(
        {
            "overdue_pms": [1, 2],
            "extra__spare_parts_count": [5, 6],
        }
    )
    weights = resolve_priority_weights(df)
    assert "overdue_pms" in weights
    assert "extra__spare_parts_count" not in weights


def test_resolve_priority_weights_falls_back_to_extra_numeric_columns():
    # A differently-shaped mining dataset with no known CMMS field names.
    df = pd.DataFrame(
        {
            "machine_id": ["A", "B"],
            "extra__shift_utilization_hours": [7.5, 2.0],
            "extra__spare_parts_count": [1, 9],
        }
    )
    weights = resolve_priority_weights(df)
    assert set(weights) == {"extra__shift_utilization_hours", "extra__spare_parts_count"}


def test_resolve_priority_weights_empty_when_no_numeric_signal_at_all():
    df = pd.DataFrame({"machine_id": ["A", "B"], "notes": ["ok", "ok"]})
    assert resolve_priority_weights(df) == {}


def test_compute_priority_score_generalizes_to_extra_columns():
    df = pd.DataFrame(
        {
            "extra__shift_utilization_hours": [1.0, 10.0],
            "extra__spare_parts_count": [0, 0],
        }
    )
    scores = compute_priority_score(df)
    assert scores.iloc[1] > scores.iloc[0]


def test_fit_priority_thresholds_rejects_invalid_percentile_order():
    with pytest.raises(ValueError):
        fit_priority_thresholds(pd.Series([1, 2, 3]), watch_percentile=90, high_percentile=70)


def test_fit_priority_thresholds_from_training_distribution():
    train_scores = pd.Series(range(100)) / 100.0  # 0.00 .. 0.99
    thresholds = fit_priority_thresholds(train_scores, watch_percentile=70, high_percentile=90)
    assert isinstance(thresholds, PriorityThresholds)
    assert 0.65 < thresholds.watch_threshold < 0.75
    assert 0.85 < thresholds.high_threshold < 0.95


def test_apply_priority_thresholds_labels_correctly():
    thresholds = PriorityThresholds(watch_threshold=0.5, high_threshold=0.8)
    scores = pd.Series([0.1, 0.6, 0.9])
    labels = apply_priority_thresholds(scores, thresholds)
    assert list(labels) == ["Normal", "Watch", "High Priority"]


def test_construct_failure_within_horizon_label_flags_window_before_failure():
    df = pd.DataFrame(
        {
            "machine_id": ["A"] * 6,
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="h"),
            "failure_label": [0, 0, 0, 0, 1, 0],
        }
    )
    labels = construct_failure_within_horizon_label(df, horizon_hours=2)
    # Failure at index 4 (hour 4). Rows at hour 2 and 3 fall within (t, t+2h].
    assert list(labels) == [0, 0, 1, 1, 0, 0]


def test_construct_failure_within_horizon_label_excludes_the_failure_row_itself():
    df = pd.DataFrame(
        {
            "machine_id": ["A"] * 3,
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "failure_label": [0, 1, 0],
        }
    )
    labels = construct_failure_within_horizon_label(df, horizon_hours=1)
    # The row AT the failure (index 1) looks forward for its own window and
    # finds no failure after itself -- (t, t+1h] excludes t.
    assert labels.iloc[1] == 0


def test_construct_failure_within_horizon_label_never_crosses_machines():
    df = pd.DataFrame(
        {
            "machine_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 00:00", "2025-01-01 01:00"]
            ),
            "failure_label": [0, 0, 0, 1],
        }
    )
    labels = construct_failure_within_horizon_label(df, horizon_hours=2)
    # Machine A has no failure at all -- must stay 0 regardless of B's failure.
    assert labels.iloc[0] == 0
    assert labels.iloc[1] == 0


def test_construct_failure_within_horizon_label_no_failures_returns_all_zero():
    df = pd.DataFrame(
        {
            "machine_id": ["A"] * 4,
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="h"),
            "failure_label": [0, 0, 0, 0],
        }
    )
    labels = construct_failure_within_horizon_label(df, horizon_hours=2)
    assert (labels == 0).all()


def test_thresholds_fit_on_train_applied_unchanged_to_test():
    train_scores = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    thresholds = fit_priority_thresholds(train_scores, watch_percentile=70, high_percentile=90)

    # A test-set score of 0.95 must be judged against train-derived
    # thresholds, not against the test set's own (possibly very different)
    # distribution.
    test_scores = pd.Series([0.95])
    labels = apply_priority_thresholds(test_scores, thresholds)
    assert labels.iloc[0] == "High Priority"
