"""Phase 8 tests: feature groups and the orchestrating feature pipeline.

Verifies no cross-machine leakage (each row's features are computed only
from that row's own values -- there is no groupby/rolling operation that
could pull in another machine's data) and that lag/event/frequency groups
stay honest no-ops for this cross-sectional dataset rather than fabricating
signal.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.event_features import build_event_features
from src.features.feature_pipeline import (
    build_features,
    build_numeric_features,
    fit_categorical_encoder,
)
from src.features.frequency_features import build_frequency_features
from src.features.lag_features import build_lag_features
from src.features.raw_features import build_extra_numeric_features, build_raw_features
from src.features.statistical_features import build_statistical_features
from src.features.temporal_features import build_temporal_features, build_trend_features


@pytest.fixture()
def standardized_df():
    return pd.DataFrame(
        {
            "machine_id": ["A", "B", "C"],
            "equipment_category": ["Crusher", "Pump", "Crusher"],
            "manufacturer": ["KSB", None, "KSB"],
            "criticality": [None, None, None],
            "overdue_pms": [2, 0, 5],
            "overdue_wos": [1, 0, 2],
            "total_completed_pms": [30, 10, 0],
            "total_completed_wos": [2, 0, 1],
            "downtime_minutes": [120.0, 0.0, 500.0],
            "total_cost": [0.0, 0.0, 0.0],
            "last_completed_pm": pd.to_datetime(["2025-06-04", "2025-06-10", None]),
            "last_completed_wo": pd.to_datetime([None, None, "2025-05-08"]),
        }
    )


def test_build_raw_features_selects_known_columns(standardized_df):
    features = build_raw_features(standardized_df)
    assert "overdue_pms" in features.columns
    assert "machine_id" not in features.columns


def test_build_statistical_features_ratios(standardized_df):
    features = build_statistical_features(standardized_df)
    # machine C: overdue_pms=5, total_completed_pms=0 -> ratio = 5/1 = 5
    assert features["pm_overdue_ratio"].iloc[2] == 5.0
    # machine B: no overdue, some completed -> ratio close to 0
    assert features["pm_overdue_ratio"].iloc[1] == 0.0


def test_build_temporal_features_recency(standardized_df):
    features = build_temporal_features(standardized_df)
    assert "days_since_last_completed_pm" in features.columns
    # machine A's last PM (2025-06-04) is earlier than machine B's
    # (2025-06-10), so more days should have passed for A given the same
    # reference date (defaults to the max date present, 2025-06-10).
    assert features["days_since_last_completed_pm"].iloc[0] > features["days_since_last_completed_pm"].iloc[1]


def test_temporal_features_respects_explicit_reference_date(standardized_df):
    import datetime

    ref = datetime.datetime(2025, 7, 1)
    features = build_temporal_features(standardized_df, reference_date=ref)
    assert features["days_since_last_completed_pm"].iloc[0] == 27  # 2025-06-04 -> 2025-07-01


def test_build_extra_numeric_features_picks_up_numeric_extras(standardized_df):
    df = standardized_df.copy()
    df["extra__shift_utilization_hours"] = [7.5, 2.0, 8.1]
    df["extra__site_name"] = ["North", "South", "North"]  # non-numeric, must be excluded

    features = build_extra_numeric_features(df)

    assert list(features.columns) == ["extra__shift_utilization_hours"]
    assert list(features["extra__shift_utilization_hours"]) == [7.5, 2.0, 8.1]


def test_build_extra_numeric_features_empty_when_no_extras(standardized_df):
    assert build_extra_numeric_features(standardized_df).empty


def test_lag_event_frequency_groups_are_honest_no_ops(standardized_df):
    # standardized_df has no `timestamp` column (it's the CMMS snapshot
    # shape) -- lag/trend features must stay no-ops for it.
    assert build_lag_features(standardized_df).empty
    assert build_event_features(standardized_df).empty
    assert build_frequency_features(standardized_df).empty
    assert build_trend_features(standardized_df).empty


@pytest.fixture()
def time_series_df():
    # Two machines, 5 hourly readings each -- enough to exercise lag(1),
    # rolling(3), diff(1) without needing the full synthetic dataset.
    return pd.DataFrame(
        {
            "machine_id": ["A"] * 5 + ["B"] * 5,
            "timestamp": pd.concat(
                [pd.Series(pd.date_range("2025-01-01", periods=5, freq="h"))] * 2, ignore_index=True
            ),
            "vibration": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def test_build_lag_features_shifts_within_machine_only(time_series_df):
    features = build_lag_features(time_series_df, sensor_columns=["vibration"], lags=[1], rolling_windows=[])
    combined = time_series_df.join(features)
    machine_a = combined[combined["machine_id"] == "A"].reset_index(drop=True)
    machine_b = combined[combined["machine_id"] == "B"].reset_index(drop=True)

    assert pd.isna(machine_a["vibration_lag_1"].iloc[0])  # no prior observation
    assert machine_a["vibration_lag_1"].iloc[1] == 1.0
    assert pd.isna(machine_b["vibration_lag_1"].iloc[0])  # must not see machine A's last value


def test_build_lag_features_rolling_mean(time_series_df):
    features = build_lag_features(time_series_df, sensor_columns=["vibration"], lags=[], rolling_windows=[3])
    combined = time_series_df.join(features)
    machine_a = combined[combined["machine_id"] == "A"].reset_index(drop=True)
    # rolling mean of [1,2,3] at index 2 (window=3, min_periods=1)
    assert machine_a["vibration_rolling_mean_3"].iloc[2] == pytest.approx(2.0)


def test_build_trend_features_diff_and_pct_change(time_series_df):
    features = build_trend_features(time_series_df, sensor_columns=["vibration"])
    combined = time_series_df.join(features)
    machine_a = combined[combined["machine_id"] == "A"].reset_index(drop=True)
    assert pd.isna(machine_a["vibration_diff_1"].iloc[0])
    assert machine_a["vibration_diff_1"].iloc[1] == pytest.approx(1.0)
    assert machine_a["vibration_pct_change_1"].iloc[1] == pytest.approx(1.0)  # 2.0 is +100% over 1.0


def test_build_numeric_features_combines_active_groups(standardized_df):
    features = build_numeric_features(standardized_df, groups=["raw", "statistical"])
    assert "overdue_pms" in features.columns
    assert "pm_overdue_ratio" in features.columns
    assert "days_since_last_completed_pm" not in features.columns  # temporal not requested


def test_build_numeric_features_rejects_unknown_group(standardized_df):
    with pytest.raises(ValueError):
        build_numeric_features(standardized_df, groups=["not_a_real_group"])


def test_categorical_encoder_handles_unseen_category_at_apply_time(standardized_df):
    train = standardized_df.iloc[:2]  # only sees "Crusher" and "Pump"
    test = standardized_df.iloc[2:]  # also "Crusher", nothing new here but verifies no crash

    encoder = fit_categorical_encoder(train, columns=["equipment_category"])
    features = build_features(test, groups=["raw"], categorical_encoder=encoder)
    assert any(col.startswith("equipment_category_") for col in features.columns)


def test_categorical_encoder_ignores_truly_unseen_category_without_crashing():
    train = pd.DataFrame({"equipment_category": ["Crusher", "Pump"]})
    test = pd.DataFrame({"equipment_category": ["NeverSeenBefore"]})

    encoder = fit_categorical_encoder(train, columns=["equipment_category"])
    encoded = build_features(test, groups=[], categorical_encoder=encoder)
    # handle_unknown="ignore" -> all-zero one-hot row, not an exception
    assert (encoded.iloc[0] == 0).all()


def test_build_features_without_encoder_returns_numeric_only(standardized_df):
    features = build_features(standardized_df, groups=["raw"], categorical_encoder=None)
    assert not any(col.startswith("equipment_category_") for col in features.columns)


def test_no_cross_machine_leakage_in_statistical_features():
    # Each row's ratio must depend only on its own values -- shuffling row
    # order must not change any individual row's computed feature values.
    df = pd.DataFrame(
        {
            "overdue_pms": [1, 2, 3],
            "total_completed_pms": [10, 20, 30],
            "overdue_wos": [0, 0, 0],
            "total_completed_wos": [1, 1, 1],
            "downtime_minutes": [0, 0, 0],
        }
    )
    shuffled = df.iloc[[2, 0, 1]].reset_index(drop=True)

    original = build_statistical_features(df)
    from_shuffled = build_statistical_features(shuffled)

    assert original["pm_overdue_ratio"].iloc[0] == from_shuffled["pm_overdue_ratio"].iloc[1]
