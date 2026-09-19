"""Tests for src/analysis/peer_adjusted.py: the Component II (peer-adjusted
anomaly / pattern-discovery) display view, kept separate from the
Component III intervention-priority classifier's feature set."""

from __future__ import annotations

import pandas as pd

from src.analysis.peer_adjusted import (
    ALL_FIELDS,
    build_peer_adjusted_view,
    has_peer_adjusted_data,
)
from src.features.feature_pipeline import build_numeric_features
from src.features.raw_features import RAW_FEATURE_COLUMNS, build_extra_numeric_features


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": ["A", "B", "C"],
            "overdue_pms": [1, 0, 5],
            "hierarchy_level": [
                "Individual equipment item",
                "Individual equipment item",
                "Area/System node",
            ],
            "pm_setup_count": [2, 1, 0],
            "no_setup_zero_overdue": [False, False, True],
            "expected_overdue_wos_peer_adjusted": [
                0.034906,
                0.244866,
                "Not Applicable (area/system node or infrastructure item)",
            ],
            "peer_adjusted_residual": [
                -0.034906,
                -0.244866,
                "Not Applicable (area/system node or infrastructure item)",
            ],
            "global_anomaly_score": [
                0.323757,
                0.309241,
                "Not Applicable (area/system node or infrastructure item)",
            ],
            "rank_raw_overdue": [451, 451, 451],
            "rank_peer_adjusted": [
                1970,
                4069,
                "Not Applicable (area/system node or infrastructure item)",
            ],
            "rank_global_anomaly": [
                1529,
                3829,
                "Not Applicable (area/system node or infrastructure item)",
            ],
            "maintenance_pattern_cluster": [
                "Moderate activity",
                "Minimal activity",
                "Not clustered (equipment category has <50 observations, or not "
                "individual process-plant equipment)",
            ],
        }
    )


def test_has_peer_adjusted_data_true_when_any_field_populated():
    assert has_peer_adjusted_data(_sample_df()) is True


def test_has_peer_adjusted_data_false_for_ordinary_dataset():
    df = pd.DataFrame({"machine_id": ["A", "B"], "overdue_pms": [1, 0]})
    assert has_peer_adjusted_data(df) is False


def test_build_peer_adjusted_view_includes_machine_id_and_all_present_fields():
    view = build_peer_adjusted_view(_sample_df())
    assert list(view["machine_id"]) == ["A", "B", "C"]
    for field in ALL_FIELDS:
        assert field in view.columns


def test_build_peer_adjusted_view_coerces_sentinel_string_to_nan():
    view = build_peer_adjusted_view(_sample_df())
    assert view["global_anomaly_score"].iloc[2] != view["global_anomaly_score"].iloc[2]  # NaN
    assert view["global_anomaly_score"].iloc[0] == 0.323757
    assert pd.api.types.is_numeric_dtype(view["global_anomaly_score"])
    assert pd.api.types.is_numeric_dtype(view["rank_peer_adjusted"])


def test_build_peer_adjusted_view_keeps_categorical_and_flag_fields_as_is():
    view = build_peer_adjusted_view(_sample_df())
    assert list(view["hierarchy_level"]) == [
        "Individual equipment item",
        "Individual equipment item",
        "Area/System node",
    ]
    assert list(view["no_setup_zero_overdue"]) == [False, False, True]


def test_build_peer_adjusted_view_handles_dataset_with_no_peer_adjusted_columns():
    df = pd.DataFrame({"machine_id": ["A", "B"], "overdue_pms": [1, 0]})
    view = build_peer_adjusted_view(df)
    assert list(view["machine_id"]) == ["A", "B"]
    assert not any(field in view.columns for field in ALL_FIELDS)


def test_peer_adjusted_fields_are_never_swept_into_classifier_features():
    # Guards the Component II / Component III separation: none of these
    # peer-adjusted-analysis field names may appear in the raw classifier
    # feature columns, and none should be picked up by the numeric-feature
    # builder even when present (unprefixed, so the extra__ pass-through
    # never sees them either).
    assert not set(ALL_FIELDS) & set(RAW_FEATURE_COLUMNS)

    df = _sample_df()
    numeric = build_numeric_features(df, groups=["raw", "temporal", "extra"])
    assert not set(ALL_FIELDS) & set(numeric.columns)

    extra_only = build_extra_numeric_features(df)
    assert extra_only.empty or not set(ALL_FIELDS) & set(extra_only.columns)
