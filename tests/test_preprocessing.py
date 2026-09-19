"""Phase 6 tests: cleaning, missing-value handling, outlier detection,
scaling, and the orchestrating pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaning import clean, count_duplicate_rows, drop_duplicate_rows
from src.preprocessing.missing_values import UNKNOWN_LABEL, fill_missing, summarize_missing
from src.preprocessing.outliers import detect_outliers_iqr, summarize_outliers
from src.preprocessing.pipeline import run_preprocessing
from src.preprocessing.scaling import apply_scaler, fit_scaler, load_scaler, save_scaler


@pytest.fixture()
def raw_df():
    return pd.DataFrame(
        {
            "machine_id": [" 4212-CR-001 ", "4212-PU-002", "4212-CR-001 "],
            "equipment_category": ["Crusher", None, "Crusher"],
            "manufacturer": ["", "KSB", ""],
            "criticality": [None, None, None],
            "overdue_pms": ["2", "0", "2"],
            "overdue_wos": [1, 0, 1],
            "total_completed_pms": [30, 10, 30],
            "total_completed_wos": [2, 0, 2],
            "downtime_minutes": [120.0, 0.0, 120.0],
            "total_cost": [None, None, None],
            "last_completed_pm": ["2025/06/04", "2025/06/10", "2025/06/04"],
            "last_completed_wo": [None, None, None],
            "failure_label": [None, None, None],
            "source_dataset": ["x"] * 3,
        }
    )


def test_clean_strips_whitespace_and_coerces_numerics(raw_df):
    cleaned = clean(raw_df)
    assert cleaned["machine_id"].iloc[0] == "4212-CR-001"
    assert pd.api.types.is_numeric_dtype(cleaned["overdue_pms"])
    assert cleaned["overdue_pms"].iloc[0] == 2


def test_clean_converts_empty_string_to_none(raw_df):
    cleaned = clean(raw_df)
    assert cleaned["manufacturer"].iloc[0] is None


def test_clean_parses_dates(raw_df):
    cleaned = clean(raw_df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["last_completed_pm"])


def test_count_and_drop_duplicate_rows(raw_df):
    cleaned = clean(raw_df)
    assert count_duplicate_rows(cleaned) == 1
    deduped, n_dropped = drop_duplicate_rows(cleaned)
    assert n_dropped == 1
    assert len(deduped) == 2


def test_summarize_missing(raw_df):
    cleaned = clean(raw_df)
    missing = summarize_missing(cleaned)
    assert missing["criticality"] == 100.0
    assert missing["overdue_pms"] == 0.0


def test_fill_missing_numeric_defaults_to_zero(raw_df):
    cleaned = clean(raw_df)
    filled = fill_missing(cleaned)
    assert (filled["total_cost"] == 0.0).all()


def test_fill_missing_categorical_defaults_to_unknown_not_mode(raw_df):
    cleaned = clean(raw_df)
    filled = fill_missing(cleaned)
    assert (filled["criticality"] == UNKNOWN_LABEL).all()


def test_fill_missing_never_imputes_machine_id():
    df = pd.DataFrame({"machine_id": ["A", None, "C"]})
    filled = fill_missing(df)
    assert filled["machine_id"].isna().sum() == 1


def test_detect_outliers_iqr_flags_extreme_value():
    df = pd.DataFrame({"downtime_minutes": [10, 12, 11, 9, 10, 500]})
    flags = detect_outliers_iqr(df, columns=["downtime_minutes"])
    assert flags["downtime_minutes"].iloc[-1] == True  # noqa: E712
    assert flags["downtime_minutes"].iloc[0] == False  # noqa: E712


def test_summarize_outliers_counts_per_column():
    df = pd.DataFrame({"downtime_minutes": [10, 12, 11, 9, 10, 500]})
    counts = summarize_outliers(df, columns=["downtime_minutes"])
    assert counts["downtime_minutes"] == 1


def test_run_preprocessing_reports_duplicates_and_missingness(raw_df):
    result_df, report = run_preprocessing(raw_df)
    assert report.rows_before == 3
    assert report.duplicates_dropped == 1
    assert report.rows_after == 2
    assert result_df["criticality"].tolist() == [UNKNOWN_LABEL, UNKNOWN_LABEL]


def test_fit_and_apply_scaler_standard():
    df = pd.DataFrame({"downtime_minutes": [0.0, 10.0, 20.0, 30.0]})
    scaler = fit_scaler(df, columns=["downtime_minutes"], method="standard")
    scaled = apply_scaler(df, scaler, columns=["downtime_minutes"])
    assert np.isclose(scaled["downtime_minutes"].mean(), 0.0, atol=1e-9)


def test_scaler_must_be_fit_only_on_training_then_applied_to_test():
    train = pd.DataFrame({"downtime_minutes": [0.0, 10.0, 20.0]})
    test = pd.DataFrame({"downtime_minutes": [100.0]})  # out-of-range value
    scaler = fit_scaler(train, columns=["downtime_minutes"], method="minmax")
    scaled_test = apply_scaler(test, scaler, columns=["downtime_minutes"])
    # scaled using train's min/max (0-20), so 100 maps far outside [0, 1]
    assert scaled_test["downtime_minutes"].iloc[0] > 1.0


def test_save_and_load_scaler_roundtrip(tmp_path):
    df = pd.DataFrame({"downtime_minutes": [0.0, 10.0, 20.0]})
    scaler = fit_scaler(df, columns=["downtime_minutes"], method="standard")
    path = tmp_path / "scaler.joblib"
    save_scaler(scaler, path)
    loaded = load_scaler(path)
    scaled = apply_scaler(df, loaded, columns=["downtime_minutes"])
    assert np.isclose(scaled["downtime_minutes"].mean(), 0.0, atol=1e-9)
