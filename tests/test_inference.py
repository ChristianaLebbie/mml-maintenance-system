"""Phase 13 tests: the inference pipeline (validate -> features -> predict
-> intervention priority -> explain), built against a real fitted model so
the feature construction is exercised exactly as it runs in production,
not mocked."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.feature_pipeline import fit_categorical_encoder
from src.inference.model_loader import LoadedModel
from src.inference.predictor import build_inference_features, predict, validate_input
from src.services.model_service import NUMERIC_FEATURE_GROUPS
from src.features.feature_pipeline import build_numeric_features, apply_categorical_encoder
from src.labels.failure_labels import resolve_priority_weights
from src.training.random_forest import train_random_forest


@pytest.fixture()
def standardized_df():
    return pd.DataFrame(
        {
            "machine_id": ["A", "B", "C", "D", "E", "F"],
            "equipment_category": ["Crusher", "Pump", "Crusher", "Pump", "Crusher", "Pump"],
            "manufacturer": ["KSB", "WEG", "KSB", "WEG", "KSB", "WEG"],
            "criticality": ["Unknown"] * 6,
            "total_completed_pms": [30, 10, 25, 5, 40, 2],
            "total_completed_wos": [2, 0, 1, 3, 0, 5],
            "total_cost": [0.0] * 6,
            "last_completed_pm": pd.to_datetime(
                ["2025-06-01", "2025-05-01", "2025-06-10", "2025-01-01", "2025-06-15", "2024-12-01"]
            ),
            "last_completed_wo": pd.to_datetime([None] * 6),
        }
    )


@pytest.fixture()
def loaded_model(standardized_df):
    numeric = build_numeric_features(standardized_df, groups=NUMERIC_FEATURE_GROUPS)
    numeric = numeric.drop(columns=list(resolve_priority_weights(standardized_df)), errors="ignore")
    encoder = fit_categorical_encoder(standardized_df)
    categorical = apply_categorical_encoder(standardized_df, encoder)
    X = pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
    y = pd.Series([0, 1, 0, 1, 0, 1])  # arbitrary but fits without error

    model = train_random_forest(X, y)
    metadata = {
        "feature_columns": list(X.columns),
        "watch_threshold": 0.3,
        "high_threshold": 0.6,
    }
    return LoadedModel(
        model=model,
        encoder=encoder,
        metadata=metadata,
        model_version_id=1,
        version="1.0.0-test",
        model_type="random_forest",
    )


def test_validate_input_requires_machine_id():
    df = pd.DataFrame({"total_completed_pms": [1]})
    problems = validate_input(df)
    assert any("machine_id" in p for p in problems)


def test_validate_input_passes_with_machine_id():
    df = pd.DataFrame({"machine_id": ["A"]})
    assert validate_input(df) == []


def test_build_inference_features_matches_training_columns(standardized_df, loaded_model):
    X = build_inference_features(standardized_df, loaded_model)
    assert list(X.columns) == loaded_model.metadata["feature_columns"]


def test_build_inference_features_fills_unseen_category_columns_with_zero(loaded_model):
    new_row = pd.DataFrame(
        {
            "machine_id": ["Z"],
            "equipment_category": ["NeverSeenBefore"],
            "manufacturer": ["NeverSeenBefore"],
            "criticality": ["Unknown"],
            "total_completed_pms": [1],
            "total_completed_wos": [1],
            "total_cost": [0.0],
            "last_completed_pm": pd.to_datetime(["2025-01-01"]),
            "last_completed_wo": pd.to_datetime([None]),
        }
    )
    X = build_inference_features(new_row, loaded_model)
    assert list(X.columns) == loaded_model.metadata["feature_columns"]
    assert not X.isna().any().any()


def test_predict_raises_on_missing_machine_id(loaded_model):
    df = pd.DataFrame({"total_completed_pms": [1]})
    with pytest.raises(ValueError):
        predict(df, loaded_model)


def test_predict_returns_one_result_per_row(standardized_df, loaded_model):
    results = predict(standardized_df, loaded_model)
    assert len(results) == len(standardized_df)
    assert {r.machine_id for r in results} == set(standardized_df["machine_id"])


def test_predict_probabilities_are_valid(standardized_df, loaded_model):
    results = predict(standardized_df, loaded_model)
    for r in results:
        assert 0.0 <= r.failure_probability <= 1.0
        assert r.predicted_class in (0, 1)
        assert r.intervention_priority in ("Normal", "Watch", "High Priority")


def test_predict_includes_shap_explanations_when_background_given(standardized_df, loaded_model):
    results = predict(standardized_df, loaded_model, background=standardized_df)
    assert all(len(r.top_factors) > 0 for r in results)
    for r in results:
        ranks = [f["rank"] for f in r.top_factors]
        assert ranks == sorted(ranks)


def test_predict_without_background_skips_explanations(standardized_df, loaded_model):
    results = predict(standardized_df, loaded_model, background=None)
    assert all(r.top_factors == [] for r in results)


def test_predict_reports_progress_once_per_row(standardized_df, loaded_model):
    calls: list[tuple[str, float]] = []
    predict(
        standardized_df,
        loaded_model,
        progress_callback=lambda msg, frac: calls.append((msg, frac)),
    )

    # One call per row scored, plus a final "Done" call at fraction 1.0.
    assert len(calls) == len(standardized_df) + 1
    fractions = [f for _, f in calls]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_predict_without_progress_callback_still_works(standardized_df, loaded_model):
    # progress_callback is optional -- must not be required.
    results = predict(standardized_df, loaded_model, progress_callback=None)
    assert len(results) == len(standardized_df)
