"""Phase 10 tests: SHAP global importance and local per-prediction
explanations against a real fitted Random Forest."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.explainability.shap_service import build_explainer, explain_instance, global_importance
from src.training.random_forest import train_random_forest


@pytest.fixture()
def fitted_model_and_data():
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, weights=[0.8, 0.2], random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y_series = pd.Series(y)
    model = train_random_forest(X_df, y_series)
    return model, X_df


def test_global_importance_returns_all_features_ranked(fitted_model_and_data):
    model, X = fitted_model_and_data
    explainer = build_explainer(model, background=X.sample(50, random_state=42))
    importance = global_importance(explainer, X)

    assert set(importance.index) == set(X.columns)
    assert (importance.values >= 0).all()
    # sorted descending
    assert list(importance.values) == sorted(importance.values, reverse=True)


def test_explain_instance_returns_ranked_local_explanations(fitted_model_and_data):
    model, X = fitted_model_and_data
    explainer = build_explainer(model, background=X.sample(50, random_state=42))
    single_row = X.iloc[[0]]

    explanations = explain_instance(explainer, single_row)
    assert len(explanations) == len(X.columns)
    ranks = [e.importance_rank for e in explanations]
    assert ranks == sorted(ranks)
    magnitudes = [abs(e.shap_value) for e in explanations]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_instance_feature_values_match_input_row(fitted_model_and_data):
    model, X = fitted_model_and_data
    explainer = build_explainer(model, background=X.sample(50, random_state=42))
    single_row = X.iloc[[3]]

    explanations = explain_instance(explainer, single_row)
    by_name = {e.feature_name: e.feature_value for e in explanations}
    for col in X.columns:
        assert by_name[col] == pytest.approx(single_row.iloc[0][col])
