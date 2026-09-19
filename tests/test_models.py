"""Phase 9-10 tests: validation split, baseline models, evaluation
metrics, Random Forest, and XGBoost training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.training.baselines import train_decision_tree, train_logistic_regression
from src.training.evaluation import evaluate_classifier
from src.training.random_forest import train_random_forest
from src.training.validation import chronological_split, stratified_split
from src.training.xgboost_model import train_xgboost


@pytest.fixture()
def classification_data():
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        weights=[0.85, 0.15],
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y_series = pd.Series(y)
    return X_df, y_series


def test_stratified_split_preserves_class_ratio(classification_data):
    _, y = classification_data
    split = stratified_split(y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(split.train_idx) + len(split.val_idx) + len(split.test_idx) == len(y)
    overall_ratio = y.mean()
    train_ratio = y.loc[split.train_idx].mean()
    assert abs(train_ratio - overall_ratio) < 0.08


def test_stratified_split_has_no_overlap(classification_data):
    _, y = classification_data
    split = stratified_split(y)
    train_set, val_set, test_set = set(split.train_idx), set(split.val_idx), set(split.test_idx)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_stratified_split_rejects_bad_ratios(classification_data):
    _, y = classification_data
    with pytest.raises(ValueError):
        stratified_split(y, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_chronological_split_train_precedes_val_precedes_test():
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=100, freq="h")})
    split = chronological_split(df, time_col="timestamp", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(split.train_idx) + len(split.val_idx) + len(split.test_idx) == 100
    assert df.loc[split.train_idx, "timestamp"].max() <= df.loc[split.val_idx, "timestamp"].min()
    assert df.loc[split.val_idx, "timestamp"].max() <= df.loc[split.test_idx, "timestamp"].min()


def test_chronological_split_rejects_bad_ratios():
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=10, freq="h")})
    with pytest.raises(ValueError):
        chronological_split(df, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_chronological_split_handles_unsorted_input():
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=20, freq="h")})
    shuffled = df.sample(frac=1, random_state=1)
    split = chronological_split(shuffled, time_col="timestamp", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert df.loc[split.train_idx, "timestamp"].max() <= df.loc[split.test_idx, "timestamp"].min()


def test_train_logistic_regression_fits_and_predicts(classification_data):
    X, y = classification_data
    model = train_logistic_regression(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_train_decision_tree_fits_and_predicts(classification_data):
    X, y = classification_data
    model = train_decision_tree(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_train_random_forest_fits_and_predicts_probabilities(classification_data):
    X, y = classification_data
    model = train_random_forest(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_train_xgboost_fits_and_predicts(classification_data):
    X, y = classification_data
    model = train_xgboost(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_evaluate_classifier_reports_expected_fields():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.6, 0.7, 0.9, 0.4])

    result = evaluate_classifier(y_true, y_pred, y_proba)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert result.roc_auc is not None
    assert result.pr_auc is not None
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_evaluate_classifier_handles_single_class_probabilities_gracefully():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    result = evaluate_classifier(y_true, y_pred, y_proba=np.array([0.1, 0.2, 0.1]))
    assert result.roc_auc is None  # undefined with only one class present
    assert result.pr_auc is None
    assert result.roc_curve is None
    assert result.pr_curve is None


def test_evaluate_classifier_includes_curve_data_for_visualization():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.6, 0.7, 0.9, 0.4])

    result = evaluate_classifier(y_true, y_pred, y_proba)
    assert result.roc_curve is not None
    assert "fpr" in result.roc_curve and "tpr" in result.roc_curve
    assert len(result.roc_curve["fpr"]) == len(result.roc_curve["tpr"])
    assert result.pr_curve is not None
    assert "precision" in result.pr_curve and "recall" in result.pr_curve


def test_evaluation_result_still_constructible_without_curve_fields():
    # Backward compatibility: code/tests constructing EvaluationResult
    # directly (or JSON saved before curve data existed) must keep working.
    from src.training.evaluation import EvaluationResult

    result = EvaluationResult(
        accuracy=0.9,
        precision=0.8,
        recall=0.7,
        f1=0.75,
        roc_auc=0.85,
        pr_auc=0.6,
        confusion_matrix=[[1, 0], [0, 1]],
        false_positives=0,
        false_negatives=0,
    )
    assert result.roc_curve is None
    assert result.pr_curve is None
