"""SHAP explainability for tree-based models (Random Forest, XGBoost).

Provides global feature importance (for the Model Performance page) and
per-prediction local explanations (for the Explainability page / stored
prediction_explanations rows) -- both computed on real fitted models only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


@dataclass
class LocalExplanation:
    feature_name: str
    feature_value: float
    shap_value: float
    importance_rank: int


def build_explainer(model, background: pd.DataFrame) -> shap.TreeExplainer:
    return shap.TreeExplainer(model, data=background)


def global_importance(explainer: shap.TreeExplainer, X: pd.DataFrame) -> pd.Series:
    shap_values = explainer.shap_values(X)
    values = _positive_class_values(shap_values)
    mean_abs = np.abs(values).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)


def explain_instance(
    explainer: shap.TreeExplainer, x_row: pd.DataFrame
) -> list[LocalExplanation]:
    """`x_row` must be a single-row DataFrame with the model's feature
    columns."""
    shap_values = explainer.shap_values(x_row)
    values = _positive_class_values(shap_values)[0]

    explanations = [
        LocalExplanation(
            feature_name=col,
            feature_value=float(x_row.iloc[0][col]),
            shap_value=float(val),
            importance_rank=0,
        )
        for col, val in zip(x_row.columns, values)
    ]
    explanations.sort(key=lambda e: abs(e.shap_value), reverse=True)
    for rank, explanation in enumerate(explanations, start=1):
        explanation.importance_rank = rank
    return explanations


def _positive_class_values(shap_values) -> np.ndarray:
    """Normalize shap_values across shap/model versions: binary
    classifiers may return a list of two arrays ([class0, class1]) or a
    single (n_samples, n_features) array for the positive class already."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) -- take the positive class
        return arr[:, :, -1]
    return arr
