"""Baseline models: Logistic Regression and Decision Tree. These establish
how hard the task is before reaching for Random Forest / XGBoost."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config.settings import MODEL_CONFIG, RANDOM_SEED


def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    cfg = MODEL_CONFIG["baselines"]["logistic_regression"]
    model = LogisticRegression(
        max_iter=cfg["max_iter"],
        class_weight=cfg["class_weight"],
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)
    return model


def train_decision_tree(X: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
    cfg = MODEL_CONFIG["baselines"]["decision_tree"]
    model = DecisionTreeClassifier(
        max_depth=cfg["max_depth"],
        class_weight=cfg["class_weight"],
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)
    return model
