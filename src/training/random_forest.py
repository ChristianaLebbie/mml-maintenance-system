"""Random Forest -- the main explainable model (paired with SHAP in
src/explainability/shap_service.py)."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config.settings import MODEL_CONFIG, RANDOM_SEED


def train_random_forest(
    X: pd.DataFrame, y: pd.Series, overrides: dict | None = None
) -> RandomForestClassifier:
    cfg = {**MODEL_CONFIG["random_forest"], **(overrides or {})}
    model = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        class_weight=cfg["class_weight"],
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model
