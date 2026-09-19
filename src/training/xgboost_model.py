"""XGBoost model. Hyperparameter search lives in optimization.py (Optuna,
train/validation only -- never optimized against the test set)."""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from config.settings import MODEL_CONFIG, RANDOM_SEED


def train_xgboost(
    X: pd.DataFrame, y: pd.Series, overrides: dict | None = None
) -> xgb.XGBClassifier:
    cfg = {**MODEL_CONFIG["xgboost"], **(overrides or {})}
    # scale_pos_weight defaults to the auto-computed class-imbalance ratio
    # (unchanged production behavior); pass overrides={"scale_pos_weight": 1.0}
    # to disable imbalance weighting for comparison (see
    # scripts/run_advanced_ablation.py).
    if overrides and "scale_pos_weight" in overrides:
        scale_pos_weight = overrides["scale_pos_weight"]
    else:
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        min_child_weight=cfg["min_child_weight"],
        gamma=cfg["gamma"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X, y)
    return model
