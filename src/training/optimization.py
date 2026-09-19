"""Optuna hyperparameter search for XGBoost. Optimizes PR-AUC on a
validation split -- the test split is never touched here, per the
leakage-prevention rule (optimize on train/val only, report final numbers
on test separately in evaluation.py)."""

from __future__ import annotations

from dataclasses import dataclass

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from config.settings import MODEL_CONFIG, RANDOM_SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    best_params: dict
    best_value: float
    n_trials: int


def _suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", *search_space["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *search_space["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *search_space["subsample"]),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *search_space["colsample_bytree"]
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", *search_space["min_child_weight"]
        ),
        "gamma": trial.suggest_float("gamma", *search_space["gamma"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *search_space["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *search_space["reg_lambda"]),
    }


def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int | None = None,
) -> OptimizationResult:
    cfg = MODEL_CONFIG["xgboost"]["optuna"]
    n_trials = n_trials or cfg["n_trials"]
    search_space = cfg["search_space"]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, search_space)
        model = xgb.XGBClassifier(
            **params, random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=-1
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials)

    return OptimizationResult(
        best_params=study.best_params, best_value=study.best_value, n_trials=len(study.trials)
    )
