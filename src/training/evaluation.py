"""Classification evaluation metrics, emphasizing recall/F1/PR-AUC over
raw accuracy since the high-priority class is expected to be the minority
(see section 29 of the project brief)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: list[list[int]]
    false_positives: int
    false_negatives: int
    # Curve data for visualization (Model Performance page) -- optional
    # with a default so existing code constructing EvaluationResult
    # directly (or reading older saved metrics JSON without these keys)
    # keeps working unchanged.
    roc_curve: dict[str, list[float]] | None = field(default=None)
    pr_curve: dict[str, list[float]] | None = field(default=None)


def evaluate_classifier(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> EvaluationResult:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    roc_auc = None
    pr_auc = None
    roc_curve_data = None
    pr_curve_data = None
    if y_proba is not None and len(set(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        pr_curve_data = {"precision": precision_vals.tolist(), "recall": recall_vals.tolist()}

    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm.tolist(),
        false_positives=int(fp),
        false_negatives=int(fn),
        roc_curve=roc_curve_data,
        pr_curve=pr_curve_data,
    )
