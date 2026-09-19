# Model Selection Report: complete_sample_dbg (version dbg1)

## Candidates evaluated

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| logistic_regression **(selected)** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| decision_tree | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| random_forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| xgboost | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Selection rationale
**logistic_regression** was selected because it has the highest PR-AUC (1.000) among candidates, which is the appropriate primary metric given the high-risk class is the minority class. Recall is used as the tie-breaker. Accuracy was not used as the deciding factor (section 33).
