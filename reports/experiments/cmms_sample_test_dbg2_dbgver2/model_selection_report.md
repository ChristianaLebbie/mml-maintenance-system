# Model Selection Report: cmms_sample_test_dbg2 (version dbgver2)

## Candidates evaluated

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| logistic_regression | 0.800 | 0.000 | 0.000 | 0.000 | 0.345 | 0.050 |
| decision_tree | 0.833 | 0.000 | 0.000 | 0.000 | 0.431 | 0.033 |
| random_forest **(selected)** | 0.967 | 0.000 | 0.000 | 0.000 | 0.500 | 0.062 |
| xgboost | 0.967 | 0.000 | 0.000 | 0.000 | 0.483 | 0.062 |

## Selection rationale
**random_forest** was selected because it has the highest PR-AUC (0.062) among candidates, which is the appropriate primary metric given the high-risk class is the minority class. Recall is used as the tie-breaker. Accuracy was not used as the deciding factor (section 33).
