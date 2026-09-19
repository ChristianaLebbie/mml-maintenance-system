# Model Selection Report: pm_spot_check (version 1.0.0)

## Candidates evaluated

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| decision_tree | 0.940 | 0.701 | 0.701 | 0.701 | 0.803 | 0.603 |
| logistic_regression | 0.846 | 0.368 | 0.731 | 0.490 | 0.921 | 0.625 |
| random_forest | 0.946 | 0.763 | 0.672 | 0.714 | 0.951 | 0.742 |
| xgboost **(selected)** | 0.938 | 0.703 | 0.672 | 0.687 | 0.945 | 0.766 |

## Selection rationale
**xgboost** was selected because it has the highest PR-AUC (0.766) among candidates, which is the appropriate primary metric given the high-risk class is the minority class. Recall is used as the tie-breaker. Accuracy was not used as the deciding factor (section 33).

## Known limitation
No LSTM/GRU sequence model was evaluated: the confirmed dataset is a single point-in-time CMMS snapshot with no per-machine sequential observations, so a sequence model has no valid input structure here (see project memory 'project-modeling-design-decisions').
