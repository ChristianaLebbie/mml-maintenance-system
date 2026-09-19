# System Architecture

## Overview

This system implements an explainable, CMMS-based maintenance-risk
decision-support pipeline for mining-relevant industrial equipment. It was
originally scoped around sensor-telemetry failure prediction, but the
actual research data (a Limble CMMS export from Marampa Mines) was
inspected before any modeling decisions were made (see
`docs/dataset_mapping.md` and project memory), and it contains no
continuous sensor telemetry and no populated failure/breakdown log. The
primary task was therefore reframed to **maintenance-risk classification**:
predicting each asset's current risk level from its profile and PM/work-
order compliance activity.

## Data Flow

```
Excel/CSV (Data Repo, data/raw/)
      |
      v
Ingestion (src/ingestion/excel_loader.py, csv_loader.py)
  - inspect(): profile sheets/columns without loading everything
  - validate(): extension, readability, sheet existence
  - load(): full DataFrame for the selected sheet
      |
      v
Column Mapping (src/ingestion/mapper.py)
  - suggest_mapping(): alias-based auto-suggestion
  - apply_mapping(): source columns -> standardized schema
  - save_mapping()/load_mapping(): persisted in config/datasets.yaml
      |
      v
Standardized Schema (src/ingestion/schemas.py)
  machine_id, equipment_category, manufacturer, criticality,
  overdue_pms, overdue_wos, total_completed_pms, total_completed_wos,
  downtime_minutes, total_cost, last_completed_pm, last_completed_wo,
  failure_label (present only if a dataset actually has one)
      |
      v
Preprocessing (src/preprocessing/)
  - cleaning.py: whitespace/type normalization
  - missing_values.py: 0 for numeric compliance metrics, "Unknown" for
    categorical attributes (never mode-imputed)
  - outliers.py: IQR detection, report-only (never auto-removed)
  - scaling.py: fit-on-train-only StandardScaler/MinMaxScaler
      |
      v
Dataset Feasibility (src/feasibility/analyzer.py)
  Reports which modelling tasks the data actually supports -- never
  assumes a failure label exists.
      |
      v
Label Construction (src/labels/failure_labels.py)
  Composite risk score from overdue_pms/overdue_wos/downtime_minutes;
  Watch/High-Risk thresholds fit ONLY on the training split's own score
  distribution (never invented, never leaked from test data).
      |
      v
Feature Engineering (src/features/)
  - raw_features.py, temporal_features.py, statistical_features.py: real
    signal for this dataset.
  - lag_features.py, event_features.py, frequency_features.py: honest
    no-ops -- this dataset has no per-machine time series or event log or
    vibration waveform to build them from.
  - feature_pipeline.py: combines active numeric groups + a
    fit-on-train-only OneHotEncoder for categorical attributes.
  IMPORTANT: overdue_pms/overdue_wos/downtime_minutes are excluded from
  the model's feature set because the risk label is built from them --
  including them would make the task circular (see project memory
  "project-modeling-design-decisions").
      |
      v
Model Training (src/training/)
  - validation.py: stratified random split (not chronological -- the data
    is a single point-in-time cross-section, so there is no time order to
    respect; see the module's own docstring).
  - baselines.py, random_forest.py, xgboost_model.py, optimization.py
  - evaluation.py: accuracy/precision/recall/F1/ROC-AUC/PR-AUC/confusion
    matrix, with PR-AUC/recall prioritized over accuracy.
      |
      v
Explainability (src/explainability/shap_service.py)
  TreeExplainer over Random Forest/XGBoost; global importance and
  per-prediction local explanations.
      |
      v
Model Selection (scripts/select_model.py)
  Ranks by PR-AUC then recall; writes model_selection_report.md; registers
  the winning ModelVersion as active in the database.
      |
      v
Inference Engine (src/inference/)
  - model_loader.py: loads the active model + encoder + metadata only
    (training and inference are fully separate -- see scripts/train_models.py
    vs. the Streamlit pages).
  - predictor.py: rebuilds features identically to training, predicts,
    classifies risk, explains.
  - risk_engine.py: probability -> Normal/Watch/High Risk using the
    thresholds fit during training.
      |
      v
Alerts (src/alerts/alert_service.py) + Database (database/)
      |
      v
Streamlit Decision-Support Interface (app.py, pages/)
Power BI Export (scripts/export_powerbi_data.py -> data/exports/powerbi/)
```

## Why not the original sensor-telemetry design

See `docs/dataset_mapping.md` for the field-by-field mapping and
`reports/dataset_feasibility/*.md` for the generated feasibility reports
that justify this reframing from real, inspected data rather than
assumption.

## Future Real-Time Scaling

See `docs/real_time_scaling.md`.
