# System Requirements

## Functional requirements (status against section 68's acceptance criteria)

| # | Requirement | Status |
|---|---|---|
| 1-2 | Application launches; database initializes | Done -- `app.py`, `scripts/initialize_database.py` |
| 3-7 | Excel upload, sheet inspection, preview, column mapping, mapping validation | Done -- `pages/2_Import_Dataset.py`, `src/ingestion/` |
| 8 | Feasibility report generated | Done -- `src/feasibility/`, `pages/3_Dataset_Feasibility.py` |
| 9 | Processed data saved | Done -- `data/processed/*.parquet` |
| 10 | Valid target constructed where supported | Done, reframed -- `src/labels/failure_labels.py` builds a maintenance-risk label (not a fabricated failure label; see docs/dataset_mapping.md) |
| 11 | Leakage-free splits | Done, reframed -- stratified random split, not chronological (see `src/training/validation.py` docstring for why) |
| 12-15 | Baseline, RF, XGBoost trained; sequence model | Done for baseline/RF/XGBoost on the real CMMS data. LSTM/GRU (`src/training/sequence_model.py`) is implemented and trained/evaluated against the **synthetic** demonstration dataset (`scripts/train_sequence_model.py`) -- see `docs/synthetic_demo_track.md`. Not run against the real CMMS data, which has no per-machine time series to model. |
| 16 | Evaluation reports generated | Done -- `reports/experiments/<dataset>_<version>/*_metrics.json` |
| 17 | Feature ablation | Done -- `scripts/run_ablation.py` compares `raw`/`temporal`/`raw+temporal` (the `statistical` group is deliberately excluded; see the script's docstring). Machine-holdout and imbalance-strategy comparisons are a follow-up |
| 18 | Model artifacts saved | Done -- `models/trained/`, `models/preprocessors/`, `models/metadata/` |
| 19-21 | Active model loads; prediction generated; probability appears | Done -- `src/inference/`, `pages/5_Run_Prediction.py` |
| 22-24 | Risk level; SHAP explanation; prediction saved | Done -- `src/inference/risk_engine.py`, `src/explainability/`, `src/services/prediction_service.py` |
| 25-26 | Alert generated/visible | Done -- `src/alerts/`, `pages/7_Alerts.py` |
| 27-28 | Prediction/machine history visible | Done -- `pages/4_Machine_Monitoring.py`, `pages/8_Prediction_History.py` |
| 29-30 | Model performance; dataset feasibility pages | Done -- `pages/9_Model_Performance.py`, `pages/3_Dataset_Feasibility.py` |
| 31 | Power BI exports | Done -- `scripts/export_powerbi_data.py` |
| 32 | Automated tests pass | Done -- 128 tests across ingestion, mapping, feasibility, preprocessing, labels, features, models, explainability, inference, risk, alerts, database, Streamlit pages, Power BI export |
| 33 | Documentation | Done -- this file plus architecture/database/dataset-mapping/real-time-scaling/Power BI docs |

## Technology stack (as actually installed)

Python 3.11.9 (a separate venv from the machine's default 3.14, required
for TensorFlow/SHAP/XGBoost wheel availability -- see project memory
"project-python-environment-setup"). Exact pinned versions: see
`requirements.txt`.

## Environment

- OS: Windows 11
- Database: SQLite (`database/predictive_maintenance.db`)
- No GPU required; all models are CPU-trained (Random Forest, XGBoost,
  Logistic Regression, Decision Tree)

## Known deviations from the original design brief

All documented with rationale in `docs/architecture.md`,
`docs/dataset_mapping.md`, and project memory
"project-modeling-design-decisions": primary task reframing (sensor
failure prediction -> CMMS maintenance-risk classification), stratified
split instead of chronological, feature/label separation to avoid a
circular model, and no LSTM/GRU sequence model.
