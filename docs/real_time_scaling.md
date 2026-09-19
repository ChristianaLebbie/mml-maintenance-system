# Future Real-Time Scaling

## Current state (Phase A -- research prototype)

```
Excel/CSV CMMS Export (Data Repo)
      |
      v
Dataset Validation & Feasibility
      |
      v
Model Training (offline, scripts/train_models.py)
      |
      v
Streamlit Decision Support (reads serialized artifacts only)
```

This prototype explicitly is **not** real-time. It uses a historical CMMS
export, and training happens only via `scripts/train_models.py` --
Streamlit never trains a model at request time (section 36).

## Planned evolution (Phase B -- operational deployment)

```
Live CMMS API / IoT Gateway / SCADA / Industrial Historian
      |
      v
Real-Time Data Adapter (new src/ingestion/*_source.py implementing the
same DataSource interface -- inspect()/load()/validate())
      |
      v
Same Standardized Schema (src/ingestion/schemas.py)
      |
      v
Same Preprocessing Pipeline (src/preprocessing/)
      |
      v
Validated, Versioned Model (unchanged -- src/inference/model_loader.py
already only loads serialized artifacts, so this layer requires no change)
      |
      v
Live Prediction -> SHAP Explanation -> Alerts -> Database
      |
      v
Streamlit / Power BI (unchanged)
```

## What would need to change

1. A new `DataSource` implementation (e.g. `APIDataSource`,
   `IoTStreamDataSource`) satisfying the same `inspect()/load()/validate()`
   contract as `ExcelDataSource`/`CSVDataSource` (`src/ingestion/base_source.py`).
2. If genuine repeated per-machine observations become available in the
   REAL data (not the synthetic demo track), switch the CMMS pipeline over
   to the already-implemented time-series-capable modules -- proven out
   against the synthetic dataset (see `docs/synthetic_demo_track.md`) so
   they're ready to point at real data the moment it exists:
   - `src/training/validation.py:chronological_split` (real CMMS training
     currently uses `stratified_split` -- the module's docstring explains
     why).
   - `src/features/lag_features.py` and
     `src/features/temporal_features.py:build_trend_features` (real CMMS
     feature building currently only uses the cross-sectional functions in
     those same modules).
   - `src/labels/failure_labels.py:construct_failure_within_horizon_label`
     (real CMMS labeling currently uses `compute_risk_score` +
     `fit_risk_thresholds` instead).
   - `src/training/sequence_model.py` (LSTM/GRU) -- trained and evaluated
     against the synthetic dataset via `scripts/train_sequence_model.py`;
     point it at real time-series data once available.
   - `src/features/frequency_features.py`: still a no-op -- implement
     FFT/vibration features only if a genuine high-frequency waveform
     source (e.g. PHM 2012-style bearing data) is integrated; the
     synthetic dataset is hourly telemetry, not a waveform, so it
     deliberately doesn't exercise this module either (see that module's
     docstring).
3. Controlled retraining cycle (section 37): predictions get stored,
   verified outcomes accumulate, and retraining stays a deliberate,
   versioned step (`scripts/train_models.py` + `scripts/select_model.py`)
   -- never automatic/online learning.

## What does not need to change

The database schema, the inference engine (`src/inference/`), the SHAP
explainability service, the alert engine, and the Streamlit pages are all
already source-agnostic -- they operate on the standardized schema and
serialized model artifacts, not on Excel specifically.
