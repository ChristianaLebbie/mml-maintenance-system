# Testing Guide

This system has three layers of testing: automated unit/integration tests,
end-to-end pipeline runs via scripts (against both the real and synthetic
data), and manual verification of the Streamlit application. Run them in
this order when validating a fresh checkout or a new change.

## 1. Automated test suite (fastest, run this first)

```powershell
.\venv\Scripts\Activate.ps1
pytest -v
```

144 tests, organized by what they cover:

| Test file | Covers |
|---|---|
| `test_foundation.py` | Config loading, path resolution, logging |
| `test_database.py` | ORM models, repositories, CRUD, foreign keys |
| `test_excel_loader.py`, `test_ingestion.py` | Excel/CSV inspect/load/validate, duplicate/missingness detection |
| `test_mapping.py` | Column-mapping suggestion, validation, application, persistence |
| `test_feasibility.py` | Feasibility analysis, including the "never fabricate a failure label" rule |
| `test_preprocessing.py` | Cleaning, missing-value handling, outlier detection, scaling (fit/apply split) |
| `test_labels.py` | Risk-score/threshold fit-apply split; near-term failure-horizon label windowing (including the machine-boundary and off-by-one edge cases) |
| `test_features.py` | Every feature group, including the no-cross-machine-leakage guarantee for lag/trend features |
| `test_models.py` | Stratified split, chronological split, baseline/RF/XGBoost training, evaluation metrics |
| `test_explainability.py` | SHAP global importance and local explanations against a real fitted model |
| `test_inference.py`, `test_risk.py`, `test_alerts.py` | Full inference pipeline, risk classification, alert decision logic |
| `test_sequence_model.py` | LSTM windowing (including machine-boundary and insufficient-history cases), training/prediction plumbing |
| `test_services.py` | `train_and_activate` (one ExperimentRun per candidate, exactly one active ModelVersion, deactivation on retrain), progress-callback sequencing for import/training, and `delete_dataset`'s full cascade (DB rows across every dependent table, every candidate model's files -- not just the active one, saved mapping) -- against an isolated in-memory DB and tmp_path filesystem, never the real project database/model files |
| `test_streamlit_pages.py` | Every page executes without exception (via Streamlit's `AppTest` -- catches real runtime bugs a plain HTTP check would miss) |
| `test_powerbi_export.py` | CSV export functions, including graceful handling of an empty database |

Run a single file or test during development:
```powershell
pytest tests/test_labels.py -v
pytest tests/test_labels.py::test_construct_failure_within_horizon_label_excludes_the_failure_row_itself -v
```

If you change `database/models.py`, delete and reinitialize the dev
database (do NOT do this if it holds data you want to keep):
```powershell
Remove-Item database\predictive_maintenance.db
python scripts/initialize_database.py
```

## 2. End-to-end pipeline: real CMMS data track

Verifies ingestion through inference actually works against real data, not
just isolated units.

```powershell
python scripts/initialize_database.py
python scripts/prepare_dataset.py "Data Repo\PM Spot Check.xlsx" pm_spot_check PROCESS_PLANT
python scripts/train_models.py pm_spot_check 1.0.0
python scripts/select_model.py pm_spot_check 1.0.0
python scripts/evaluate_models.py pm_spot_check xgboost 1.0.0
python scripts/run_ablation.py pm_spot_check 1.0.0
python scripts/export_powerbi_data.py
```

What to check at each step:
- `prepare_dataset.py` prints machine count, duplicates dropped, and the
  feasibility report's `supported_tasks`/`limitations` -- confirm these
  match what you'd expect from the file you imported (e.g. no
  `binary_failure_classification` if the file has no failure column).
- `train_models.py` prints a metrics table for all 4 models -- sanity
  check that PR-AUC/recall are non-trivial (not all 0 or all 1, which
  would indicate a label or feature bug) and that `models/trained/`,
  `models/preprocessors/`, `models/metadata/` got new files.
- `select_model.py` prints which model won and why -- open the generated
  `reports/experiments/<dataset>_<version>/model_selection_report.md`.
- `evaluate_models.py` should reproduce the same metrics `train_models.py`
  printed for that model (both use the same fixed random seed) -- if they
  differ, something about the split or feature construction is
  non-deterministic and needs investigating.
- `export_powerbi_data.py` -- open the CSVs under `data/exports/powerbi/`
  and confirm row counts match what's in the database
  (`sqlite3 database/predictive_maintenance.db "select count(*) from predictions"`).

## 3. End-to-end pipeline: synthetic demonstration track

Verifies the time-series-dependent methodology (see
`docs/synthetic_demo_track.md`) that the real data can't exercise.

```powershell
python scripts/generate_synthetic_dataset.py
python scripts/train_sequence_model.py 24 24
```

Check: the printed comparison table should show both models with
ROC-AUC well above 0.5 (random) -- if not, the synthetic degradation
signal or the label window isn't wired correctly. Confirm
`models/trained/lstm_synthetic_demo.keras` was written.

## 4. Manual verification of the Streamlit application

Automated `AppTest` checks (in the pytest suite) confirm every page runs
without raising, but they don't verify the UI *looks* and *behaves*
correctly -- do this manually after any page change:

```powershell
streamlit run app.py
```

Walkthrough (assumes step 2 above has been run so there's real data to
look at):
1. **Overview** (`app.py`) -- KPI cards should show non-zero machine
   count, active model, prediction count once you've run predictions.
2. **Import Dataset** -- try the optional "Download template" button
   first (confirms it produces an openable .xlsx). Upload a small
   CSV/Excel -- including one with columns that DON'T match any known
   field, to confirm they still come through as `extra__*` rather than
   vanishing. Confirm the sheet picker (Excel only), preview table,
   column-profile expander, and mapping dropdowns all populate; submitting
   with a required field unmapped should disable/block the "Prepare
   Dataset" button. After a successful prepare, use the new "Train Models"
   section: enter/keep a version string and click "Train Models on This
   Dataset" -- confirm the comparison table appears and that
   `models/trained/`, `reports/experiments/<dataset>_<version>/`, and the
   active `ModelVersion` in the database all update. Re-uploading a
   different file should clear the previous prepare/train results from
   the page.
3. **Dataset Feasibility** -- select the prepared dataset, confirm the
   rendered report matches the file under `reports/dataset_feasibility/`.
   Test the **Danger Zone**: confirm the delete button stays disabled
   until both the checkbox is checked AND the typed name exactly matches;
   confirm switching to a different dataset resets both (no carried-over
   "armed" state); after deleting a test dataset, confirm it disappears
   from every page that lists datasets/machines (Dataset Feasibility,
   Machine Monitoring, Run Prediction, Alerts) without needing a manual
   refresh -- this depends on `clear_all_caches()` being called after
   deletion, not just the DB row being gone.
4. **Machine Monitoring** -- pick a machine, confirm metadata and (if any
   predictions exist for it) the probability-over-time chart render.
5. **Run Prediction** -- both modes: select an existing machine and run a
   prediction (check probability/risk/top-factors appear); upload a small
   CSV with just a `machine_id` column and confirm it doesn't crash on
   missing optional fields.
6. **Explainability** -- pick a recent prediction, confirm the SHAP bar
   chart and plain-language summary render and the direction (increases/
   decreases risk) matches the sign of the SHAP value.
7. **Alerts** -- filter by status/machine, change an alert's status, and
   confirm the change persists after `st.rerun()` (reload the page).
8. **Prediction History** -- confirm the CSV download button produces a
   file matching the on-screen table.
9. **Model Performance** -- pick an experiment, confirm the comparison
   table/confusion matrices match `reports/experiments/<name>/*.json`, and
   that the selection report renders.
10. **System Information** -- static content; confirm it reflects the
    currently active model/dataset counts. Click **Clear Cache** and
    confirm it succeeds without error (the underlying data won't visibly
    change unless something was actually stale).

## 5. What "passing" actually means here

A green pytest run and a clickable Streamlit app are necessary but not
sufficient for a thesis claim. Before citing any number in the write-up,
trace it back to an actual script run (the JSON/Markdown files under
`reports/`) rather than trusting a remembered figure -- and if a number
can't be traced to a file, regenerate it rather than guessing.
