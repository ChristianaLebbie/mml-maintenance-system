# Explainable Intervention Prioritization System

Explainable machine learning-based predictive maintenance for
mining-relevant industrial equipment — a Master's thesis research system.

> **Status:** All 17 phases implemented and verified against real data (193 automated tests, including live Streamlit page execution checks). See [Development Phases](#development-phases) for what's fully done vs. follow-up work (e.g. machine-holdout and imbalance-strategy experiments).

> **New here?** Read [`docs/user_guide.md`](docs/user_guide.md) first — a complete beginner's guide to what this system does, how to install and run it, a page-by-page tour, a glossary of terms, current limitations, and suggested future enhancements. The rest of this README is a quick reference for people who already know the project.

## Project Objective

To design, develop and evaluate an explainable machine learning-based
predictive-maintenance framework and integrated decision-support system for
mining-relevant industrial equipment, initially using historical/sample
Excel data and public benchmark datasets, while providing a scalable
architecture for future integration with real-time industrial sensor data.

### Accepting any mining dataset

Ingestion is not hardcoded to Marampa Mines' exact column layout. Any
uploaded column that doesn't match a known field is still carried through
(`extra__<name>`) rather than dropped, numeric extras become candidate
model features automatically, and the intervention-priority score falls back to whatever
real numeric operational columns exist if the usual PM/work-order
compliance fields aren't present. An optional downloadable template is
available on the Import Dataset page, and that page's "Train Models"
button runs the full training-and-selection pipeline against whatever
dataset was just imported — no CLI required. See
`docs/dataset_mapping.md` ("Accepting a differently-shaped mining
dataset").

### Synthetic demonstration track

`docs/synthetic_demo_track.md` describes a separate, clearly-labeled
**synthetic** dataset (`data/raw/synthetic/`) used only to prove out the
time-series-dependent methodology the real CMMS data cannot exercise
(lag/rolling/trend features, near-term failure-horizon labels,
chronological split, LSTM/GRU). It is never merged with or presented as
the real Data Repo. See that doc before running
`scripts/generate_synthetic_dataset.py` / `scripts/train_sequence_model.py`.

### Primary ML task (confirmed from the actual research data)

The researcher's `Data Repo/` workbooks are a **Limble CMMS export** (asset
registry, PM/CBM task templates, PM/work-order compliance snapshots, and a
narrow-scope dated labor log) — not continuous sensor telemetry, and not a
populated failure/breakdown event log. Per this project's data-integrity
rules (never fabricate a failure label the data doesn't support), the
primary task is therefore framed as **CMMS-based intervention-priority
classification**: scoring/classifying each asset's intervention priority from
static asset attributes (criticality, category, manufacturer) and current
PM/work-order compliance metrics (overdue PMs/WOs, accumulated downtime,
completion history), explained with SHAP. This is a cross-sectional
classification task rather than a time-horizon "failure within H hours"
prediction — see `docs/dataset_mapping.md` (populated once Phase 4 mapping
is implemented) for the field-level mapping once it exists.

## Thesis Context

This system is a core deliverable of the thesis, not a secondary demo. It
demonstrates the full path from a research dataset to an operationalized,
explainable predictive-maintenance decision-support tool:

```
Excel / CSV Sample Data
      -> Schema Mapping -> Validation -> Standardized Schema
      -> Leakage-Free Preprocessing -> Feature Engineering -> Label Construction
      -> Model Training & Evaluation -> Model Selection -> Serialization
      -> Inference Engine -> Intervention-Priority Classification -> SHAP Explainability
      -> Alert Engine -> Database
      -> Streamlit Decision-Support Interface -> Power BI Management Analytics
```

The prototype uses **historical/sample data**. It is explicitly not a
real-time system yet — see [`docs/real_time_scaling.md`](docs/real_time_scaling.md)
for the planned evolution path.

## Technology Stack

- **Language:** Python 3.11+
- **Data:** pandas, numpy, scipy, openpyxl
- **ML:** scikit-learn, XGBoost, Optuna, imbalanced-learn
- **Explainability:** SHAP
- **Deep learning:** TensorFlow/Keras (LSTM/GRU)
- **App:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **Database:** SQLite via SQLAlchemy
- **Testing:** pytest
- **Management analytics:** Power BI (via CSV/Parquet export, not embedded)

## Installation

### 1. Prerequisites

Python **3.11** is required (TensorFlow does not yet publish wheels for
newer interpreter versions such as 3.14). If your system's default Python
is newer, install 3.11 alongside it — it will not affect your system
default:

```powershell
winget install --id Python.Python.3.11 -e
```

### 2. Create and activate a virtual environment

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Central, non-secret settings live in `config/application.yaml` and
`config/model_config.yaml` — read via `config/settings.py`. Copy
`.env.example` to `.env` for environment-specific values (e.g. an
alternate `DATABASE_URL`):

```powershell
copy .env.example .env
```

## Dataset Placement

Place researcher-supplied Excel/CSV files under `data/raw/excel/` or
`data/raw/csv/`. Original workbooks are never modified in place; imports
create standardized copies under `data/processed/`. See
[`docs/dataset_mapping.md`](docs/dataset_mapping.md) once the ingestion
and mapping modules are in place.

## Database Initialization

```powershell
python scripts/initialize_database.py
```

## Preparing a Dataset

```powershell
python scripts/prepare_dataset.py "Data Repo\PM Spot Check.xlsx" pm_spot_check PROCESS_PLANT
```

Ingests the file, applies (or auto-suggests and saves) a column mapping,
preprocesses it, generates a feasibility report, and persists everything
(processed Parquet file + `datasets`/`machines` rows).

## Model Training

```powershell
python scripts/train_models.py pm_spot_check 1.0.0
python scripts/select_model.py pm_spot_check 1.0.0
```

`train_models.py` trains and evaluates all candidates (Logistic Regression,
Decision Tree, Random Forest, XGBoost) on a held-out test split and saves
every candidate's artifacts under `models/`. `select_model.py` picks the
best by PR-AUC (recall tie-break), writes a model-selection report, and
registers it as the active model.

```powershell
python scripts/evaluate_models.py pm_spot_check xgboost 1.0.0
python scripts/run_ablation.py pm_spot_check 1.0.0
```

`evaluate_models.py` re-evaluates a saved model and saves confusion-matrix/
ROC/PR-curve figures under `reports/figures/`. `run_ablation.py` compares
feature-group combinations (see its module docstring for why the
`statistical` group is excluded from this dataset's ablation).

## Running the Application

```powershell
streamlit run app.py
```

The app is gated behind a login screen (`src/utils/auth_ui.py`): the
first person to open it sets up the first named account, and everyone
after that signs in with their own username and password rather than a
shared password. Additional accounts (e.g. for MML colleagues) are added
from **System Information → Accounts** once signed in.

## Deploying Online

To make the app reachable from any device, anywhere (not just the local
network) -- with named logins and a real hosted database instead of the
local SQLite file -- see
[`docs/deployment_guide.md`](docs/deployment_guide.md) for a full,
step-by-step walkthrough (GitHub, Streamlit Community Cloud, and a free
hosted Postgres database).

## Running Tests

```powershell
pytest
```

## Power BI Export

```powershell
python scripts/export_powerbi_data.py
```

Outputs land in `data/exports/powerbi/`. See
[`docs/powerbi_dashboard_specification.md`](docs/powerbi_dashboard_specification.md).

## Development Phases

This system was built incrementally and verified at each phase against
real data, not synthetic fixtures alone:

1. Foundation — done
2. Database — done
3. Excel/CSV ingestion — done
4. Column mapping — done
5. Dataset feasibility — done
6. Preprocessing — done
7. Label construction — done (reframed to an intervention-priority label; see below)
8. Feature engineering — done (lag/event/frequency groups are honest no-ops for this data type)
9. Baseline models — done
10. Advanced models — done for RF+SHAP and XGBoost+Optuna; **LSTM/GRU intentionally not implemented** (no time-series data exists — see `docs/system_requirements.md`)
11. Research experiments — done for feature-group ablation (`scripts/run_ablation.py`); machine-holdout/imbalance-strategy comparisons are a follow-up
12. Model selection — done (`scripts/select_model.py`)
13. Inference engine — done
14. Streamlit application — done (all 11 pages, verified via `AppTest`)
14b. Peer-Adjusted Analysis (Component II) — done: a dedicated page (`pages/11_Peer_Adjusted_Analysis.py`, `src/analysis/peer_adjusted.py`) surfaces the peer-adjusted anomaly/rank and maintenance-pattern-cluster columns from datasets run through the offline peer-adjustment analysis (e.g. `PM_Spot_Check_Analysis_Ready.xlsx`), kept deliberately separate from the Component III intervention-priority classifier's feature set
15. Power BI support — done
16. System testing — 193 automated tests across every layer
17. Research documentation — done (see `docs/`)

## Project Limitations

- This is a research prototype, not a certified safety system. Predictions
  must be interpreted alongside qualified maintenance judgment.
- Equipment category labels (e.g. "pump", "conveyor") are only used where a
  dataset's actual metadata verifies them — the system does not fabricate
  equipment mappings for generic benchmark data.
- Real-time (IoT/SCADA/API) integration is architected for but not
  implemented in the thesis prototype; see `docs/real_time_scaling.md`.
- Failure-prediction targets are only constructed when a dataset's
  actual content supports them; datasets without defensible failure
  signals are routed toward feasibility reporting / anomaly-detection
  framing instead of fabricated labels.
