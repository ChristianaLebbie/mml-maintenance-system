# User Guide: Predictive Maintenance Decision-Support System

A complete, beginner-friendly guide to what this system is, how it works,
how to install and run it, how to use every page, and where its current
edges are. If you've never seen this project before, start here rather
than the more technical `docs/architecture.md`.

---

## 1. What is this project?

This is a Master's thesis research system: an **explainable machine
learning application that helps predict and explain maintenance risk for
industrial equipment at a mining site**. It was built around a real
dataset from Marampa Mines (an iron ore mine), but it is designed to
accept data from any mining operation, not just that one.

In plain terms, it answers a question a maintenance engineer or plant
manager actually asks: *"Which of my machines are most at risk of needing
attention soon, and why?"* — using the maintenance records (not sensor
readings) that most mine sites already keep in a CMMS (Computerized
Maintenance Management System, e.g. Limble, IBM Maximo, SAP PM).

It is a **research prototype**, not a certified safety system. Every
prediction is a probabilistic estimate meant to support — never replace —
a qualified engineer's judgment. That disclaimer appears throughout the
app itself.

## 2. Who is this guide for?

Anyone opening this repository for the first time: a thesis examiner, a
future maintainer, a mining-operations person curious what the tool does,
or a student picking up the project later. No machine-learning background
is assumed — unfamiliar terms (SHAP, PR-AUC, CMMS, etc.) are explained in
plain language as they come up, and again in the glossary (§10).

## 3. The problem this system solves

Mining sites run expensive, safety-critical equipment (crushers, mills,
conveyors, pumps, transformers) and typically already log maintenance
activity in a CMMS: which preventive-maintenance (PM) tasks are overdue,
how many work orders (WOs) are outstanding, how much downtime an asset has
accumulated, when it was last serviced. That data is a genuine early-
warning signal — an asset with many overdue PMs and rising downtime is at
higher risk — but it usually just sits in spreadsheets and reports,
un-modeled and un-explained.

This system turns that CMMS data into:
- a **risk score and risk level** (Normal / Watch / High Risk) per machine,
- an **explanation** of *why* a machine got that score (which factors
  drove it, and in which direction),
- **alerts** when a machine crosses into higher risk,
- and **management-level dashboards** (in-app and via Power BI export) to
  see the whole fleet at a glance.

## 4. What data actually powers this today

This is the single most important thing to understand before using the
system, because it shapes what predictions mean.

**The real research dataset** (a folder called `Data Repo/`, not shipped
in this repository for privacy — see below) is a Limble CMMS export from
an actual mine: asset registers, PM/work-order compliance snapshots, and
maintenance schedules. Critically, when this was inspected in depth (see
`docs/dataset_mapping.md`), it turned out to have:
- **no continuous sensor telemetry** (no vibration/temperature/pressure
  readings over time), and
- **no populated failure/breakdown log** (the fields exist in the
  spreadsheets but are empty).

So the original idea of "predict a machine will fail in the next 24
hours from its sensor readings" isn't something this data can honestly
support — building that would mean inventing a failure record that
doesn't exist. Per this project's core rule (**never fabricate a label,
sensor reading, or equipment mapping**), the task was instead reframed to
**maintenance-risk classification**: score each machine's risk from its
real, current PM/work-order compliance metrics. This is a legitimate,
widely-used framing in real-world predictive maintenance — it's just a
different (and, given the data, more honest) question than "when exactly
will this fail."

Three kinds of data exist in/around this repository, and they are **never
mixed**:

| Data | What it is | Where | Real or synthetic? |
|---|---|---|---|
| Real Data Repo | The actual mine's CMMS export (23 Excel workbooks) | `Data Repo/` (kept outside version control) | **Real** |
| CMMS sample/test dataset | A stand-in with the *same column shape* as the real data, but fully populated (no gaps) | `data/raw/sample_test/` | Synthetic — for testing the pipeline without messy real-world gaps |
| Synthetic sensor-telemetry dataset | Simulated hourly sensor readings with labeled failure events, used only to prove out the time-series features (lag/rolling, LSTM) the real data can't exercise | `data/raw/synthetic/` | Synthetic — clearly disclaimed, never cited as real findings |

If you don't have access to the real `Data Repo/`, use the sample/test
dataset (§7) to try the whole system end-to-end in minutes.

## 5. How the system works, end to end

```
Excel / CSV upload
      |
      v
Ingestion  (read the file, profile its columns, never modify the original)
      |
      v
Column Mapping  (match source columns to known fields; unknown columns
                 are kept, not discarded -- see "Accepting any mining
                 dataset" below)
      |
      v
Preprocessing  (clean text, coerce types, handle missing values, flag
                but don't blindly remove outliers)
      |
      v
Dataset Feasibility Report  (what can this data actually support? --
                              generated automatically, never assumed)
      |
      v
Feature Engineering  (turn raw columns into model inputs)
      |
      v
Risk Labeling  (build a maintenance-risk score from real compliance data)
      |
      v
Model Training  (4 candidate models trained and compared)
      |
      v
Model Selection  (best candidate picked by a defensible metric, not
                   just accuracy)
      |
      v
Inference  (score a machine, get a probability + risk level)
      |
      v
Explainability (SHAP)  (which factors drove this score, and how much)
      |
      v
Alerts + Database  (persist everything, raise alerts on high risk)
      |
      v
Streamlit App  +  Power BI Export
```

Every arrow above is a real, working, tested piece of code — not a
diagram of an aspiration. See `docs/architecture.md` for the fully
detailed technical version of this diagram.

### Accepting any mining dataset

The system does not require your data to look exactly like the original
mine's export. Any uploaded column that doesn't match a known field name
is still kept (renamed internally, never dropped), and if it's numeric it
automatically becomes something the model can learn from. If your data
doesn't have the exact "overdue PMs / overdue work orders / downtime"
columns the original dataset had, the risk score automatically falls back
to whatever real numeric operational columns *are* present. You can also
download an optional blank import template from the Import Dataset page
to see the recommended (but not required) column layout.

## 6. Getting started: installing and running the application

These steps assume Windows with PowerShell (the environment this project
was built and tested in).

### Step 1 — Install Python 3.11

TensorFlow (used for one optional demonstration model) doesn't yet
support very new Python versions, so this project uses Python 3.11 in its
own virtual environment, regardless of what Python version your system
already has:

```powershell
winget install --id Python.Python.3.11 -e
```

### Step 2 — Create and activate a virtual environment

From the project's root folder:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

You'll know it worked because your prompt will show `(venv)` at the start.

### Step 3 — Install the project's dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs everything listed in §11 below. It can take a while the
first time (some packages, like TensorFlow, are large).

### Step 4 — Set up configuration

```powershell
copy .env.example .env
```

Nothing in `.env` needs to be edited for a first run — it just points at
a local SQLite database file.

### Step 5 — Initialize the database

```powershell
python scripts/initialize_database.py
```

This creates an empty database file at `database/predictive_maintenance.db`
with all the tables the app needs (see §9's table list).

### Step 6 — Launch the application

```powershell
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`). At
this point the app runs, but with nothing in it yet — no dataset, no
model. Continue to §7 to populate it.

## 7. Trying it out without your own data

The fastest way to see the whole system working is with the included
synthetic sample/test dataset, which mirrors the real data's shape but is
fully populated:

```powershell
python scripts/generate_cmms_sample_dataset.py
python scripts/prepare_dataset.py "data/raw/sample_test/cmms_sample_dataset.csv" cmms_sample_test
python scripts/train_models.py cmms_sample_test 1.0.0
python scripts/select_model.py cmms_sample_test 1.0.0
streamlit run app.py
```

After this, the Overview page will show real numbers (machines, an active
model), and every page described in §8 will have something to show.
Alternatively, do the same steps through the UI itself — see §9.

If you have access to the real `Data Repo/` folder, substitute it for the
sample file, e.g.:

```powershell
python scripts/prepare_dataset.py "Data Repo\PM Spot Check.xlsx" pm_spot_check PROCESS_PLANT
python scripts/train_models.py pm_spot_check 1.0.0
python scripts/select_model.py pm_spot_check 1.0.0
```

## 8. Tour of the application, page by page

The sidebar lists every page. This is the order a first-time user would
naturally go through them.

**Overview** (the home page, `app.py`) — Landing page with the research
disclaimer and live KPI numbers: how many machines are registered, how
many alerts are active, which model is currently active, how many
predictions have been made, and how many machines are currently High
Risk. If nothing has been imported yet, it tells you to start with Import
Dataset.

**Import Dataset** — Upload any Excel/CSV file. You'll see a preview of
its rows and columns, a profile of each column (data type, % missing,
sample values), and an automatically-suggested mapping from your columns
to the system's internal fields (you can always correct the suggestion).
An optional blank template is available to download first if you want a
guide for how to lay out your own data. Clicking "Prepare Dataset" shows
a live progress bar while it preprocesses and generates a feasibility
report. If the data supports it, a "Train Models" section then appears —
clicking it shows another progress bar while it trains and activates a
model right from the browser, no command line needed.

**Dataset Feasibility** — For any dataset you've imported, this shows
what it can and can't support: row/column/machine counts, a chart of
missingness by column, and a clear list of "supported modelling tasks"
versus "limitations" — the system is deliberately honest here rather than
assuming your data supports more than it does. At the bottom, a
**Danger Zone** section lets you permanently delete a dataset and
everything derived from it (processed data, machines, predictions,
alerts, trained models, experiment reports) — this requires checking a
confirmation box *and* typing the dataset's exact name before the delete
button becomes active, specifically to prevent an accidental click from
destroying data.

**Machine Monitoring** — Pick one machine and see everything about it:
its metadata, a chart of its risk/probability history, its full
prediction history, and any alerts ever raised for it.

**Run Prediction** — Score a machine (or a whole batch, via CSV upload)
with the currently active model. You'll see the failure probability, the
risk level (Normal/Watch/High Risk), and the top factors that drove that
score.

**Explainability** — Pick any past prediction and see a plain-language
breakdown of *why* the model scored it that way — which specific factors
pushed the risk up or down, visualized as a bar chart plus a written
summary.

**Alerts** — A filterable list of every alert the system has raised
(e.g., "this machine just crossed into High Risk"). You can acknowledge,
review, or close alerts here, tracking who's looked at what.

**Prediction History** — The full log of every prediction ever made,
with trend charts and a button to export it all as CSV.

**Model Performance** — Compare every model that was trained in a given
experiment (Logistic Regression, Decision Tree, Random Forest, XGBoost):
a metrics table, a chart you can switch between metrics, confusion-matrix
heatmaps, ROC and Precision-Recall curves, and a written explanation of
why one model was selected as "best."

**System Information** — A technical summary page: the architecture
diagram, why the task is framed as maintenance-risk classification, the
active configuration, a **Clear Cache** button (the app caches dataset
lists, processed data, and the active model in memory for speed; this
clears it manually if you ever suspect a page is showing stale data), and
a bulleted list of the system's known limitations (the same ones in §12
of this guide).

## 9. Key concepts explained (glossary)

- **CMMS** — Computerized Maintenance Management System: the software a
  mine (or any industrial site) uses to schedule and track maintenance
  tasks, work orders, and asset records. Limble, SAP PM, and IBM Maximo
  are examples.
- **PM (Preventive Maintenance)** — Scheduled, routine maintenance done
  *before* something breaks (e.g., a monthly lubrication check).
- **WO (Work Order)** — A record of maintenance work that was (or needs
  to be) done, whether scheduled or reactive.
- **Overdue PMs/WOs** — Preventive-maintenance tasks or work orders that
  are past their scheduled date and haven't been completed yet — a
  classic early-warning sign of a neglected asset.
- **Risk score / Risk level** — This system's composite score of how much
  maintenance attention a machine likely needs right now, based on its
  real compliance history, bucketed into Normal / Watch / High Risk.
  Thresholds for these buckets are calculated from the data itself
  (never hand-picked).
- **SHAP** — A well-established explainability technique that tells you,
  for one specific prediction, how much each input factor pushed the
  result up or down. This is what powers the Explainability page.
- **PR-AUC (Precision-Recall Area Under Curve)** — A metric this project
  prioritizes over plain accuracy when comparing models, because the
  "High Risk" class is a minority — a model that just predicts "Normal"
  for everything would have high accuracy but be useless. PR-AUC and
  recall better reflect whether the model actually catches the
  higher-risk machines.
- **Feasibility report** — An automatically-generated, per-dataset report
  that states plainly what modelling tasks the data can and cannot
  support, so nothing downstream gets built on an assumption the data
  doesn't back up.
- **Chronological vs. stratified split** — Two ways of dividing data into
  training/validation/test sets. Chronological (train on earlier data,
  test on later data) is correct for genuine time-series data; stratified
  random (used for the real CMMS data here, since it's a single snapshot
  in time rather than a time series) preserves the same proportion of
  risk levels across the splits.
- **Synthetic demonstration data** — Fabricated data used only to prove a
  piece of methodology works (e.g., an LSTM sequence model needs
  time-series data the real dataset doesn't have). Always clearly
  labeled, and never used as evidence about real equipment.

## 10. Technologies used, and why

| Purpose | Technology | Why |
|---|---|---|
| Language | Python 3.11 | Full compatibility with every ML library below (newer Python versions don't yet have wheels for some of them) |
| Data handling | pandas, numpy, scipy | Standard for tabular data manipulation |
| Reading Excel/CSV | openpyxl, xlrd | Read real workbooks without ever modifying the original file |
| Core ML models | scikit-learn (Logistic Regression, Decision Tree, Random Forest) | Well-understood, interpretable baselines |
| Advanced ML model | XGBoost, tuned with Optuna | Typically the strongest performer on tabular data like this |
| Handling imbalanced risk classes | imbalanced-learn | Techniques for when "High Risk" is a small minority of cases |
| Explainability | SHAP | Industry-standard way to explain individual predictions |
| Deep learning (demo only) | TensorFlow/Keras (LSTM/GRU) | Used only against the synthetic time-series dataset, to demonstrate sequence modelling |
| Web application | Streamlit | Lets a data-science pipeline become a usable, interactive app quickly |
| Charts | Plotly, Matplotlib | Interactive dashboards (Plotly) and static research figures (Matplotlib) |
| Database | SQLite via SQLAlchemy | Simple, file-based database appropriate for a prototype; SQLAlchemy makes a future move to PostgreSQL straightforward |
| Model/data persistence | joblib, Parquet | Standard formats for saving trained models and processed datasets |
| Configuration | YAML, python-dotenv | Keep settings out of code, in one readable place |
| Testing | pytest | 168 automated tests across every layer of the system |
| Management dashboards | Power BI (via CSV export) | Lets non-technical stakeholders explore the results in a familiar BI tool, without embedding Power BI inside the app itself |

## 11. Current limitations

Being upfront about what this system does *not* do is as important as
what it does:

- **This is a decision-support prototype, not a certified safety
  system.** Every prediction must be read alongside real engineering
  inspection and judgment — never acted on alone.
- **The risk label is a compliance-based proxy, not a verified failure
  history.** Because the real data has no populated failure/breakdown
  log, "High Risk" means "this machine's maintenance compliance looks
  poor," not "this machine is confirmed to be about to fail." That's a
  meaningful, defensible signal — but it's a different claim than a
  model trained on confirmed failures would make.
- **No genuine time-series prediction on real data.** The real dataset is
  a single point-in-time snapshot per machine, so lag/rolling features,
  chronological validation, and the LSTM/GRU sequence model only run
  against the synthetic demonstration dataset, not the real one.
- **Equipment categories are only shown where the data itself verifies
  them.** The system never invents that a machine is a "pump" or
  "crusher" if the source data doesn't say so.
- **Sparse metadata in the real dataset.** Fields like manufacturer,
  criticality, and equipment category are frequently missing in the real
  export, which limits how much the model can lean on them.
- **No real-time data integration yet.** The system is architected to
  support future IoT/SCADA/API data sources (see
  `docs/real_time_scaling.md`), but currently only ingests static
  file uploads.
- **Single-machine research deployment.** SQLite, no authentication, and
  no multi-user concurrency controls — appropriate for a thesis
  prototype, not a production multi-site deployment.
- **No automated retraining.** Retraining currently requires a manual
  click (via the UI) or CLI command; there's no scheduled or
  trigger-based retraining pipeline yet.

## 12. Suggestions for future enhancement

Roughly in order of how much value they'd add relative to the effort:

1. **Periodic CMMS snapshots for real time-series prediction.** If the
   mine can export the same PM/work-order report on a recurring schedule
   (e.g., monthly), the system could track each machine's compliance
   *trend* over time rather than a single snapshot — enabling genuine
   chronological validation and lag/rolling features on real data.
2. **A real historian/SCADA data source.** The equipment tag lists
   already reference instrumentation (level/pressure transmitters, etc.)
   that suggests a historian may exist elsewhere. Integrating even a
   sample export from it would let the LSTM/GRU sequence model and
   frequency-domain (vibration) features run against real data instead
   of only the synthetic track.
3. **Verified failure/breakdown records.** If the mine's CMMS can supply
   populated `BREAKDOWN-INITIAL-CAUSE`/`BREAKDOWN-ROOT-CAUSE` fields (they
   exist as columns today but are empty), the system could support true
   failure classification alongside the current risk-classification
   framing.
4. **Machine-holdout and imbalance-strategy experiments.** The project
   brief calls for comparing general vs. machine-specific models and
   testing SMOTE against class-weighting; these are implemented as
   reusable building blocks but not yet run as a formal comparison.
5. **Scheduled/automated retraining.** A background job (e.g., a nightly
   or weekly scheduled task) that re-imports the latest CMMS export,
   retrains, and only activates a new model version if it beats the
   current one on held-out data.
6. **Authentication and role-based access.** Distinguishing Maintenance
   Engineer / Operations Manager / Researcher roles (per the original
   design brief) with actual login, rather than a single shared,
   unauthenticated app.
7. **Move from SQLite to PostgreSQL.** Straightforward given SQLAlchemy
   is already the data-access layer — worthwhile once multiple concurrent
   users or a production deployment is needed.
8. **Multi-site support.** Extend the dataset registry so multiple mines'
   data (and their own trained models) can coexist and be compared side
   by side, rather than one active model at a time.
9. **Direct Power BI or notification integrations.** Push alerts to
   email/Slack/Teams, and consider a live Power BI connection instead of
   periodic CSV export, once the underlying database supports concurrent
   read access well (see PostgreSQL migration above).
10. **A proper usability study.** The project brief calls for a usability
    evaluation framework with real maintenance-engineer participants —
    not yet conducted (and must never be fabricated, per this project's
    own rules).

## 13. Where to learn more

| Document | What's in it |
|---|---|
| `README.md` | Quick-reference install/run commands and current status |
| `docs/architecture.md` | The full technical data-flow diagram and design rationale |
| `docs/dataset_mapping.md` | Exactly what the real Data Repo contains, file by file, and how columns map |
| `docs/database_schema.md` | Full database table/column reference |
| `docs/synthetic_demo_track.md` | What the synthetic sensor-telemetry dataset is for and its real (not invented) results |
| `docs/real_time_scaling.md` | The planned path from static file uploads to real-time data sources |
| `docs/powerbi_dashboard_specification.md` | How to connect Power BI to the exported data |
| `docs/system_requirements.md` | A requirement-by-requirement status table |
| `docs/testing_guide.md` | How to run and interpret the automated test suite and manual UI checks |
| `docs/user_guide.md` | This document |
