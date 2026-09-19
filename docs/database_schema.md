# Database Schema

SQLite for the thesis prototype (`database/predictive_maintenance.db`),
managed via SQLAlchemy ORM models in `database/models.py`. Architecture
permits migration to PostgreSQL/MySQL by changing `DATABASE_URL` only.

Note on naming: field names (`failure_probability`, `prediction_horizon`)
were kept from the project's original sensor-telemetry design for schema
stability, but the confirmed primary task is maintenance-risk
classification, not time-horizon failure prediction:
- `failure_probability` holds the model's maintenance-risk probability.
- `prediction_horizon` is nullable and unused unless a future dataset
  supports genuine time-horizon prediction.

## Tables

### `datasets`
| Column | Type | Notes |
|---|---|---|
| id | int, PK | |
| name | str | unique per import, e.g. `pm_spot_check` |
| source_type | str | `excel` or `csv` |
| source_filename | str | original filename |
| description | text | optional |
| imported_at | datetime | |
| row_count, column_count, machine_count | int | post-preprocessing counts |
| start_date, end_date | datetime | nullable |
| sampling_interval | str | nullable |
| mapping_json, quality_summary_json | text | nullable |
| status | str | Imported / Validated / Processed / Training Ready / Rejected / Archived |

### `machines`
| Column | Type | Notes |
|---|---|---|
| id | int, PK | |
| machine_identifier | str | e.g. `4212-CR-001` |
| machine_model | str | nullable |
| equipment_category | str | **nullable** -- only populated when the source dataset's own metadata verifies it, never fabricated |
| age | float | nullable |
| dataset_id | FK -> datasets.id | nullable |
| created_at | datetime | |

### `model_versions`
| Column | Type | Notes |
|---|---|---|
| id | int, PK | |
| version | str | e.g. `1.0.0` |
| model_name, model_type | str | e.g. `xgboost` |
| dataset_id | FK -> datasets.id | |
| prediction_horizon | int | nullable, unused for this task |
| threshold | float | classification threshold (default 0.5) |
| metrics_json | text | full EvaluationResult |
| artifact_path, preprocessor_path | str | paths under models/ |
| created_at | datetime | |
| active | bool | exactly one active version per dataset at a time |

### `predictions`
| Column | Type | Notes |
|---|---|---|
| id | int, PK | |
| machine_id | FK -> machines.id | |
| model_version_id | FK -> model_versions.id | |
| prediction_timestamp | datetime | |
| failure_probability | float | risk probability, see naming note above |
| predicted_class | int | 0/1 |
| risk_level | str | Normal / Watch / High Risk |
| prediction_horizon | int | nullable |
| created_at | datetime | |

### `prediction_explanations`
| Column | Type | Notes |
|---|---|---|
| id | int, PK | |
| prediction_id | FK -> predictions.id | |
| feature_name | str | |
| feature_value | float | nullable |
| shap_value | float | |
| importance_rank | int | 1 = most important |

### `alerts`
| Column | Type | Notes |
|---|---|---|
| id | int, PK | |
| prediction_id | FK -> predictions.id | |
| machine_id | FK -> machines.id | |
| risk_level | str | |
| message | text | |
| status | str | New / Acknowledged / Reviewed / Closed |
| created_at, acknowledged_at, closed_at | datetime | latter two nullable |

### `maintenance_events`
| Column | Type | Notes |
|---|---|---|
| id | int, PK | |
| machine_id | FK -> machines.id | |
| event_timestamp | datetime | |
| maintenance_type, component, notes | str/text | nullable |
| dataset_id | FK -> datasets.id | nullable |

### `experiment_runs`
| Column | Type | Notes |
|---|---|---|
| id | int, PK | |
| experiment_id | str | unique, e.g. `pm_spot_check_xgboost_1.0.0` |
| dataset, model, feature_group, validation_strategy | str | |
| parameters_json, metrics_json | text | |
| run_timestamp | datetime | |

## Session behavior

`database/database.py`'s `get_session()` context manager uses
`expire_on_commit=False`. This matters for Streamlit pages: without it,
ORM object attributes fetched inside a `with get_session() as session:`
block raise `DetachedInstanceError` when read after the block exits (this
was a real bug caught by `tests/test_streamlit_pages.py`'s `AppTest`-based
checks, not a hypothetical one).
