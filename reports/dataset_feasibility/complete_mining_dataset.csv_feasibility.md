# Dataset Feasibility Report: complete_mining_dataset.csv

## Dataset Structure
- Rows: 72000
- Columns: 19
- Unique machines: 25
- Mean records per machine: 2880.0
- Duplicate rows: 0

## Temporal Properties
- `last_completed_pm`: 2025-04-08 to 2026-08-24
- `last_completed_wo`: 2026-02-04 to 2026-08-22
- Single point-in-time snapshot: No

## Data Quality (missingness by column)
- `machine_id`: 0.0% missing
- `equipment_category`: 0.0% missing
- `manufacturer`: 0.0% missing
- `criticality`: 0.0% missing
- `overdue_pms`: 0.0% missing
- `overdue_wos`: 0.0% missing
- `total_completed_pms`: 0.0% missing
- `total_completed_wos`: 0.0% missing
- `downtime_minutes`: 0.0% missing
- `total_cost`: 0.0% missing
- `last_completed_pm`: 0.0% missing
- `last_completed_wo`: 0.0% missing
- `failure_label`: 0.0% missing
- `timestamp`: 0.0% missing
- `voltage`: 0.0% missing
- `rotation`: 0.0% missing
- `pressure`: 0.0% missing
- `vibration`: 0.0% missing
- `source_dataset`: 0.0% missing

## Target Feasibility
- Failure label present: Yes

### Supported modelling options
- binary_failure_classification
- maintenance_risk_classification
- time_series_analysis
- anomaly_detection_across_machines

### Limitations
- None identified.
