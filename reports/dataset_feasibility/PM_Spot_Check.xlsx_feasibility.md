# Dataset Feasibility Report: PM Spot Check.xlsx

## Dataset Structure
- Rows: 4425
- Columns: 14
- Unique machines: 4425
- Mean records per machine: 1.0
- Duplicate rows: 0

## Temporal Properties
- `last_completed_pm`: 2025-02-07 to 2025-06-16
- `last_completed_wo`: 2025-01-23 to 2025-06-16
- Single point-in-time snapshot: Yes

## Data Quality (missingness by column)
- `failure_label`: 100.0% missing
- `last_completed_wo`: 82.03% missing
- `last_completed_pm`: 49.9% missing
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
- `source_dataset`: 0.0% missing

## Target Feasibility
- Failure label present: No

### Supported modelling options
- maintenance_risk_classification
- anomaly_detection_across_machines

### Limitations
- No populated failure_label column -- binary/multi-class failure classification is not supported without fabricating a label.
- Data is a single point-in-time snapshot (~1 row per machine) -- time-series / near-term failure-horizon prediction is not supported unless multiple dated snapshots of the same machines are supplied.
