# Dataset Feasibility Report: sample.csv

## Dataset Structure
- Rows: 3
- Columns: 19
- Unique machines: 2
- Mean records per machine: 1.5
- Duplicate rows: 0

## Temporal Properties
- No populated date fields found.
- Single point-in-time snapshot: No

## Data Quality (missingness by column)
- `last_completed_pm`: 100.0% missing
- `last_completed_wo`: 100.0% missing
- `failure_label`: 100.0% missing
- `timestamp`: 100.0% missing
- `voltage`: 100.0% missing
- `rotation`: 100.0% missing
- `pressure`: 100.0% missing
- `vibration`: 100.0% missing
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
