# Dataset Mapping

## Source: Data Repo (Marampa Mines Limble CMMS export)

23 Excel workbooks were inspected directly with openpyxl (read-only, no
files modified) before any modeling decisions were made. Summary of what
each file actually is:

| File | Content |
|---|---|
| `EQUIPTMENT LIST.xlsx`, `INSPECTOR Assets-2025-01-20.xlsx`, `LIMBLE ASSET LIST.xlsx` | Asset registry: real equipment taxonomy (crusher, SAG/ball mill, conveyor, pump, motor, transformer, compressor, screen, cyclone, level/pressure/flow transmitters) |
| `PMs.xlsx`, `ELECTRICAL PMs 02.xlsx`, `PMTemplateLimble (1).xlsx`, `TRT_PMLists.xlsx`, `CBML.xlsx` | PM/CBM task templates -- recurring checklists including condition-based-monitoring thresholds (instructions, not logged readings) |
| `PM Spot Check.xlsx`, `PMList.xlsx` | PM/work-order compliance **snapshots**: one row per PM-task-per-asset with Overdue PMs/WOs, Total Completed PMs/WOs, Last Completed dates, accumulated Downtime, Total Cost. Single point-in-time cross-section, not a time series. `PMList.xlsx`'s 359MB size is mostly Excel formatting bleed (`max_row` in the millions vs. a few thousand real rows) -- always scan for actual non-empty rows, never trust `ws.max_row` directly. |
| `Work Labor List.xlsx` | The only genuinely dated (2022-11-01 to 2025-01-17), multi-year transactional log -- but scoped to workshop tools (welding machines), not the process-plant fleet. Numeric `Work Type` codes decoded by cross-referencing free-text descriptions (not guessed): code 1 = scheduled PM, 21/24/26 = training/housekeeping/procurement, 5/6/27 = genuine reactive/corrective repair language (small minority of rows). |
| `IML-Export - PMs.xlsx` | Has `BREAKDOWN-INITIAL-CAUSE`/`BREAKDOWN-ROOT-CAUSE` columns, but they are 100% blank across all 211 rows -- template placeholders, not real breakdown records. |
| `EmployeeExport.xlsx` | Real employee roster (matches "Assigned To" IDs elsewhere) |
| `employee_dataset.xlsx` | Appears synthetic (sequential "Employee_0", "Employee_1" names, rounded values) -- not used as real data |

## Column mapping (example: `PM Spot Check.xlsx`, sheet `PROCESS_PLANT`)

Auto-suggested by `src/ingestion/mapper.py:suggest_mapping()` and confirmed
via the Import Dataset page; persisted in `config/datasets.yaml`:

| Internal field | Source column | Required |
|---|---|---|
| machine_id | Asset Name | yes |
| equipment_category | Category | no |
| manufacturer | Manufacturer | no |
| criticality | Criticality Classification | no |
| overdue_pms | Overdue PMs | no |
| overdue_wos | Overdue WOs | no |
| total_completed_pms | Total Completed PMs | no |
| total_completed_wos | Total Completed WOs | no |
| downtime_minutes | Total time spent on PMs in minutes | no |
| total_cost | *(not present in this sheet)* | no |
| last_completed_pm | Last Completed PM | no |
| last_completed_wo | Last Completed WO | no |
| failure_label | *(not present)* | no |

Real missingness observed (4426 rows): `total_cost` 100%, `failure_label`
100%, `equipment_category` 88.4%, `last_completed_wo` 82.0%, `manufacturer`
76.9%, `criticality` 69.5%, `last_completed_pm` 49.9%. This sparsity is why
the trained model relies most heavily on `total_completed_pms` and
`days_since_last_completed_pm` rather than the (frequently missing)
categorical attributes -- confirmed via SHAP, not assumed.

Every new dataset must go through the same inspect-before-mapping workflow
(`ExcelDataSource.inspect()` / `CSVDataSource.inspect()`) -- never assume a
new file shares the same column names or structure.

## Accepting a differently-shaped mining dataset (2026-08-26)

The system is not limited to Marampa Mines' exact column layout. Three
things generalize ingestion to any mining site's export:

1. **Extra-column pass-through** (`src/ingestion/mapper.py:apply_mapping`):
   any source column that doesn't match a known field is still carried
   through as `extra__<sanitized_name>` rather than dropped. It shows up
   in previews, feasibility reports, and -- if numeric --
   `src/features/raw_features.py:build_extra_numeric_features` makes it a
   candidate model feature automatically (registered as feature group
   `"extra"`, included by default).
2. **Risk-score fallback** (`src/labels/failure_labels.py:resolve_risk_weights`):
   if none of the known CMMS compliance columns (overdue PMs/WOs,
   downtime) are populated, the composite risk score falls back to
   equally-weighting whatever numeric `extra__` columns are present. Still
   a real, auditable score derived from the uploaded data -- never a
   fabricated label. Whichever columns actually fed the score are excluded
   from model features (mirroring the original label-formula-exclusion
   rule), so the model can't trivially "predict" its own label inputs.
3. **Optional downloadable template** (`src/ingestion/template.py`, wired
   into the Import Dataset page): shows the recommended column layout and
   field meanings, but using it is never required -- any layout can be
   mapped manually.

Net effect: a different mine's CMMS export, or a dataset with entirely
different operational metrics, can be uploaded, mapped (auto-suggested
where names match, manual otherwise), and trained on end-to-end from the
Import Dataset page's "Train Models" button -- without a code change,
as long as it has a machine identifier and at least one real numeric
operational column.
