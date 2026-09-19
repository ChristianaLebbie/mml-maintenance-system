"""Generates a downloadable, OPTIONAL import template so a researcher can
see the recommended column layout before uploading their own mining
dataset. Any actual column layout can still be uploaded and mapped
manually on the Import Dataset page (see src/ingestion/mapper.py) --
nothing requires using this template; it exists purely to speed up
first-time mapping for a new mine's export.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from src.ingestion.mapper import INTERNAL_FIELDS

FIELD_DESCRIPTIONS: dict[str, str] = {
    "machine_id": "Unique asset/equipment identifier (required)",
    "equipment_category": "e.g. Crusher, Conveyor, Pump, Motor, Transformer",
    "manufacturer": "Equipment manufacturer / make",
    "criticality": "Asset criticality classification (e.g. C1/C2/C3, High/Medium/Low)",
    "overdue_pms": "Count of overdue preventive-maintenance tasks",
    "overdue_wos": "Count of overdue work orders",
    "total_completed_pms": "Total completed preventive-maintenance tasks (lifetime or period)",
    "total_completed_wos": "Total completed work orders (lifetime or period)",
    "downtime_minutes": "Accumulated downtime, in minutes",
    "total_cost": "Total maintenance cost for this asset",
    "last_completed_pm": "Date of the last completed PM (YYYY-MM-DD)",
    "last_completed_wo": "Date of the last completed work order (YYYY-MM-DD)",
    "failure_label": "1 if a failure/breakdown occurred, 0 otherwise -- only include if genuinely known, never estimated",
    "timestamp": "Reading timestamp -- only for repeated time-series sensor data, e.g. one row per machine per hour",
    "voltage": "Sensor reading: voltage",
    "rotation": "Sensor reading: rotation / RPM",
    "pressure": "Sensor reading: pressure",
    "vibration": "Sensor reading: vibration",
}

EXAMPLE_ROW: dict[str, object] = {
    "machine_id": "4212-CR-001",
    "equipment_category": "Crusher",
    "manufacturer": "FLSmidth",
    "criticality": "C1 - High Criticality",
    "overdue_pms": 2,
    "overdue_wos": 1,
    "total_completed_pms": 34,
    "total_completed_wos": 5,
    "downtime_minutes": 120.5,
    "total_cost": 4500.00,
    "last_completed_pm": "2025-06-04",
    "last_completed_wo": "2025-05-20",
    "failure_label": "",
    "timestamp": "",
    "voltage": "",
    "rotation": "",
    "pressure": "",
    "vibration": "",
}


def build_template_workbook() -> bytes:
    """Return an .xlsx file (as bytes) with two sheets:
    - 'Data': header row (canonical field names) + one filled-in example
      row, ready to be replaced with real rows.
    - 'Field Reference': every field, whether it's required, and what it
      means -- including that any of these can be left blank/omitted if
      not applicable to your dataset, and that extra columns beyond this
      list are also accepted (carried through automatically).

    Any column layout can still be uploaded directly and mapped manually;
    this template is a convenience, never a requirement.
    """
    fields = list(INTERNAL_FIELDS.keys())
    data_df = pd.DataFrame([{f: EXAMPLE_ROW.get(f, "") for f in fields}])

    reference_rows = [
        {
            "field": f,
            "required": "Yes" if spec["required"] else "No",
            "description": FIELD_DESCRIPTIONS.get(f, ""),
        }
        for f, spec in INTERNAL_FIELDS.items()
    ]
    reference_rows.append(
        {
            "field": "(any other column)",
            "required": "No",
            "description": (
                "Extra columns beyond this list are accepted too -- they are "
                "carried through automatically and, if numeric, can be used "
                "as model features. You do not need to remove or rename "
                "columns that don't match a field above."
            ),
        }
    )
    reference_df = pd.DataFrame(reference_rows)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data_df.to_excel(writer, sheet_name="Data", index=False)
        reference_df.to_excel(writer, sheet_name="Field Reference", index=False)
    return buffer.getvalue()
