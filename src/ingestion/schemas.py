"""Standardized internal schema.

Every data source (Excel, CSV, and future database/API/IoT sources) is
normalized into this shape before entering preprocessing, feature
engineering, or the ML pipeline. Only `dataset_id`, `machine_id`, and
`source_dataset` are mandatory -- everything else is optional because not
every dataset carries every field (see docs/dataset_mapping.md).

`attributes` deliberately stays an open dict rather than a fixed set of
sensor columns: the confirmed primary task (CMMS-based intervention-priority
classification, see README) is driven by asset attributes and PM/work-order
compliance metrics (e.g. overdue_pms, downtime_minutes), not fixed sensor
channels like the voltage/vibration/pressure fields a telemetry dataset
would use. A future sensor or time-series source can populate `attributes`
with its own fields without changing this schema.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StandardizedRecord:
    dataset_id: str
    machine_id: str
    source_dataset: str
    timestamp: datetime.datetime | None = None
    failure_label: int | None = None
    failure_type: str | None = None
    maintenance_event: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


REQUIRED_FIELDS = ("dataset_id", "machine_id", "source_dataset")
