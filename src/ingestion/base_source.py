"""Common abstraction every data source (Excel, CSV, and future
database/API/IoT sources) implements, so the rest of the pipeline
(mapping, feasibility, preprocessing) never needs to know where a
dataset came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class InspectionResult:
    """What `inspect()` reports about a source, before any data is loaded
    into the pipeline. Used by the Import Dataset page and by
    src/feasibility to decide what the dataset can support."""

    source_name: str
    sheet_names: list[str] | None
    selected_sheet: str | None
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    duplicate_row_count: int
    preview: list[dict[str, Any]]
    # True when row_count/duplicate_row_count/columns were computed from a
    # bounded scan rather than the whole sheet (large-file safeguard --
    # some Limble/Excel exports report a `max_row` in the millions that is
    # mostly formatting bleed, not real data; see project memory).
    is_partial_scan: bool = False


class DataSource(ABC):
    """Abstract interface for a raw data source."""

    @abstractmethod
    def inspect(self) -> InspectionResult:
        """Profile the source without committing to loading all of it."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the (selected) tabular data into a DataFrame."""

    @abstractmethod
    def validate(self) -> list[str]:
        """Return a list of human-readable validation problems (empty if none)."""
