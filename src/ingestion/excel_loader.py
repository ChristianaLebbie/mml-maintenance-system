"""Excel (.xlsx/.xls) data source.

Uses openpyxl in read_only mode for inspection so even very large workbooks
(hundreds of MB) can be profiled without loading everything into memory,
and never opens a workbook in write mode -- the original file is never
modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

from src.ingestion.base_source import ColumnProfile, DataSource, InspectionResult

SUPPORTED_EXTENSIONS = (".xlsx", ".xls")
# Bound how many rows `inspect()` scans directly. Some real-world exports
# (see project memory: PMList.xlsx) report a `max_row` in the millions that
# is almost entirely formatting bleed rather than real data, so profiling
# must not assume max_row is trustworthy or scan the whole thing eagerly.
MAX_INSPECT_ROWS = 5000
PREVIEW_ROWS = 10


class ExcelDataSource(DataSource):
    def __init__(self, file_path: str | Path, sheet_name: str | None = None):
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name

    def _sheet_names(self) -> list[str]:
        wb = openpyxl.load_workbook(self.file_path, read_only=True, data_only=True)
        try:
            return wb.sheetnames
        finally:
            wb.close()

    def inspect(self) -> InspectionResult:
        wb = openpyxl.load_workbook(self.file_path, read_only=True, data_only=True)
        try:
            sheet_names = wb.sheetnames
            sheet_name = self.sheet_name or sheet_names[0]
            ws = wb[sheet_name]

            rows: list[tuple] = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                rows.append(row)
                if i >= MAX_INSPECT_ROWS:
                    break
            is_partial = len(rows) > MAX_INSPECT_ROWS
            rows = rows[: MAX_INSPECT_ROWS + 1]
        finally:
            wb.close()

        if not rows:
            return InspectionResult(
                source_name=self.file_path.name,
                sheet_names=sheet_names,
                selected_sheet=sheet_name,
                row_count=0,
                column_count=0,
                columns=[],
                duplicate_row_count=0,
                preview=[],
            )

        raw_header = list(rows[0])
        data_rows = rows[1:]

        # Trim trailing columns that have neither a header nor any data --
        # a common Excel formatting-bleed artifact (styled-but-empty
        # columns) that pandas.read_excel() already excludes, so inspect()
        # must match load()'s column count rather than over-reporting it.
        last_real = len(raw_header) - 1
        while last_real >= 0:
            has_header = raw_header[last_real] is not None
            has_data = any(
                len(r) > last_real and r[last_real] is not None for r in data_rows
            )
            if has_header or has_data:
                break
            last_real -= 1
        raw_header = raw_header[: last_real + 1]
        data_rows = [row[: last_real + 1] for row in data_rows]

        header = [str(h) if h is not None else f"column_{i}" for i, h in enumerate(raw_header)]
        df = pd.DataFrame(data_rows, columns=header)

        columns = []
        for col in df.columns:
            series = df[col]
            missing = int(series.isna().sum())
            columns.append(
                ColumnProfile(
                    name=col,
                    dtype=str(_infer_dtype(series)),
                    missing_count=missing,
                    missing_pct=round(100 * missing / len(series), 2) if len(series) else 0.0,
                    sample_values=series.dropna().head(3).tolist(),
                )
            )

        duplicate_row_count = int(df.duplicated().sum())
        preview = df.head(PREVIEW_ROWS).to_dict(orient="records")

        return InspectionResult(
            source_name=self.file_path.name,
            sheet_names=sheet_names,
            selected_sheet=sheet_name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            duplicate_row_count=duplicate_row_count,
            preview=preview,
            is_partial_scan=is_partial,
        )

    def load(self) -> pd.DataFrame:
        sheet_name = self.sheet_name or self._sheet_names()[0]
        return pd.read_excel(self.file_path, sheet_name=sheet_name, engine="openpyxl")

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            problems.append(
                f"Unsupported file extension '{self.file_path.suffix}'. "
                f"Expected one of {SUPPORTED_EXTENSIONS}."
            )
            return problems
        if not self.file_path.exists():
            problems.append(f"File not found: {self.file_path}")
            return problems
        try:
            sheet_names = self._sheet_names()
        except Exception as exc:
            problems.append(f"Could not open workbook: {exc}")
            return problems
        if not sheet_names:
            problems.append("Workbook has no sheets.")
        if self.sheet_name and self.sheet_name not in sheet_names:
            problems.append(f"Sheet '{self.sheet_name}' not found. Available: {sheet_names}")
        return problems


def _infer_dtype(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    try:
        pd.to_numeric(non_null)
        return "numeric"
    except (ValueError, TypeError):
        pass
    try:
        pd.to_datetime(non_null, errors="raise", format="mixed")
        return "datetime"
    except (ValueError, TypeError):
        pass
    return "text"
