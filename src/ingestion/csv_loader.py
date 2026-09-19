"""CSV data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.base_source import ColumnProfile, DataSource, InspectionResult
from src.ingestion.excel_loader import MAX_INSPECT_ROWS, PREVIEW_ROWS, _infer_dtype

SUPPORTED_EXTENSIONS = (".csv",)


class CSVDataSource(DataSource):
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def inspect(self) -> InspectionResult:
        df = pd.read_csv(self.file_path, nrows=MAX_INSPECT_ROWS + 1)
        is_partial = len(df) > MAX_INSPECT_ROWS
        df = df.iloc[:MAX_INSPECT_ROWS]

        columns = []
        for col in df.columns:
            series = df[col]
            missing = int(series.isna().sum())
            columns.append(
                ColumnProfile(
                    name=str(col),
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
            sheet_names=None,
            selected_sheet=None,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            duplicate_row_count=duplicate_row_count,
            preview=preview,
            is_partial_scan=is_partial,
        )

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)

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
            pd.read_csv(self.file_path, nrows=1)
        except Exception as exc:
            problems.append(f"Could not parse CSV: {exc}")
        return problems
