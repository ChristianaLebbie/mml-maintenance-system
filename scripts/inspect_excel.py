"""Developer utility: inspect a workbook's sheets/columns before deciding
on a mapping. Prints the same InspectionResult the Import Dataset page
uses, without loading the whole file or touching the database.

Usage:
    python scripts/inspect_excel.py "<path to file>" [sheet_name]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.excel_loader import ExcelDataSource  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    file_path = Path(sys.argv[1])
    sheet_name = sys.argv[2] if len(sys.argv) > 2 else None

    source = ExcelDataSource(file_path, sheet_name=sheet_name)
    problems = source.validate()
    if problems:
        print("Validation problems:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    result = source.inspect()
    print(f"File: {result.source_name}")
    print(f"Sheets: {result.sheet_names}")
    print(f"Selected sheet: {result.selected_sheet}")
    print(
        f"Rows: {result.row_count}"
        f"{' (partial scan -- file is larger)' if result.is_partial_scan else ''}"
    )
    print(f"Columns: {result.column_count}")
    print(f"Duplicate rows: {result.duplicate_row_count}")
    print("\nColumn profile:")
    for c in result.columns:
        print(f"  {c.name:40s} dtype={c.dtype:10s} missing={c.missing_pct:6.2f}% sample={c.sample_values}")


if __name__ == "__main__":
    main()
