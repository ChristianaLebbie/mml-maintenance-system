"""Import an Excel/CSV file, map it to the standardized schema, preprocess
it, generate a feasibility report, and persist it (processed file +
database rows).

Usage:
    python scripts/prepare_dataset.py "<path to file>" <dataset_name> [sheet_name]

Example (against the researcher's real data):
    python scripts/prepare_dataset.py "Data Repo/PM Spot Check.xlsx" pm_spot_check PROCESS_PLANT
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.database import init_db  # noqa: E402
from src.services.dataset_service import import_and_prepare  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_name = sys.argv[2]
    sheet_name = sys.argv[3] if len(sys.argv) > 3 else None

    init_db()
    result = import_and_prepare(file_path, dataset_name, sheet_name=sheet_name)

    print(f"Dataset prepared: {result.dataset_name} (id={result.dataset_id})")
    print(f"  Processed file: {result.processed_path}")
    print(f"  Feasibility report: {result.feasibility_report_path}")
    print(f"  Machines: {result.machine_count}")
    print(f"  Duplicates dropped: {result.preprocessing_report.duplicates_dropped}")
    print(f"  Supported tasks: {result.feasibility_report.supported_tasks}")
    print(f"  Limitations: {result.feasibility_report.limitations}")


if __name__ == "__main__":
    main()
