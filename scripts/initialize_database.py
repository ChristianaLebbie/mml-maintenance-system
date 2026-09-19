"""Create the SQLite database and all tables if they don't already exist.

Usage:
    python scripts/initialize_database.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.database import get_database_url, init_db  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.initialize_database")


def main() -> None:
    url = get_database_url()
    if url.startswith("sqlite:///"):
        db_path = Path(url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db()
    logger.info(f"Database initialized at {url}")
    print(f"Database initialized at {url}")


if __name__ == "__main__":
    main()
