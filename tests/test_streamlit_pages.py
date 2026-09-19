"""Phase 14 tests: every Streamlit page must execute without raising, via
Streamlit's official AppTest harness (a plain HTTP GET only checks that the
static shell loads -- it never actually runs the page's Python code, which
is what caught real bugs here: a missing sys.path insert in app.py, and a
DetachedInstanceError from accessing SQLAlchemy attributes after the
session closed).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAGE_SCRIPTS = [
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "pages" / "2_Import_Dataset.py",
    PROJECT_ROOT / "pages" / "3_Dataset_Feasibility.py",
    PROJECT_ROOT / "pages" / "4_Machine_Monitoring.py",
    PROJECT_ROOT / "pages" / "5_Run_Prediction.py",
    PROJECT_ROOT / "pages" / "6_Explainability.py",
    PROJECT_ROOT / "pages" / "7_Alerts.py",
    PROJECT_ROOT / "pages" / "8_Prediction_History.py",
    PROJECT_ROOT / "pages" / "9_Model_Performance.py",
    PROJECT_ROOT / "pages" / "10_System_Information.py",
    PROJECT_ROOT / "pages" / "11_Peer_Adjusted_Analysis.py",
]


@pytest.mark.parametrize("script_path", PAGE_SCRIPTS, ids=[p.name for p in PAGE_SCRIPTS])
def test_page_runs_without_exception(script_path: Path):
    at = AppTest.from_file(str(script_path))
    at.run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
