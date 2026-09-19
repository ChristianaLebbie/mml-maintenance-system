"""Phase 5 tests: feasibility analysis over a standardized DataFrame."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feasibility.analyzer import analyze
from src.feasibility.report_generator import load_report_json, save_report, save_report_json, to_markdown


@pytest.fixture()
def cmms_snapshot_df():
    return pd.DataFrame(
        {
            "machine_id": ["4212-CR-001", "4212-PU-002", "4213-CV-202"],
            "equipment_category": ["Crusher", "Pump", None],
            "manufacturer": [None, None, None],
            "criticality": [None, None, None],
            "overdue_pms": [2, 0, 5],
            "overdue_wos": [1, 0, 2],
            "total_completed_pms": [30, 10, 0],
            "total_completed_wos": [2, 0, 1],
            "downtime_minutes": [120.0, 0.0, 500.0],
            "total_cost": [None, None, None],
            "last_completed_pm": ["2025/06/04", "2025/06/10", None],
            "last_completed_wo": [None, None, "2025/05/08"],
            "failure_label": [None, None, None],
            "source_dataset": ["pm_spot_check.xlsx"] * 3,
        }
    )


def test_analyze_reports_basic_structure(cmms_snapshot_df):
    report = analyze(cmms_snapshot_df, source_dataset="pm_spot_check.xlsx")
    assert report.row_count == 3
    assert report.machine_count == 3
    assert report.records_per_machine_mean == 1.0


def test_analyze_detects_single_snapshot(cmms_snapshot_df):
    report = analyze(cmms_snapshot_df, source_dataset="pm_spot_check.xlsx")
    assert report.is_single_snapshot is True


def test_analyze_reports_no_failure_label_and_no_fabrication(cmms_snapshot_df):
    report = analyze(cmms_snapshot_df, source_dataset="pm_spot_check.xlsx")
    assert report.has_failure_label is False
    assert "binary_failure_classification" not in report.supported_tasks
    assert any("failure_label" in limitation for limitation in report.limitations)


def test_analyze_supports_intervention_priority_classification_from_extra_numeric_fields():
    # A differently-shaped mining dataset with no known CMMS field names at
    # all, but real numeric operational data via the extra__ pass-through.
    df = pd.DataFrame(
        {
            "machine_id": ["A", "B", "C"],
            "extra__shift_utilization_hours": [7.5, 2.0, 8.1],
            "extra__spare_parts_count": [1, 9, 0],
            "source_dataset": ["other_mine.csv"] * 3,
        }
    )
    report = analyze(df, source_dataset="other_mine.csv")
    assert "intervention_priority_classification" in report.supported_tasks
    assert not any("compliance metrics" in limitation for limitation in report.limitations)


def test_analyze_reports_limitation_when_truly_no_numeric_signal():
    df = pd.DataFrame(
        {
            "machine_id": ["A", "B"],
            "extra__site_notes": ["ok", "needs review"],
            "source_dataset": ["other_mine.csv"] * 2,
        }
    )
    report = analyze(df, source_dataset="other_mine.csv")
    assert "intervention_priority_classification" not in report.supported_tasks
    assert any("compliance metrics" in limitation for limitation in report.limitations)


def test_analyze_supports_intervention_priority_classification(cmms_snapshot_df):
    report = analyze(cmms_snapshot_df, source_dataset="pm_spot_check.xlsx")
    assert "intervention_priority_classification" in report.supported_tasks


def test_analyze_with_populated_failure_label():
    df = pd.DataFrame(
        {
            "machine_id": ["A", "B"],
            "overdue_pms": [1, 0],
            "failure_label": [1, 0],
            "source_dataset": ["x"] * 2,
        }
    )
    report = analyze(df, source_dataset="x")
    assert report.has_failure_label is True
    assert "binary_failure_classification" in report.supported_tasks


def test_analyze_computes_date_range(cmms_snapshot_df):
    report = analyze(cmms_snapshot_df, source_dataset="pm_spot_check.xlsx")
    assert "last_completed_pm" in report.date_range
    start, end = report.date_range["last_completed_pm"]
    assert start == "2025-06-04"
    assert end == "2025-06-10"


def test_to_markdown_includes_key_sections(cmms_snapshot_df):
    report = analyze(cmms_snapshot_df, source_dataset="pm_spot_check.xlsx")
    md = to_markdown(report)
    assert "# Dataset Feasibility Report" in md
    assert "## Target Feasibility" in md
    assert "intervention_priority_classification" in md


def test_save_report_writes_file(cmms_snapshot_df, tmp_path):
    report = analyze(cmms_snapshot_df, source_dataset="pm_spot_check.xlsx")
    out_path = save_report(report, output_dir=tmp_path)
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("# Dataset Feasibility Report")


def test_save_and_load_report_json_roundtrip(cmms_snapshot_df, tmp_path):
    report = analyze(cmms_snapshot_df, source_dataset="pm_spot_check.xlsx")
    save_report_json(report, output_dir=tmp_path)

    loaded = load_report_json("pm_spot_check.xlsx", output_dir=tmp_path)
    assert loaded is not None
    assert loaded["row_count"] == report.row_count
    assert loaded["supported_tasks"] == report.supported_tasks


def test_load_report_json_returns_none_when_missing(tmp_path):
    assert load_report_json("does_not_exist.xlsx", output_dir=tmp_path) is None
