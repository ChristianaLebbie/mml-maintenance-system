"""Phase 3 tests: ExcelDataSource against small synthetic workbooks that
mirror the real Data Repo's shape (asset ID, PM compliance metrics,
completion dates) without depending on the large external files."""

from __future__ import annotations

import openpyxl
import pytest

from src.ingestion.excel_loader import ExcelDataSource


@pytest.fixture()
def sample_workbook(tmp_path):
    path = tmp_path / "sample_assets.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Assets"
    ws.append(["Asset ID", "Category", "Overdue PMs", "Downtime", "Last Completed PM"])
    ws.append(["4212-CR-001", "Crusher", 2, 45.5, "2025-01-03"])
    ws.append(["4212-PU-002", "Pump", 0, 0, "2025-01-10"])
    ws.append(["4212-CR-001", "Crusher", 2, 45.5, "2025-01-03"])  # exact duplicate
    ws.append(["4213-CV-202", None, 5, 120.0, None])

    ws2 = wb.create_sheet("Notes")
    ws2.append(["free text sheet"])

    wb.save(path)
    return path


def test_inspect_reports_sheets_and_columns(sample_workbook):
    source = ExcelDataSource(sample_workbook, sheet_name="Assets")
    result = source.inspect()

    assert result.sheet_names == ["Assets", "Notes"]
    assert result.selected_sheet == "Assets"
    assert result.row_count == 4
    assert result.column_count == 5
    assert {c.name for c in result.columns} == {
        "Asset ID",
        "Category",
        "Overdue PMs",
        "Downtime",
        "Last Completed PM",
    }
    assert result.is_partial_scan is False


def test_inspect_detects_duplicate_row(sample_workbook):
    source = ExcelDataSource(sample_workbook, sheet_name="Assets")
    result = source.inspect()
    assert result.duplicate_row_count == 1


def test_inspect_reports_missingness(sample_workbook):
    source = ExcelDataSource(sample_workbook, sheet_name="Assets")
    result = source.inspect()
    category_col = next(c for c in result.columns if c.name == "Category")
    assert category_col.missing_count == 1


def test_default_sheet_is_first_when_unspecified(sample_workbook):
    source = ExcelDataSource(sample_workbook)
    result = source.inspect()
    assert result.selected_sheet == "Assets"


def test_load_returns_dataframe(sample_workbook):
    source = ExcelDataSource(sample_workbook, sheet_name="Assets")
    df = source.load()
    assert len(df) == 4
    assert list(df.columns) == [
        "Asset ID",
        "Category",
        "Overdue PMs",
        "Downtime",
        "Last Completed PM",
    ]


def test_validate_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("not excel")
    source = ExcelDataSource(bad_file)
    problems = source.validate()
    assert any("Unsupported file extension" in p for p in problems)


def test_validate_rejects_missing_file(tmp_path):
    source = ExcelDataSource(tmp_path / "does_not_exist.xlsx")
    problems = source.validate()
    assert any("not found" in p for p in problems)


def test_validate_rejects_unknown_sheet(sample_workbook):
    source = ExcelDataSource(sample_workbook, sheet_name="NoSuchSheet")
    problems = source.validate()
    assert any("not found" in p for p in problems)


def test_validate_passes_for_good_file(sample_workbook):
    source = ExcelDataSource(sample_workbook, sheet_name="Assets")
    assert source.validate() == []


def test_original_workbook_is_never_modified(sample_workbook):
    before = sample_workbook.read_bytes()
    source = ExcelDataSource(sample_workbook, sheet_name="Assets")
    source.inspect()
    source.load()
    source.validate()
    after = sample_workbook.read_bytes()
    assert before == after
