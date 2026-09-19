"""Phase 1 smoke tests: configuration loads and resolves expected paths."""

from pathlib import Path

from config.settings import (
    APP_NAME,
    PATHS,
    PREDICTION_HORIZON_HOURS,
    PROJECT_ROOT,
    SUPPORTED_EXTENSIONS,
    get_database_url,
)


def test_app_name_is_set():
    assert isinstance(APP_NAME, str)
    assert len(APP_NAME) > 0


def test_prediction_horizon_is_positive_int():
    assert isinstance(PREDICTION_HORIZON_HOURS, int)
    assert PREDICTION_HORIZON_HOURS > 0


def test_supported_extensions_include_excel_and_csv():
    assert ".xlsx" in SUPPORTED_EXTENSIONS
    assert ".csv" in SUPPORTED_EXTENSIONS


def test_resolved_paths_are_under_project_root():
    for key, path in PATHS.items():
        assert isinstance(path, Path), key
        assert str(path).startswith(str(PROJECT_ROOT)), key


def test_database_url_has_a_default():
    url = get_database_url()
    assert url.startswith("sqlite:///")


def test_logger_can_be_created_and_logs():
    from src.utils.logger import get_logger

    logger = get_logger("tests.foundation")
    logger.info("Phase 1 logger smoke test.")
    assert logger.name == "pdm.tests.foundation"
