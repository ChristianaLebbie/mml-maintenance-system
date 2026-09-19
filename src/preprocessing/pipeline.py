"""Orchestrates cleaning, duplicate handling, and missing-value filling
into one reusable, reportable step. Scaling is intentionally excluded from
this function -- it must only be fit on a training split (see
src/preprocessing/scaling.py), which this dataset-level pipeline has no
knowledge of.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.preprocessing.cleaning import clean, drop_duplicate_rows
from src.preprocessing.missing_values import fill_missing, summarize_missing
from src.preprocessing.outliers import summarize_outliers
from src.utils.logger import get_logger

logger = get_logger("preprocessing.pipeline")


@dataclass
class PreprocessingReport:
    rows_before: int
    rows_after: int
    duplicates_dropped: int
    missingness_before: dict[str, float]
    missingness_after: dict[str, float]
    outlier_counts: dict[str, int]


def run_preprocessing(df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
    rows_before = len(df)
    missingness_before = summarize_missing(df)

    cleaned = clean(df)
    deduped, duplicates_dropped = drop_duplicate_rows(cleaned)
    outlier_counts = summarize_outliers(deduped)
    filled = fill_missing(deduped)

    report = PreprocessingReport(
        rows_before=rows_before,
        rows_after=len(filled),
        duplicates_dropped=duplicates_dropped,
        missingness_before=missingness_before,
        missingness_after=summarize_missing(filled),
        outlier_counts=outlier_counts,
    )
    logger.info(
        f"Preprocessing: {rows_before} -> {report.rows_after} rows "
        f"({duplicates_dropped} duplicates dropped)"
    )
    return filled, report
