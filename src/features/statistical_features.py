"""Statistical/derived feature group: ratios between compliance metrics
that carry more signal than the raw counts alone (e.g. an asset with 2
overdue PMs out of 50 completed is very different from 2 out of 2)."""

from __future__ import annotations

import pandas as pd


def build_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    if {"overdue_pms", "total_completed_pms"}.issubset(df.columns):
        features["pm_overdue_ratio"] = df["overdue_pms"] / (df["total_completed_pms"] + 1)

    if {"overdue_wos", "total_completed_wos"}.issubset(df.columns):
        features["wo_overdue_ratio"] = df["overdue_wos"] / (df["total_completed_wos"] + 1)

    if {"downtime_minutes", "total_completed_wos"}.issubset(df.columns):
        features["downtime_per_completed_wo"] = df["downtime_minutes"] / (
            df["total_completed_wos"] + 1
        )

    if {"total_completed_pms", "overdue_pms"}.issubset(df.columns):
        denom = df["total_completed_pms"] + df["overdue_pms"]
        features["pm_compliance_rate"] = (df["total_completed_pms"] / denom.replace(0, 1)).where(
            denom > 0, 1.0
        )

    return features
