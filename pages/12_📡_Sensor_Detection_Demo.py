"""Sensor-Based Detection page: a supplementary, standalone demonstration
that the sensor-detection methodology (four-model comparison + SHAP) works
correctly on real, published sensor benchmark datasets.

This is deliberately NOT part of the Component I/II/III intervention-
priority framework shown elsewhere in this app. The case-study CMMS export
has no sensor telemetry, so no claim about sensor-based detection can be
made from it; this page validates the method on independent public
benchmarks (AI4I 2020, Azure Predictive Maintenance) instead, and its
results are never merged with, or presented as, findings about the
case-study equipment (see docs/dataset_mapping.md, "Synthetic track").

Results shown are pre-computed offline (see /research/sensor_demo/) rather
than trained live in this page, consistent with how this app keeps
research artifacts reproducible without retraining on every page load.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import APP_NAME
from database.database import init_db
from src.utils.auth_ui import require_login
from src.utils.theme import inject_global_css, render_page_header
from src.utils.viz_theme import CATEGORICAL

st.set_page_config(
    page_title=f"Sensor Detection Demo - {APP_NAME}", page_icon="📡", layout="wide"
)
init_db()
require_login()
inject_global_css()

render_page_header("Sensor-Based Detection", icon="📡")
st.warning(
    "**Supplementary demonstration -- not part of the Intervention-Priority "
    "analysis.** Validates the sensor-detection method on independent, "
    "published benchmark datasets, since the case-study CMMS export has no "
    "sensor telemetry to validate it on. Never merged with, or presented "
    "as, case-study findings."
)
st.caption(
    "Same four-model comparison and SHAP explainability pipeline used "
    "elsewhere in this app (Decision Tree, Logistic Regression, Random "
    "Forest, XGBoost -- selected by PR-AUC), applied here to two published "
    "sensor benchmarks."
)

RESULTS = {
    "AI4I 2020 Predictive Maintenance": {
        "description": (
            "10,000 milling-machine sensor readings (air/process "
            "temperature, rotational speed, torque, tool wear), 3.39% "
            "documented failure rate. Published benchmark (Matzka, 2020 "
            "IEEE AI4I)."
        ),
        "metrics": pd.DataFrame(
            {
                "Model": [
                    "Decision Tree",
                    "Logistic Regression",
                    "Random Forest",
                    "XGBoost (selected)",
                ],
                "F1": [0.473, 0.240, 0.641, 0.743],
                "Precision": [0.325, 0.141, 0.913, 0.722],
                "Recall": [0.871, 0.812, 0.494, 0.765],
                "ROC-AUC": [0.888, 0.883, 0.970, 0.977],
                "PR-AUC": [0.550, 0.389, 0.783, 0.810],
            }
        ),
        "shap_features": pd.DataFrame(
            {
                "feature": [
                    "Torque",
                    "Tool wear",
                    "Air temperature",
                    "Rotational speed",
                    "Process temperature",
                ],
                "mean_abs_shap": [4.18, 3.27, 2.11, 2.02, 1.05],
            }
        ),
        "selected_model": "XGBoost",
        "note": (
            "Top features (torque, tool wear) match this dataset's own "
            "documented failure modes -- evidence the model learned "
            "genuine relationships, not noise."
        ),
        "rows": "10,000",
    },
    "Azure Predictive Maintenance": {
        "description": (
            "876,100 hourly telemetry readings (voltage, rotation, "
            "pressure, vibration) across 100 machines over one year, with "
            "761 documented failure events. Published benchmark "
            "(Microsoft/Kaggle, simulated data)."
        ),
        "metrics": pd.DataFrame(
            {
                "Model": [
                    "Decision Tree",
                    "Logistic Regression",
                    "Random Forest",
                    "XGBoost (selected)",
                ],
                "F1": [0.949, 0.497, 0.985, 0.957],
                "Precision": [0.920, 0.334, 0.978, 0.922],
                "Recall": [0.981, 0.972, 0.993, 0.994],
                "ROC-AUC": [0.984, 0.994, 1.000, 1.000],
                "PR-AUC": [0.956, 0.775, 0.994, 0.995],
            }
        ),
        "shap_features": pd.DataFrame(
            {
                "feature": [
                    "Voltage (24h mean)",
                    "Days since maint. (comp4)",
                    "Days since maint. (comp2)",
                    "Days since maint. (comp3)",
                    "Rotation (24h mean)",
                ],
                "mean_abs_shap": [1.02, 0.86, 0.80, 0.77, 0.75],
            }
        ),
        "selected_model": "XGBoost",
        "note": (
            "Top features combine sensor drift with maintenance recency -- "
            "matching known component wear-out patterns. Evaluated on a "
            "time-based train/test split (not random), avoiding the "
            "leakage risk of adjacent, correlated hourly readings "
            "appearing in both sets."
        ),
        "rows": "876,100",
    },
}

dataset = st.selectbox("Select benchmark dataset", list(RESULTS.keys()))
r = RESULTS[dataset]

kpi_cols = st.columns(3)
kpi_cols[0].metric("Records", r["rows"])
kpi_cols[1].metric("Selected Model", r["selected_model"])
kpi_cols[2].metric("PR-AUC", f"{r['metrics']['PR-AUC'].max():.3f}")

st.markdown(f"**Dataset:** {r['description']}")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Model Comparison")
    st.dataframe(
        r["metrics"].style.highlight_max(subset=["PR-AUC"], color="#d4edda"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Selected model: **{r['selected_model']}** (highest PR-AUC -- the "
        "primary selection criterion under class imbalance, consistent "
        "with the rest of this system's evaluation protocol)."
    )

with col2:
    st.subheader("SHAP Feature Importance")
    fig = px.bar(
        r["shap_features"].sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="feature",
        color_discrete_sequence=CATEGORICAL,
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Mean |SHAP value|",
        yaxis_title=None,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(r["note"])

st.divider()
st.markdown(
    "**Why this page exists:** the case-study CMMS export contains no "
    "continuous sensor telemetry, so no claim about sensor-based failure "
    "detection can be made from that data. This page shows the detection "
    "methodology is implemented correctly and produces genuine, explainable "
    "results elsewhere -- a feasibility demonstration, not a finding about "
    "the case-study mining equipment."
)
