"""Run Prediction page.

Mode A: select an existing machine from a prepared dataset and score it
with the active model. Mode B: upload a small CSV of machines with the
standardized columns and score all of them. Both modes go through
src.services.prediction_service so results are always persisted the same
way regardless of entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.settings import APP_NAME
from database.database import init_db
from src.utils.auth_ui import require_login
from src.utils.theme import inject_global_css, render_page_header
from src.services.prediction_service import run_and_save_predictions
from src.utils.streamlit_helpers import (
    list_dataset_names_cached,
    load_active_model_cached,
    load_processed_dataset_cached,
    make_progress_callback,
)

st.set_page_config(page_title=f"Run Prediction - {APP_NAME}", page_icon="⛏️", layout="wide")
init_db()
require_login()
inject_global_css()

render_page_header("Run Prediction", icon="🎯")
st.caption(
    "Component III (pipeline demonstration): this classifier is trained on a "
    "compliance-derived proxy label, not an engineering-validated intervention "
    "decision -- treat its output as a starting point for review, not a "
    "verified finding, until the labeling pilot with MML engineering staff is "
    "complete (see System Information)."
)

try:
    loaded = load_active_model_cached()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Active model: {loaded.model_type} (version {loaded.version})")

mode = st.radio("Input mode", ["Select existing machine", "Upload data"])

dataset_names = list_dataset_names_cached()

if mode == "Select existing machine":
    if not dataset_names:
        st.info("No prepared datasets available. Use Import Dataset first.")
        st.stop()

    dataset_name = st.selectbox("Dataset", dataset_names)
    df = load_processed_dataset_cached(dataset_name)
    machine_id = st.selectbox("Machine", sorted(df["machine_id"].dropna().unique()))

    if st.button("Run Prediction"):
        row = df[df["machine_id"] == machine_id].head(1)
        background = df.sample(min(100, len(df)), random_state=42)
        with st.spinner("Scoring machine..."):
            results = run_and_save_predictions(row, loaded, background=background)
        result = results[0]

        st.metric("Failure Probability", f"{result.failure_probability:.3f}")
        st.metric("Intervention Priority", result.intervention_priority)
        st.write("**Top Contributing Factors:**")
        st.dataframe(pd.DataFrame(result.top_factors), use_container_width=True)

else:
    st.caption(
        "Upload a CSV with at least a `machine_id` column, plus any of: "
        "equipment_category, manufacturer, criticality, total_completed_pms, "
        "total_completed_wos, total_cost, last_completed_pm, last_completed_wo."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        upload_df = pd.read_csv(uploaded)
        st.dataframe(upload_df.head(10), use_container_width=True)

        if "machine_id" not in upload_df.columns:
            st.error("Uploaded file is missing a required 'machine_id' column.")
            st.stop()

        if st.button("Run Prediction on Uploaded Data"):
            background = None
            if dataset_names:
                background_source = load_processed_dataset_cached(dataset_names[0])
                background = background_source.sample(
                    min(100, len(background_source)), random_state=42
                )

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            callback = make_progress_callback(progress_bar, status_text)
            try:
                results = run_and_save_predictions(
                    upload_df, loaded, background=background, progress_callback=callback
                )
            finally:
                progress_bar.empty()
                status_text.empty()

            result_df = pd.DataFrame(
                [
                    {
                        "machine_id": r.machine_id,
                        "failure_probability": r.failure_probability,
                        "intervention_priority": r.intervention_priority,
                    }
                    for r in results
                ]
            )
            st.dataframe(result_df, use_container_width=True)
