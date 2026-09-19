"""Explainability page: pick a saved prediction and show its stored SHAP
local explanation (feature, value, contribution direction) in plain
language."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import APP_NAME
from database.database import get_session, init_db
from src.utils.auth_ui import require_login
from database.models import Machine, Prediction, PredictionExplanation

st.set_page_config(page_title=f"Explainability - {APP_NAME}", layout="wide")
init_db()
require_login()

st.title("Explainability")
st.caption(
    "Component III (pipeline demonstration): these SHAP explanations describe "
    "the classifier's own reasoning, which is only as sound as its proxy "
    "label -- useful for understanding the model, not yet a validated "
    "engineering judgment (see System Information)."
)

with get_session() as session:
    predictions = (
        session.query(Prediction, Machine.machine_identifier)
        .join(Machine, Prediction.machine_id == Machine.id)
        .order_by(Prediction.prediction_timestamp.desc())
        .limit(200)
        .all()
    )
    options = {
        f"{p.prediction_timestamp} | {machine_id} | {p.intervention_priority} ({p.failure_probability:.3f})": p.id
        for p, machine_id in predictions
    }

if not options:
    st.info("No predictions recorded yet. Use Run Prediction first.")
    st.stop()

selected_label = st.selectbox("Prediction", list(options.keys()))
prediction_id = options[selected_label]

with get_session() as session:
    prediction = session.get(Prediction, prediction_id)
    machine = session.get(Machine, prediction.machine_id)
    explanations = (
        session.query(PredictionExplanation)
        .filter_by(prediction_id=prediction_id)
        .order_by(PredictionExplanation.importance_rank)
        .all()
    )
    explanation_rows = [
        {
            "feature": e.feature_name,
            "value": e.feature_value,
            "shap_value": e.shap_value,
            "direction": "increases priority" if e.shap_value > 0 else "decreases priority",
            "rank": e.importance_rank,
        }
        for e in explanations
    ]

col1, col2, col3 = st.columns(3)
col1.metric("Machine", machine.machine_identifier)
col2.metric("Failure Probability", f"{prediction.failure_probability:.3f}")
col3.metric("Intervention Priority", prediction.intervention_priority)

if explanation_rows:
    df = pd.DataFrame(explanation_rows)
    fig = px.bar(
        df,
        x="shap_value",
        y="feature",
        color="direction",
        orientation="h",
        title="Top contributing factors (SHAP)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

    st.subheader("Plain-language summary")
    for row in explanation_rows[:3]:
        st.write(
            f"- **{row['feature']}** (value: {row['value']}) {row['direction']} "
            f"for this machine (SHAP contribution: {row['shap_value']:.4f})."
        )
else:
    st.info("No SHAP explanation was stored for this prediction (no background sample was available).")
