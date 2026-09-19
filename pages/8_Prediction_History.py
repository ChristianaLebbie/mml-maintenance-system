"""Prediction History page: full prediction log with trend charts and CSV
export."""

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
from database.models import Machine, Prediction

st.set_page_config(page_title=f"Prediction History - {APP_NAME}", layout="wide")
init_db()
require_login()

st.title("Prediction History")

with get_session() as session:
    rows = (
        session.query(Prediction, Machine.machine_identifier)
        .join(Machine, Prediction.machine_id == Machine.id)
        .order_by(Prediction.prediction_timestamp.desc())
        .all()
    )
    data = [
        {
            "prediction_timestamp": p.prediction_timestamp,
            "machine": machine_id,
            "failure_probability": p.failure_probability,
            "intervention_priority": p.intervention_priority,
            "model_version_id": p.model_version_id,
        }
        for p, machine_id in rows
    ]

if not data:
    st.info("No predictions recorded yet.")
    st.stop()

df = pd.DataFrame(data)

col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", len(df))
col2.metric("High Priority", int((df["intervention_priority"] == "High Priority").sum()))
col3.metric("Watch", int((df["intervention_priority"] == "Watch").sum()))

fig1 = px.histogram(
    df, x="intervention_priority", title="Intervention priority distribution", color="intervention_priority"
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(
    df.sort_values("prediction_timestamp"),
    x="prediction_timestamp",
    y="failure_probability",
    color="machine",
    title="Failure probability trend by machine",
)
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(df, use_container_width=True)
st.download_button(
    "Download as CSV", df.to_csv(index=False).encode("utf-8"), file_name="prediction_history.csv"
)
