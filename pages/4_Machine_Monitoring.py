"""Machine Monitoring page: select a machine, view its metadata,
prediction history, and alert history."""

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
from database.models import Alert, Machine, Prediction
from src.utils.streamlit_helpers import list_machine_identifiers_cached

st.set_page_config(page_title=f"Machine Monitoring - {APP_NAME}", layout="wide")
init_db()
require_login()

st.title("Machine Monitoring")

identifiers = list_machine_identifiers_cached()

if not identifiers:
    st.info("No machines registered yet. Import a dataset first.")
    st.stop()

selected_id = st.selectbox("Machine", identifiers)

with get_session() as session:
    machine = session.query(Machine).filter_by(machine_identifier=selected_id).first()
    machine_info = {
        "machine_identifier": machine.machine_identifier,
        "equipment_category": machine.equipment_category,
        "machine_model": machine.machine_model,
        "created_at": str(machine.created_at),
    }
    machine_pk = machine.id

    predictions = (
        session.query(Prediction)
        .filter_by(machine_id=machine_pk)
        .order_by(Prediction.prediction_timestamp)
        .all()
    )
    prediction_rows = [
        {
            "prediction_timestamp": p.prediction_timestamp,
            "failure_probability": p.failure_probability,
            "intervention_priority": p.intervention_priority,
            "model_version_id": p.model_version_id,
        }
        for p in predictions
    ]

    alerts = session.query(Alert).filter_by(machine_id=machine_pk).order_by(Alert.created_at.desc()).all()
    alert_rows = [
        {
            "created_at": a.created_at,
            "intervention_priority": a.intervention_priority,
            "status": a.status,
            "message": a.message,
        }
        for a in alerts
    ]

st.subheader("Machine Metadata")
st.json(machine_info)

st.subheader("Prediction History")
if prediction_rows:
    pred_df = pd.DataFrame(prediction_rows)
    fig = px.line(
        pred_df,
        x="prediction_timestamp",
        y="failure_probability",
        markers=True,
        title="Failure probability over time",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pred_df, use_container_width=True)
else:
    st.info("No predictions recorded yet for this machine.")

st.subheader("Alert History")
if alert_rows:
    st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)
else:
    st.info("No alerts recorded yet for this machine.")
