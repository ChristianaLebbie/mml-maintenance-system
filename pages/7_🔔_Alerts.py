"""Alerts page: filterable alert list with acknowledge/review/close
actions."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.settings import APP_NAME
from database.database import get_session, init_db
from src.utils.auth_ui import require_login
from src.utils.theme import inject_global_css, render_page_header
from database.models import Alert, Machine
from src.utils.streamlit_helpers import list_machine_identifiers_cached

st.set_page_config(page_title=f"Alerts - {APP_NAME}", page_icon="⛏️", layout="wide")
init_db()
require_login()
inject_global_css()

render_page_header("Alerts", icon="🔔")

machine_identifiers = ["All"] + sorted(list_machine_identifiers_cached())

col1, col2 = st.columns(2)
status_filter = col1.selectbox("Status", ["All", "New", "Acknowledged", "Reviewed", "Closed"])
machine_filter = col2.selectbox("Machine", machine_identifiers)

with get_session() as session:
    query = session.query(Alert, Machine.machine_identifier).join(
        Machine, Alert.machine_id == Machine.id
    )
    if status_filter != "All":
        query = query.filter(Alert.status == status_filter)
    if machine_filter != "All":
        query = query.filter(Machine.machine_identifier == machine_filter)
    rows = query.order_by(Alert.created_at.desc()).all()

    alert_rows = [
        {
            "id": a.id,
            "machine": machine_id,
            "intervention_priority": a.intervention_priority,
            "status": a.status,
            "message": a.message,
            "created_at": a.created_at,
        }
        for a, machine_id in rows
    ]

if not alert_rows:
    st.info("No alerts match the selected filters.")
    st.stop()

st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)

st.subheader("Update Alert Status")
alert_id = st.selectbox("Alert ID", [r["id"] for r in alert_rows])
new_status = st.selectbox("New status", ["New", "Acknowledged", "Reviewed", "Closed"])

if st.button("Apply"):
    with get_session() as session:
        alert = session.get(Alert, alert_id)
        alert.status = new_status
        now = datetime.datetime.utcnow()
        if new_status == "Acknowledged":
            alert.acknowledged_at = now
        elif new_status == "Closed":
            alert.closed_at = now
    st.success(f"Alert {alert_id} updated to {new_status}.")
    st.rerun()
