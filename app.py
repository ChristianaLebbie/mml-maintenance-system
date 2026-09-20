"""Entry point for the Explainable Intervention Prioritization System.

Run with: streamlit run app.py

Page content lives under pages/ (Streamlit's multipage convention);
this file renders the landing/overview shell (header banner + summary
metrics).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from config.settings import APP_NAME
from database.database import get_session, init_db
from src.utils.auth_ui import require_login
from src.utils.theme import inject_global_css, render_page_header
from database.models import Alert, Machine, ModelVersion, Prediction
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()
require_login()
inject_global_css()


def main() -> None:
    render_page_header(
        APP_NAME,
        icon="⛏️",
        subtitle="Explainable, peer-adjusted maintenance intervention prioritization for mining equipment.",
    )

    with get_session() as session:
        machine_count = session.query(Machine).count()
        active_alerts = session.query(Alert).filter(Alert.status == "New").count()
        active_model = session.query(ModelVersion).filter_by(active=True).first()
        prediction_count = session.query(Prediction).count()
        high_priority_count = (
            session.query(Prediction.machine_id)
            .filter(Prediction.intervention_priority == "High Priority")
            .distinct()
            .count()
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registered Machines", machine_count)
    col2.metric("Active Alerts", active_alerts)
    col3.metric(
        "Active Model",
        f"{active_model.model_type} v{active_model.version}" if active_model else "None",
    )
    col4.metric("Predictions Generated", prediction_count)

    col5, col6 = st.columns(2)
    col5.metric("High-Priority Predictions", high_priority_count)
    col6.metric("Task Type", "Intervention-Priority Classification")

    if machine_count == 0:
        st.warning(
            "No dataset has been imported yet. Start with **Import Dataset** "
            "in the sidebar.",
            icon="⚠️",
        )
    elif active_model is None:
        st.warning(
            "No active model has been selected yet. Run "
            "`scripts/train_models.py` and `scripts/select_model.py`.",
            icon="⚠️",
        )

    logger.info("Application homepage rendered.")


if __name__ == "__main__":
    main()
