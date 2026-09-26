"""Peer-Adjusted Analysis page: Component II of the thesis framework
(unsupervised pattern discovery and peer-adjusted anomaly detection) --
each asset compared against its peers to surface which ones are most
anomalous relative to similar equipment, plus a coarse activity-pattern
cluster label. This is a separate analytical layer from the Component III
intervention-priority classifier (Run Prediction / Explainability pages):
it is shown here, never fed into that model's feature set (see
src/analysis/peer_adjusted.py).

Only datasets that were run through the offline peer-adjustment analysis
before import (see PM_Spot_Check_Analysis_Ready.xlsx, "PROCESS_PLANT"
sheet, and docs/dataset_mapping.md) carry these columns -- most imports
won't, and this page says so plainly rather than showing an empty table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from config.settings import APP_NAME
from database.database import init_db
from src.analysis.peer_adjusted import build_peer_adjusted_view, has_peer_adjusted_data
from src.utils.auth_ui import require_login
from src.utils.streamlit_helpers import (
    list_dataset_names_cached,
    load_processed_dataset_cached,
)
from src.utils.theme import inject_global_css, render_page_header
from src.utils.viz_theme import CATEGORICAL

st.set_page_config(
    page_title=f"Peer-Adjusted Analysis - {APP_NAME}", page_icon="⛏️", layout="wide"
)
init_db()
require_login()
inject_global_css()

render_page_header("Peer-Adjusted Analysis", icon="🧭")
st.caption(
    "Component II: unsupervised pattern discovery and peer-adjusted anomaly "
    "detection -- each asset compared against similar peers, kept separate "
    "from the Component III intervention-priority classifier shown on the "
    "Run Prediction / Explainability pages."
)

names = list_dataset_names_cached()
if not names:
    st.info("No datasets imported yet. Use the Import Dataset page first.")
    st.stop()

selected_name = st.selectbox("Dataset", names, index=len(names) - 1)
df = load_processed_dataset_cached(selected_name)

if not has_peer_adjusted_data(df):
    st.info(
        "This dataset doesn't include peer-adjusted analysis columns "
        "(Hierarchy_Level, Global_Anomaly_Score, Rank_Peer_Adjusted, and "
        "similar). Import a dataset that has been run through the offline "
        "peer-adjustment analysis first -- e.g. "
        '`PM_Spot_Check_Analysis_Ready.xlsx` ("PROCESS_PLANT" sheet) -- '
        "to see this view."
    )
    st.stop()

view = build_peer_adjusted_view(df)

individual = (
    view[view["hierarchy_level"] == "Individual equipment item"]
    if "hierarchy_level" in view.columns
    else view
)
scored = (
    individual.dropna(subset=["global_anomaly_score"])
    if "global_anomaly_score" in individual.columns
    else individual.iloc[0:0]
)

kpi_cols = st.columns(4)
kpi_cols[0].metric("Assets in Dataset", len(view))
kpi_cols[1].metric("Individually Scored Assets", len(scored))
if "maintenance_pattern_cluster" in view.columns:
    kpi_cols[2].metric(
        "Pattern Clusters", view["maintenance_pattern_cluster"].nunique()
    )
if "hierarchy_level" in view.columns:
    kpi_cols[3].metric("Hierarchy Levels", view["hierarchy_level"].nunique())

if "maintenance_pattern_cluster" in view.columns:
    st.subheader("Maintenance Pattern Clusters")
    cluster_counts = view["maintenance_pattern_cluster"].value_counts().reset_index()
    cluster_counts.columns = ["maintenance_pattern_cluster", "count"]
    fig = px.bar(
        cluster_counts,
        x="count",
        y="maintenance_pattern_cluster",
        orientation="h",
        color="maintenance_pattern_cluster",
        color_discrete_sequence=CATEGORICAL,
    )
    fig.update_layout(
        showlegend=False, yaxis_title=None, margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

display_columns = [
    c
    for c in [
        "machine_id",
        "hierarchy_level",
        "maintenance_pattern_cluster",
        "pm_setup_count",
        "no_setup_zero_overdue",
        "expected_overdue_wos_peer_adjusted",
        "peer_adjusted_residual",
        "global_anomaly_score",
        "rank_raw_overdue",
        "rank_peer_adjusted",
        "rank_global_anomaly",
    ]
    if c in view.columns
]

if not scored.empty and "rank_peer_adjusted" in scored.columns:
    st.subheader("Most Anomalous vs. Peers (peer-adjusted overdue-WO ranking)")
    st.caption(
        "Rank 1 = furthest above what its peers (same equipment grouping) "
        "would predict for overdue work orders -- worth a closer look even "
        "when its own intervention-priority score looks unremarkable."
    )
    top_peer_adjusted = (
        scored.dropna(subset=["rank_peer_adjusted"])
        .sort_values("rank_peer_adjusted")
        .head(20)
    )
    st.dataframe(
        top_peer_adjusted[display_columns], use_container_width=True, hide_index=True
    )

if not scored.empty and "rank_global_anomaly" in scored.columns:
    st.subheader("Most Anomalous Overall (global anomaly ranking)")
    top_global = (
        scored.dropna(subset=["rank_global_anomaly"])
        .sort_values("rank_global_anomaly")
        .head(20)
    )
    st.dataframe(top_global[display_columns], use_container_width=True, hide_index=True)

with st.expander("Full peer-adjusted analysis table"):
    st.caption(
        'Assets at an "Area/System node" or similar non-individual '
        "hierarchy level have no peer-adjusted score (NaN) -- the "
        "analysis only compares individual equipment items against "
        "their peers."
    )
    st.dataframe(view[display_columns], use_container_width=True, hide_index=True)
