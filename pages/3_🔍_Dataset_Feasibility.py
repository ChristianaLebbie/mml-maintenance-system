"""Dataset Feasibility page: KPI cards and a missingness chart built from
the saved feasibility JSON, plus the full narrative Markdown report and
the registry-level summary from the database."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from config.settings import APP_NAME, PATHS
from database.database import get_session, init_db
from src.utils.auth_ui import require_login
from src.utils.theme import inject_global_css, render_page_header
from database.models import Dataset
from src.feasibility.report_generator import load_report_json
from src.services.dataset_service import delete_dataset
from src.utils.streamlit_helpers import clear_all_caches
from src.utils.viz_theme import sequential_blue_scale

st.set_page_config(page_title=f"Dataset Feasibility - {APP_NAME}", page_icon="⛏️", layout="wide")
init_db()
require_login()
inject_global_css()

render_page_header("Dataset Feasibility", icon="🔍")

with get_session() as session:
    datasets = session.query(Dataset).order_by(Dataset.imported_at.desc()).all()
    dataset_rows = [
        {
            "name": d.name,
            "source_filename": d.source_filename,
            "row_count": d.row_count,
            "machine_count": d.machine_count,
            "status": d.status,
        }
        for d in datasets
    ]

if not dataset_rows:
    st.info("No datasets imported yet. Use the Import Dataset page first.")
    st.stop()

st.subheader("Dataset Registry")
st.dataframe(dataset_rows, use_container_width=True)

names = [d["name"] for d in dataset_rows]
selected_name = st.selectbox("Select a dataset to view its feasibility report", names)
selected = next(d for d in dataset_rows if d["name"] == selected_name)

st.subheader(f"Feasibility Report: {selected_name}")

report = load_report_json(selected["source_filename"])

if report:
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Rows", report["row_count"])
    kpi_cols[1].metric("Columns", report["column_count"])
    kpi_cols[2].metric("Machines", report["machine_count"])
    kpi_cols[3].metric("Duplicate Rows", report["duplicate_row_count"])
    kpi_cols[4].metric(
        "Records / Machine", report["records_per_machine_mean"], help="Mean rows per machine"
    )

    badge_cols = st.columns(2)
    with badge_cols[0]:
        st.markdown("**Supported modelling tasks**")
        if report["supported_tasks"]:
            for task in report["supported_tasks"]:
                st.success(task.replace("_", " ").title(), icon="✅")
        else:
            st.warning("None identified from this dataset alone.")
    with badge_cols[1]:
        st.markdown("**Limitations**")
        if report["limitations"]:
            for limitation in report["limitations"]:
                st.warning(limitation, icon="⚠️")
        else:
            st.success("None identified.", icon="✅")

    missingness = report.get("missingness_pct", {})
    if missingness:
        st.markdown("**Missingness by column**")
        items = sorted(missingness.items(), key=lambda kv: kv[1])
        cols = [k for k, _ in items]
        values = [v for _, v in items]
        fig = go.Figure(
            go.Bar(
                x=values,
                y=cols,
                orientation="h",
                marker=dict(
                    color=values,
                    colorscale=sequential_blue_scale(),
                    cmin=0,
                    cmax=100,
                ),
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            )
        )
        fig.update_layout(
            xaxis_title="% missing",
            yaxis_title=None,
            height=max(320, 24 * len(cols)),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    if report.get("date_range"):
        st.markdown("**Date fields**")
        for field_name, (start, end) in report["date_range"].items():
            st.write(f"- `{field_name}`: {start} to {end}")
else:
    st.caption(
        "No saved feasibility JSON found for this dataset (it may have "
        "been prepared before charts were added) -- showing the Markdown "
        "report only."
    )

safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in selected["source_filename"])
report_path = PATHS["reports_feasibility"] / f"{safe_name}_feasibility.md"

with st.expander("Full report (Markdown)", expanded=report is None):
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.warning(
            f"No saved feasibility report found at {report_path}. "
            "Re-run Import Dataset for this file to regenerate it."
        )

st.divider()
with st.expander(f"🗑️ Danger Zone: Delete '{selected_name}'"):
    st.warning(
        f"This permanently deletes dataset **{selected_name}** and everything "
        "derived from it: its processed data file, registered machines, "
        "predictions, SHAP explanations, alerts, trained model versions, "
        "and experiment reports. **This cannot be undone.**",
        icon="⚠️",
    )

    # Keyed to the selected dataset so switching datasets always resets
    # these confirmation widgets -- a checked box or typed name must never
    # carry over and accidentally arm deletion for a different dataset.
    confirm_checkbox = st.checkbox(
        f"I understand this will permanently delete '{selected_name}' and all related data.",
        key=f"confirm_delete_checkbox_{selected_name}",
    )
    confirm_text = st.text_input(
        f"Type the dataset name ({selected_name}) to confirm:",
        key=f"confirm_delete_text_{selected_name}",
    )
    can_delete = confirm_checkbox and confirm_text == selected_name

    if st.button("Permanently Delete Dataset", disabled=not can_delete, type="primary"):
        with st.spinner(f"Deleting '{selected_name}'..."):
            summary = delete_dataset(selected_name)
        clear_all_caches()
        st.success(
            f"Deleted '{summary.dataset_name}': {summary.machines_deleted} machines, "
            f"{summary.predictions_deleted} predictions, {summary.alerts_deleted} alerts, "
            f"{summary.model_versions_deleted} model versions, and "
            f"{len(summary.files_deleted)} file(s)/director(ies) removed."
        )
        if summary.files_failed:
            st.warning(
                f"Could not remove {len(summary.files_failed)} file(s) (they may be "
                f"open elsewhere): {summary.files_failed}"
            )
        st.rerun()
