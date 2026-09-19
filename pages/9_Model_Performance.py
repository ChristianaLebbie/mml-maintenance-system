"""Model Performance page: comparison chart across trained candidates,
confusion-matrix heatmaps, ROC/PR curves, and the selection rationale for
the active model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import APP_NAME, PATHS
from database.database import get_session, init_db
from src.utils.auth_ui import require_login
from src.utils.theme import inject_global_css, render_page_header
from database.models import ModelVersion
from src.utils.viz_theme import MODEL_COLORS, MODEL_ORDER, sequential_blue_scale

st.set_page_config(page_title=f"Model Performance - {APP_NAME}", page_icon="⛏️", layout="wide")
init_db()
require_login()
inject_global_css()

render_page_header("Model Performance", icon="📈")
st.caption(
    "Component III (pipeline demonstration): these metrics show the classifier "
    "learned its compliance-derived proxy label well -- they are not evidence "
    "about real intervention outcomes, which the planned engineering-validated "
    "labeling pilot will establish (see System Information)."
)

experiments_root = PATHS["reports_experiments"]
experiment_dirs = sorted([d for d in experiments_root.iterdir() if d.is_dir()]) if experiments_root.exists() else []

if not experiment_dirs:
    st.info("No training experiments found yet. Run scripts/train_models.py first.")
    st.stop()

experiment_name = st.selectbox("Experiment", [d.name for d in experiment_dirs], index=len(experiment_dirs) - 1)
experiment_dir = experiments_root / experiment_name

metrics_files = list(experiment_dir.glob("*_metrics.json"))
rows = []
for f in metrics_files:
    metrics = json.loads(f.read_text(encoding="utf-8"))
    rows.append({"model": f.stem.replace("_metrics", ""), **metrics})
# Fixed, non-cycled model order -- a model's color/position never shifts
# based on which other models happen to be in this experiment.
rows.sort(key=lambda r: MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else len(MODEL_ORDER))

if rows:
    df = pd.DataFrame(rows)[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positives", "false_negatives"]
    ]
    st.subheader("Comparison Table")
    st.dataframe(df, use_container_width=True)

    metric_choice = st.selectbox("Metric to chart", ["f1", "recall", "precision", "roc_auc", "pr_auc", "accuracy"])
    fig = px.bar(
        df,
        x="model",
        y=metric_choice,
        color="model",
        color_discrete_map=MODEL_COLORS,
        category_orders={"model": [m for m in MODEL_ORDER if m in df["model"].values]},
        title=f"{metric_choice} by model",
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrices")
    cm_cols = st.columns(min(len(rows), 4)) or [st]
    for i, row in enumerate(rows):
        cm = row.get("confusion_matrix")
        if not cm:
            continue
        with cm_cols[i % len(cm_cols)]:
            st.caption(row["model"])
            heatmap = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=["Pred: Normal/Watch", "Pred: High Priority"],
                    y=["Actual: Normal/Watch", "Actual: High Priority"],
                    colorscale=sequential_blue_scale(),
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False,
                )
            )
            heatmap.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(heatmap, use_container_width=True)

    curve_rows = [r for r in rows if r.get("roc_curve") and r.get("pr_curve")]
    if curve_rows:
        st.subheader("ROC and Precision-Recall Curves")
        curve_cols = st.columns(2)
        with curve_cols[0]:
            roc_fig = go.Figure()
            roc_fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#c3c2b7"), name="Chance")
            )
            for row in curve_rows:
                roc_fig.add_trace(
                    go.Scatter(
                        x=row["roc_curve"]["fpr"],
                        y=row["roc_curve"]["tpr"],
                        mode="lines",
                        name=row["model"],
                        line=dict(color=MODEL_COLORS.get(row["model"])),
                    )
                )
            roc_fig.update_layout(
                title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
            )
            st.plotly_chart(roc_fig, use_container_width=True)
        with curve_cols[1]:
            pr_fig = go.Figure()
            for row in curve_rows:
                pr_fig.add_trace(
                    go.Scatter(
                        x=row["pr_curve"]["recall"],
                        y=row["pr_curve"]["precision"],
                        mode="lines",
                        name=row["model"],
                        line=dict(color=MODEL_COLORS.get(row["model"])),
                    )
                )
            pr_fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(pr_fig, use_container_width=True)
    else:
        st.caption(
            "ROC/PR curve data not available for this experiment (saved "
            "before curve data was added) -- re-run training to see them."
        )

selection_report = experiment_dir / "model_selection_report.md"
st.subheader("Model Selection Rationale")
if selection_report.exists():
    st.markdown(selection_report.read_text(encoding="utf-8"))
else:
    st.info("Run scripts/select_model.py for this experiment to generate the selection report.")

st.subheader("Currently Active Model")
with get_session() as session:
    active = session.query(ModelVersion).filter_by(active=True).order_by(ModelVersion.created_at.desc()).first()
    if active:
        st.json(
            {
                "model_type": active.model_type,
                "version": active.version,
                "threshold": active.threshold,
                "artifact_path": active.artifact_path,
                "created_at": str(active.created_at),
            }
        )
    else:
        st.info("No active model registered yet. Run scripts/select_model.py.")
