"""Import Dataset page: upload Excel/CSV, preview, map columns, validate,
and prepare the dataset (preprocessing + feasibility report + DB
persistence) via src.services.dataset_service."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

import datetime

from config.settings import APP_NAME, PATHS
from database.database import init_db
from src.utils.auth_ui import require_login
from src.utils.theme import inject_global_css, render_page_header
from src.ingestion.csv_loader import CSVDataSource
from src.ingestion.excel_loader import ExcelDataSource
from src.ingestion.mapper import INTERNAL_FIELDS, suggest_mapping, validate_mapping
from src.ingestion.template import build_template_workbook
from src.services.dataset_service import import_and_prepare
from src.services.model_service import train_and_activate
from src.utils.streamlit_helpers import clear_all_caches, make_progress_callback

st.set_page_config(page_title=f"Import Dataset - {APP_NAME}", page_icon="⛏️", layout="wide")
init_db()
require_login()
inject_global_css()

render_page_header("Import Dataset", icon="📂")
st.caption(
    "Upload any mining-site Excel/CSV workbook -- an asset register, a CMMS "
    "export, or sensor telemetry. The original file is never modified -- a "
    "copy is saved under data/uploads/ and the standardized, preprocessed "
    "result is saved separately under data/processed/. Columns don't need "
    "to match any fixed layout: map what applies below, and anything else "
    "is still carried through automatically."
)

with st.expander("Optional: download a blank import template"):
    st.write(
        "Not required -- any column layout can be uploaded and mapped "
        "manually below. This template just shows the recommended field "
        "names and an example row, with a reference sheet explaining each "
        "one."
    )
    st.download_button(
        "Download template (.xlsx)",
        data=build_template_workbook(),
        file_name="predictive_maintenance_import_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

uploaded_file = st.file_uploader("Upload workbook", type=["xlsx", "xls", "csv"])

if uploaded_file is not None and st.session_state.get("import_source_name") != uploaded_file.name:
    # A different file was uploaded -- drop any previous prepare/train
    # results so stale output from the last dataset isn't shown alongside
    # a new one.
    st.session_state.pop("prepared_dataset", None)
    st.session_state.pop("train_result", None)
    st.session_state["import_source_name"] = uploaded_file.name

if uploaded_file is not None:
    uploads_dir = PATHS["uploads"]
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_path = uploads_dir / uploaded_file.name
    saved_path.write_bytes(uploaded_file.getvalue())
    st.success(f"Saved upload to {saved_path}")

    is_excel = saved_path.suffix.lower() in (".xlsx", ".xls")
    sheet_name = None

    if is_excel:
        temp_source = ExcelDataSource(saved_path)
        problems = temp_source.validate()
        if problems:
            st.error("Validation problems: " + "; ".join(problems))
            st.stop()
        sheet_names = temp_source.inspect().sheet_names
        sheet_name = st.selectbox("Sheet", sheet_names)
        source = ExcelDataSource(saved_path, sheet_name=sheet_name)
    else:
        source = CSVDataSource(saved_path)
        problems = source.validate()
        if problems:
            st.error("Validation problems: " + "; ".join(problems))
            st.stop()

    inspection = source.inspect()
    st.subheader("Preview")
    st.write(
        f"{inspection.row_count} rows"
        f"{' (partial scan of a very large file)' if inspection.is_partial_scan else ''}, "
        f"{inspection.column_count} columns, {inspection.duplicate_row_count} duplicate rows detected."
    )
    st.dataframe(pd.DataFrame(inspection.preview), use_container_width=True)

    with st.expander("Column profile"):
        profile_df = pd.DataFrame(
            [
                {
                    "column": c.name,
                    "dtype": c.dtype,
                    "missing_pct": c.missing_pct,
                    "sample_values": ", ".join(str(v) for v in c.sample_values),
                }
                for c in inspection.columns
            ]
        )
        st.dataframe(profile_df, use_container_width=True)

    st.subheader("Column Mapping")
    st.caption(
        "Automatically suggested from the columns above -- correct any "
        "field before preparing the dataset. Fields left as 'None' will "
        "be stored as missing."
    )
    columns = [c.name for c in inspection.columns]
    suggested = suggest_mapping(columns)

    mapping: dict[str, str | None] = {}
    col1, col2 = st.columns(2)
    field_items = list(INTERNAL_FIELDS.items())
    for i, (field_name, spec) in enumerate(field_items):
        target_col = col1 if i % 2 == 0 else col2
        options = ["None"] + columns
        default = suggested.get(field_name) or "None"
        label = f"{field_name}{' *' if spec['required'] else ''}"
        choice = target_col.selectbox(label, options, index=options.index(default), key=f"map_{field_name}")
        mapping[field_name] = None if choice == "None" else choice

    mapping_problems = validate_mapping(mapping, columns)
    if mapping_problems:
        st.warning("\n".join(mapping_problems))

    dataset_name = st.text_input("Dataset name (used for storage and future reuse)", value=saved_path.stem)

    if st.button("Prepare Dataset", disabled=bool(mapping_problems) or not dataset_name):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        callback = make_progress_callback(progress_bar, status_text)
        try:
            result = import_and_prepare(
                saved_path,
                dataset_name,
                sheet_name=sheet_name,
                mapping_override=mapping,
                progress_callback=callback,
            )
        finally:
            progress_bar.empty()
            status_text.empty()
        st.session_state["prepared_dataset"] = result
        st.session_state.pop("train_result", None)
        clear_all_caches()

    if "prepared_dataset" in st.session_state:
        result = st.session_state["prepared_dataset"]
        st.success(f"Dataset '{result.dataset_name}' prepared: {result.machine_count} machines.")
        st.write("**Supported modelling tasks:**", result.feasibility_report.supported_tasks)
        if result.feasibility_report.limitations:
            st.info("**Limitations:**\n" + "\n".join(f"- {l}" for l in result.feasibility_report.limitations))
        st.write(f"Full feasibility report: `{result.feasibility_report_path}`")

        st.divider()
        st.subheader("Train Models")
        can_train = "intervention_priority_classification" in result.feasibility_report.supported_tasks
        if not can_train:
            st.warning(
                "This dataset doesn't have enough populated operational "
                "metrics (PM/work-order compliance counts, or other numeric "
                "columns) to support intervention-priority classification -- "
                "see Limitations above. Training is disabled for this dataset."
            )
        else:
            st.caption(
                "Trains logistic regression, decision tree, random forest, and "
                "XGBoost, then activates whichever performs best (ranked by "
                "PR-AUC, tie-broken by recall -- never plain accuracy). This "
                "can take a while for a large dataset."
            )
            default_version = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            version = st.text_input(
                "Model version (must be unique for this dataset)",
                value=default_version,
                key="train_version",
            )
            if st.button("Train Models on This Dataset", disabled=not version):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                callback = make_progress_callback(progress_bar, status_text)
                try:
                    train_result = train_and_activate(
                        result.dataset_name, version, progress_callback=callback
                    )
                finally:
                    progress_bar.empty()
                    status_text.empty()
                st.session_state["train_result"] = train_result
                clear_all_caches()

    if "train_result" in st.session_state:
        train_result = st.session_state["train_result"]
        st.success(
            f"Training complete for version {train_result.version}. "
            f"Selected & activated model: **{train_result.best_model_name}**"
        )
        comparison = pd.DataFrame(
            [{"model": name, **ev.__dict__} for name, ev in train_result.evaluations.items()]
        )
        st.dataframe(
            comparison[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]],
            use_container_width=True,
        )
        st.write(f"Full selection report: `{train_result.selection_report_path}`")
        st.info("See the **Model Performance** page for confusion matrices and ROC/PR curves.")
