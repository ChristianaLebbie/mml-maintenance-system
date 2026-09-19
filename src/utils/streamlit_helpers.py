"""Streamlit-specific glue: caching wrappers around slow, frequently-
repeated service/DB calls, and a small progress-callback adapter for
long-running pipeline steps.

Kept separate from src/services/* and src/inference/* so those stay
framework-agnostic and usable from the CLI scripts without a Streamlit
dependency -- only this module (and the pages that import it) know about
Streamlit.

Streamlit reruns the WHOLE page script on every widget interaction, so
without caching, each button click / page navigation re-reads the
database, re-loads the trained model from disk, and re-reads the
processed dataset from scratch. The wrappers below cache those results
for the life of the process (or until explicitly cleared -- see
`clear_all_caches`) so navigating the app feels fast.

Cache invalidation policy: no TTL. Every mutation that can make a cached
result stale (import, delete, train) is expected to call
`clear_all_caches()` itself right after succeeding, and a manual "Clear
Cache" control (System Information page) is available for the user to
force a refresh at any time.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from database.database import get_session
from database.models import Dataset, Machine
from src.inference.model_loader import LoadedModel, load_active_model
from src.services.dataset_service import load_processed_dataset

ProgressCallback = Callable[[str, float], None]


@st.cache_data(show_spinner=False)
def load_processed_dataset_cached(dataset_name: str) -> pd.DataFrame:
    return load_processed_dataset(dataset_name)


@st.cache_resource(show_spinner=False)
def load_active_model_cached(dataset_id: int | None = None) -> LoadedModel:
    return load_active_model(dataset_id)


@st.cache_data(show_spinner=False)
def list_dataset_names_cached() -> list[str]:
    with get_session() as session:
        return [d.name for d in session.query(Dataset).order_by(Dataset.name).all()]


@st.cache_data(show_spinner=False)
def list_machine_identifiers_cached() -> list[str]:
    with get_session() as session:
        return [
            m.machine_identifier
            for m in session.query(Machine).order_by(Machine.machine_identifier).all()
        ]


def clear_all_caches() -> None:
    """Clear every Streamlit cache (data + resource). Called automatically
    after import/delete/train actions; also exposed as a manual "Clear
    Cache" button (System Information page) so the user can force a
    refresh or free memory at any time."""
    st.cache_data.clear()
    st.cache_resource.clear()


def make_progress_callback(
    progress_bar, status_text
) -> ProgressCallback:
    """Adapt a (message, fraction) progress callback -- the convention
    used by the long-running service functions (dataset import, model
    training, batch prediction) -- to update a Streamlit progress bar and
    status caption in place."""

    def callback(message: str, fraction: float) -> None:
        progress_bar.progress(min(max(fraction, 0.0), 1.0))
        status_text.caption(message)

    return callback
