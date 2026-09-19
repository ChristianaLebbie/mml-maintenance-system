"""Streamlit-facing login gate. Call `require_login()` at the top of every
page (after `st.set_page_config()` and `init_db()`) -- it renders the
sign-in / first-run setup screen and stops the rest of the page from
running until someone is signed in, and does nothing (just returns) once
they are.

Session state key `auth_user` (a small dict: id/username/display_name/
role) is Streamlit's own per-browser-session state, so it naturally
persists as the person moves between pages without any extra work here.
"""

from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME
from database.database import get_session
from src.services.auth_service import any_users_exist, authenticate, create_user

_CARD_CSS = """
<style>
.eips-shell {
    max-width: 460px;
    margin: 3rem auto 1rem auto;
    padding: 2.25rem 2.25rem 1.75rem 2.25rem;
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(42,120,214,0.06) 0%, rgba(42,120,214,0.00) 60%);
    border: 1px solid rgba(42,120,214,0.18);
}
.eips-icon {
    font-size: 2.6rem;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.eips-title {
    font-size: 1.55rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
    color: #184f95;
}
.eips-tagline {
    font-size: 0.95rem;
    color: #4a5568;
    margin-bottom: 1.25rem;
    line-height: 1.4;
}
.eips-welcome-back {
    font-size: 1.1rem;
    font-weight: 600;
    color: #184f95;
}
</style>
"""


def _render_header(subtitle: str) -> None:
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="eips-shell">
            <div class="eips-icon">⛏️</div>
            <div class="eips-title">{APP_NAME}</div>
            <div class="eips-tagline">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(
        "Explainable, peer-adjusted maintenance intervention prioritization for "
        "mining equipment — a research decision-support tool, not a substitute "
        "for engineering judgment."
    )


def _render_first_run_setup() -> None:
    _render_header(
        "Nobody has an account yet. Set up the first account below — "
        "this becomes the first signed-in user, who can add teammates later "
        "from System Information."
    )
    with st.form("eips_first_run_setup", clear_on_submit=False):
        display_name = st.text_input("Your name", placeholder="e.g. Christiana Lebbie")
        username = st.text_input("Choose a username", placeholder="e.g. ama")
        password = st.text_input("Choose a password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account & sign in", type="primary")

    if submitted:
        if password != confirm:
            st.error("Passwords don't match.")
        else:
            with get_session() as session:
                result = create_user(session, username, display_name, password, role="Admin")
                if not result.ok:
                    st.error(result.error)
                else:
                    st.session_state["auth_user"] = {
                        "id": result.user.id,
                        "username": result.user.username,
                        "display_name": result.user.display_name,
                        "role": result.user.role,
                    }
                    st.session_state["_eips_just_signed_in"] = True
                    st.rerun()
    _render_footer()


def _render_login_form() -> None:
    _render_header("Sign in to view maintenance intervention priorities and analysis.")
    with st.form("eips_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        with get_session() as session:
            result = authenticate(session, username, password)
        if not result.ok:
            st.error(result.error)
        else:
            st.session_state["auth_user"] = {
                "id": result.user.id,
                "username": result.user.username,
                "display_name": result.user.display_name,
                "role": result.user.role,
            }
            st.session_state["_eips_just_signed_in"] = True
            st.rerun()
    _render_footer()


def _render_signed_in_sidebar() -> None:
    user = st.session_state["auth_user"]
    with st.sidebar:
        st.markdown(f"**{user['display_name']}**")
        st.caption(f"@{user['username']} · {user['role']}")
        if st.button("Sign out", use_container_width=True):
            del st.session_state["auth_user"]
            st.rerun()
        st.divider()

    if st.session_state.pop("_eips_just_signed_in", False):
        st.toast(f"Welcome back, {user['display_name']}.", icon="👋")


def require_login() -> None:
    if st.session_state.get("auth_user") is not None:
        _render_signed_in_sidebar()
        return

    with get_session() as session:
        setup_needed = not any_users_exist(session)

    if setup_needed:
        _render_first_run_setup()
    else:
        _render_login_form()
    st.stop()
