"""Shared visual polish for every page: a global CSS pass plus a
consistent, icon-led page-header banner.

Call `inject_global_css()` once near the top of every page (same place
`require_login()` is called) and use `render_page_header(...)` in place of
a bare `st.title(...)`. Nothing here changes any data or logic -- it is
presentation only, and if a future Streamlit version stops honoring one of
these selectors the page still renders correctly, just without that one
polish detail.
"""

from __future__ import annotations

import streamlit as st

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* -- Metric tiles: rounded cards with a soft shadow and a blue accent
   stripe, instead of plain unboxed numbers. -- */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid rgba(24,79,149,0.14);
    border-top: 3px solid #184f95;
    border-radius: 12px;
    padding: 1rem 1.1rem 0.85rem 1.1rem;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
}
div[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #4a5568;
}

/* -- Buttons: a touch more rounded, with a gentle lift on hover. -- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.06s ease-in-out, box-shadow 0.06s ease-in-out;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    box-shadow: 0 2px 8px rgba(24,79,149,0.20);
}

/* -- Tabs: a clearer active-tab underline in the brand blue. -- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [aria-selected="true"] {
    color: #184f95 !important;
    font-weight: 700;
}

/* -- Tables / dataframes: rounded corners instead of hard-edged grids. -- */
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(24,79,149,0.10);
}

/* -- Alert / info / warning boxes: consistent rounding. -- */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* -- Sidebar: a subtle tint so it reads as its own panel. -- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EAF2FB 0%, #F7FAFD 100%);
    border-right: 1px solid rgba(24,79,149,0.10);
}

/* -- Expanders: rounded, with a light border. -- */
div[data-testid="stExpander"] {
    border-radius: 10px;
    border: 1px solid rgba(24,79,149,0.12);
}
</style>
"""

_HEADER_CSS = """
<style>
.eips-page-banner {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.25rem;
    border-radius: 14px;
    background: linear-gradient(120deg, rgba(24,79,149,0.08) 0%, rgba(24,79,149,0.00) 75%);
    border: 1px solid rgba(24,79,149,0.16);
}
.eips-page-banner .eips-banner-icon {
    font-size: 2.1rem;
    line-height: 1;
}
.eips-page-banner .eips-banner-text h1 {
    font-size: 1.65rem;
    font-weight: 700;
    color: #184f95;
    margin: 0;
    padding: 0;
    line-height: 1.25;
}
.eips-page-banner .eips-banner-text p {
    font-size: 0.95rem;
    color: #4a5568;
    margin: 0.15rem 0 0 0;
    line-height: 1.4;
}
</style>
"""


def inject_global_css() -> None:
    """Apply the site-wide visual polish. Safe to call on every page --
    Streamlit de-duplicates identical injected `<style>` blocks."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def render_page_header(title: str, icon: str, subtitle: str | None = None) -> None:
    """Replacement for a bare `st.title(...)`: an icon-led banner that
    matches the sign-in screen's styling, so every page in the app reads
    as one consistent, designed system rather than a stack of default
    Streamlit pages."""
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="eips-page-banner">
            <div class="eips-banner-icon">{icon}</div>
            <div class="eips-banner-text">
                <h1>{title}</h1>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
