"""QuantProof -- can this backtest be trusted?

Entry point. Page bodies live in ``app_pages/``; the analytics live in ``qp/``.
"""

import streamlit as st

st.set_page_config(
    page_title="QuantProof",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One place for every piece of cross-rerun state the app keeps.
st.session_state.setdefault("frame", None)          # raw uploaded table
st.session_state.setdefault("source_label", None)   # what to call it on screen
st.session_state.setdefault("source_note", None)    # sample commentary, if any
st.session_state.setdefault("last_upload", None)    # de-dupes the uploader
st.session_state.setdefault("overrides", {})        # manual column choices

pages = [
    st.Page(
        "app_pages/analyze.py",
        title="Analyse a backtest",
        icon=":material/query_stats:",
        default=True,
    ),
    st.Page(
        "app_pages/methodology.py",
        title="Methodology",
        icon=":material/functions:",
    ),
]

st.navigation(pages).run()
