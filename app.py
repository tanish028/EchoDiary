"""EchoDiary entry point: `streamlit run app.py`."""
import streamlit as st

st.set_page_config(page_title="EchoDiary", page_icon="🎙️", layout="wide")

from components.sidebar import inject_css, render_sidebar  # noqa: E402
from modules import config  # noqa: E402
from modules.database import db  # noqa: E402
from modules.pipeline import backfill_visual_features  # noqa: E402

log = config.get_logger("app")


def _startup() -> None:
    """Once per browser session: create/upgrade the database and compute descriptors for older photos."""
    db.init_db()
    res = backfill_visual_features()
    if res["computed"] or res["skipped"]:
        log.info("Descriptor backfill: %s", res)
    st.session_state["_booted"] = True


if not st.session_state.get("_booted"):
    with st.spinner("Preparing your diary…"):
        _startup()
inject_css()
render_sidebar(db.count_entries())

nav = st.navigation([
    st.Page("views/dashboard.py", title="Home", icon="🏠", default=True),
    st.Page("views/create_memory.py", title="Create Memory", icon="➕"),
    st.Page("views/my_memories.py", title="My Memories", icon="📚"),
    st.Page("views/analysis.py", title="Memory Analysis", icon="📊"),
    st.Page("views/visual_search.py", title="Visual Search", icon="🔎"),
    st.Page("views/insights.py", title="Multimodal Insights", icon="🔗"),
])
nav.run()
