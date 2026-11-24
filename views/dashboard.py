import streamlit as st

from components import charts
from components.cards import entry_card, hero, metric
from modules.database import db

hero("Welcome back 👋", "Your Voice. Your Emotions. Your Story. — a multimodal diary built on DA321 concepts.")

entries = db.list_entries()

top = st.columns([1, 1, 1, 1.3])
with top[0]:
    metric("Total memories", len(entries))
with top[1]:
    metric("With audio", sum(1 for e in entries if e["audio_path"]))
with top[2]:
    metric("With photo", sum(1 for e in entries if e["image_path"]))
with top[3]:
    st.write("")
    st.page_link("views/create_memory.py", label="➕ Quick create a memory", width="stretch")

if not entries:
    st.markdown("### Your diary is empty")
    st.write("Record or upload a voice note, paste its transcript, attach a photo and choose a mood.")
    c1, c2 = st.columns(2)
    with c1:
        st.page_link("views/create_memory.py", label="Create my first memory", icon="🎙️")
    with c2:
        if st.button("Load labelled demo data (synthetic)", help="5 entries made of generated tones/images, "
                     "marked [DEMO]. Delete them anytime from My Memories."):
            from scripts.seed_demo import seed
            with st.spinner("Generating demo memories..."):
                seed()
            st.rerun()
    st.stop()

left, right = st.columns([1.15, 1])
with left:
    st.plotly_chart(charts.timeline(entries), width="stretch", theme=None)
with right:
    st.plotly_chart(charts.mood_distribution(entries), width="stretch", theme=None)

st.markdown("### Recent memories")
recent = entries[:6]
cols = st.columns(3)
for i, e in enumerate(recent):
    with cols[i % 3]:
        entry_card(e)
        if st.button("Open analysis", key=f"dash_open_{e['id']}", width="stretch"):
            st.session_state["selected_id"] = e["id"]
            st.switch_page("views/analysis.py")

if any(e["is_demo"] for e in entries):
    st.caption("Entries tagged [DEMO] are synthetic demonstration data, not real recordings.")


with st.expander("🧹 Library check (files vs database)"):
    h = db.find_orphans()
    if not h["orphan_files"] and not h["missing_files"]:
        st.success("Every saved memory has its files, and there are no unreferenced files.")
    if h["missing_files"]:
        st.warning("Memories whose media file is missing from disk (open them and use Edit to attach a new file):")
        for m in h["missing_files"]:
            st.write(f"• {m['title']} — `{m['path']}`")
    if h["orphan_files"]:
        st.warning(f"{len(h['orphan_files'])} file(s) in the data folder belong to no memory (left over from an "
                   "interrupted action).")
        ok = st.checkbox("I understand these files will be permanently deleted", key="orph_ok")
        if st.button("Delete unreferenced files", disabled=not ok):
            n = sum(db.delete_media(p) for p in h["orphan_files"])
            st.success(f"Deleted {n} file(s)."); st.rerun()
