import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components import charts
from components.cards import entry_card, esc, hero, note
from components.common import empty_state, load_entry_image, open_entry
from modules.database import db
from modules.image import edges, visual
from modules.image.histogram import gray_histogram
from modules.image.preprocessing import ImageError, fit_gray, load_image
from modules.pipeline import backfill_visual_features
from modules.viz_style import style

hero("Visual memory search", "Upload a photo and find the saved memories whose photos look alike.")
note("<b>Low-level visual similarity — not recognition.</b> Photos are compared only by their grayscale intensity "
     "histogram and Sobel edge statistics (edge strength, direction and coarse layout). EchoDiary does not know what "
     "is <i>in</i> a photo: it cannot recognise objects, people, places or scenes, so two photos of different things "
     "can look “similar” and two photos of the same thing can look “different”. Histograms and Sobel edges are course "
     "concepts (PDF2 pp.115-128); the descriptors, distances and weighting are "
     "<span class='eng'>engineering contributions</span>.")

entries = db.list_entries()
with_img = [e for e in entries if e["image_path"]]
if not with_img:
    st.info("None of your memories has a photo yet, so there is nothing to search. Add a photo to a memory first.")
    st.page_link("views/create_memory.py", label="➕ Create a memory", icon="🎙️")
    st.stop()

# ---- make sure every stored photo has persisted descriptors (older entries are upgraded once)
bf = backfill_visual_features()
if bf["computed"]:
    st.caption(f"Computed and saved descriptors for {bf['computed']} older photo(s).")
if bf["skipped"]:
    names = ", ".join(f"“{e['title']}”" for e in with_img if e["id"] in bf["skipped"])
    st.warning(f"{len(bf['skipped'])} photo(s) could not be used (file missing or unreadable) and are skipped: {names}")
with_img = [e for e in db.list_entries() if e["image_path"] and e["visual_features"]]

# ---- reference photo
src_default = 1 if st.session_state.get("vs_source_id") else 0
mode = st.radio("Reference photo", ["Upload a photo", "Use the photo of a saved memory"], index=src_default,
                horizontal=True)
ref_desc, ref_rgb, ref_entry_id = None, None, None
if mode == "Upload a photo":
    up = st.file_uploader("Reference photograph", type=["png", "jpg", "jpeg", "webp", "bmp"], key="vs_up")
    if up is None:
        st.info("Upload a photograph to compare against your saved photos.")
    else:
        try:
            ref_rgb = load_image(up.getvalue())
            ref_desc = visual.visual_descriptors(ref_rgb)
        except ImageError as exc:
            st.error(f"{exc} The file may be corrupted.")
else:
    ids = [e["id"] for e in with_img]
    pre = st.session_state.get("vs_source_id")
    choice = st.selectbox("Saved memory", ids, index=ids.index(pre) if pre in ids else 0,
                          format_func=lambda i: next(f'{e["title"]} · {e["entry_date"]}' for e in with_img if e["id"] == i))
    ref_entry_id = choice
    ref_e = next(e for e in with_img if e["id"] == choice)
    ref_rgb = load_entry_image(ref_e)
    ref_desc = ref_e["visual_features"] if ref_rgb is not None else None

if ref_desc is None:
    st.stop()

c1, c2, c3 = st.columns([1, 1, 1])
wh = c1.slider("Histogram weight", 0.0, 1.0, 0.5, 0.05,
               help="Share of the score from the intensity histogram. The rest comes from Sobel edge statistics.")
topk = c2.slider("Results to show", 1, 12, 6)
c3.caption(f"Weights → histogram {wh:.0%}, edges {1 - wh:.0%}")

candidates = [(e["id"], e["visual_features"]) for e in with_img if e["id"] != ref_entry_id]
if not candidates:
    st.info("There are no other photos to compare with yet. Save more memories with photos.")
    st.stop()

ranked = visual.rank(ref_desc, candidates, {"histogram": wh, "edges": 1 - wh})[:topk]
by_id = {e["id"]: e for e in with_img}

st.markdown("### Reference")
r1, r2, r3 = st.columns(3)
r1.image(ref_rgb, caption="Reference photo", width="stretch")
g = fit_gray(ref_rgb, visual.VISUAL_SIDE)
h = gray_histogram(g, 32, True)
fig = go.Figure(go.Bar(x=np.arange(32) * 8, y=h, marker_color="#818cf8"))
r2.plotly_chart(style(fig, "Intensity histogram (32 bins)", 240), width="stretch", theme=None)
_, _, mag, _ = edges.sobel(g)
r3.plotly_chart(charts.image_fig(mag, "Sobel magnitude", "gray", 0, float(np.percentile(mag, 99.5) or 1)),
                width="stretch", theme=None)

st.markdown(f"### Most visually similar photos ({len(ranked)} of {len(candidates)})")
LABELS = {"histogram": "intensity histogram", "edge_strength": "edge strength", "edge_direction": "edge direction",
          "edge_layout": "edge layout"}
cols = st.columns(3)
for i, r in enumerate(ranked):
    e = by_id[r["id"]]
    with cols[i % 3]:
        st.markdown(f"**#{i + 1} · similarity {r['similarity'] * 100:.0f}%**")
        entry_card(e)
        fb = go.Figure(go.Bar(x=[1 - v for v in r["components"].values()],
                              y=[LABELS[k] for k in r["components"]], orientation="h",
                              marker_color=["#818cf8", "#38bdf8", "#34d399", "#f472b6"]))
        fb.update_xaxes(range=[0, 1], title="match (1 = identical)")
        st.plotly_chart(style(fb, None, 170), width="stretch", theme=None)
        if st.button("Open diary entry", key=f"vs_open_{e['id']}", width="stretch"):
            open_entry(e["id"])
st.caption("Similarity = 1 − (w_hist·d_hist + w_edge·d_edge); every distance is bounded in [0, 1] by construction "
           "(½·L1 between normalised histograms; mean absolute difference for the edge-layout grid), and d_edge = "
           "0.4·strength + 0.3·direction + 0.3·layout. These weights are engineering choices, not taught values, "
           "and the percentage is a ranking aid, not a measured probability.")
