import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts
from components.cards import entry_card, header_line, hero, note, pill
from components.common import empty_state, load_entry_audio, load_entry_image, pick_entry
from modules.audio.visualisation import waveform_fig
from modules.text.corpus import build_corpus
from modules import config
from modules.database import db
from modules.multimodal.integration import AUDIO_KEYS, IMAGE_KEYS, mood_summary, similar_entries
from modules.text.tfidf import top_keywords
from modules.viz_style import style

hero("Multimodal insights", "One memory, three views: how voice, words and photo each describe it.")
entries = db.list_entries()
if not entries:
    empty_state(); st.stop()
e = pick_entry(entries, "insight_pick")
idx = next(i for i, x in enumerate(entries) if x["id"] == e["id"])
st.markdown(header_line(e), unsafe_allow_html=True)
note("The mood above is <b>your own selection</b>. EchoDiary does not detect emotion. The numbers below are "
     "descriptive measurements; they are not validated measures of emotion or memory quality. "
     "Fusion is an <span class='eng'>engineering design decision</span>, not a method taught in DA321.")


def percentile(key, section):
    """Where this entry sits among the OTHER entries (mid-rank percentile, 0-100); None if < 3 others."""
    others = [x[section][key] for x in entries if x["id"] != e["id"] and x.get(section) and key in x[section]]
    if len(others) < 3 or not e.get(section) or key not in e[section]:
        return None
    v = e[section][key]
    return 100.0 * (sum(1 for t in others if t < v) + 0.5 * sum(1 for t in others if t == v)) / len(others)


def rel(p):
    return "" if p is None else f" (above {p:.0f}% of your other entries)"


af, imf = e["audio_features"], e["image_features"]
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 🎙️ Audio contributes *how it was said*")
    if af:
        ad = load_entry_audio(e, warn=False)
        if ad is not None:
            st.plotly_chart(waveform_fig(ad.x, ad.fs).update_layout(height=140, title=None,
                            margin=dict(l=40, r=10, t=5, b=25)), width="stretch", theme=None)
        st.metric("Duration", f"{af['duration_s']:.1f} s")
        st.write(f"- Mean short-time energy {af['energy_mean']:.4f}{rel(percentile('energy_mean', 'audio_features'))}")
        st.write(f"- Mean ZCR {af['zcr_mean']:.3f}{rel(percentile('zcr_mean', 'audio_features'))}")
        st.write(f"- Spectral centroid ≈ {af['centroid_hz']:.0f} Hz")
        st.write(f"- Speech/sound active in {af['active_fraction'] * 100:.0f}% of frames")
        if af["f0_hz"] and af["periodicity"] > 0.3:
            st.write(f"- Autocorrelation peak ≈ {af['f0_hz']:.0f} Hz (r={af['periodicity']:.2f})")
        st.caption("Energy, ZCR: syllabus-listed; centroid, active share, peak pitch: engineering descriptors.")
    else:
        st.caption("No audio in this memory.")
with c2:
    st.markdown("### 📝 Text contributes *what was said*")
    if e["transcript"].strip():
        corpus = build_corpus(entries)            # same corpus as Search and Analysis
        kws = top_keywords(corpus.vecs[idx], 6)
        tf = e["text_features"] or {}
        st.metric("Words", tf.get("n_tokens", len(e["transcript"].split())))
        if kws:
            st.write("Distinctive terms (TF-IDF): " + ", ".join(f"**{k}**" for k, _ in kws))
        else:
            st.write("Most frequent terms: " + ", ".join(f"**{k}**" for k in tf.get("top_terms", [])[:6]))
        st.write(f"- {tf.get('n_unique', '–')} unique words, type/token ratio {tf.get('type_token_ratio', 0):.2f}")
    else:
        st.caption("No transcript in this memory.")
with c3:
    st.markdown("### 🖼️ Image contributes *the scene*")
    if imf:
        thumb = load_entry_image(e, warn=False)
        if thumb is not None:
            st.image(thumb, width=220)
        sw = "#%02x%02x%02x" % (int(imf["mean_r"]), int(imf["mean_g"]), int(imf["mean_b"]))
        st.markdown(f'<div style="height:34px;border-radius:10px;background:{sw};border:1px solid #475569"></div>',
                    unsafe_allow_html=True)
        st.caption(f"Mean colour RGB ({imf['mean_r']:.0f}, {imf['mean_g']:.0f}, {imf['mean_b']:.0f})")
        st.write(f"- Brightness {imf['brightness']:.0f}/255{rel(percentile('brightness', 'image_features'))}")
        st.write(f"- Contrast (std) {imf['contrast']:.0f}")
        st.write(f"- Fourier: {imf['low_freq_ratio'] * 100:.0f}% of spectral magnitude at low frequencies, "
                 f"{imf['high_freq_ratio'] * 100:.0f}% at high frequencies (fine detail)")
        if "edge_density" in imf:
            st.write(f"- Sobel edges: {imf['edge_density'] * 100:.0f}% of pixels are edge pixels")
        st.caption("Brightness, contrast, Fourier shares and edge density are descriptive measurements; the "
                   "summary statistics themselves are engineering choices.")
    else:
        st.caption("No photo in this memory.")

have = [n for n, ok in (("audio", af), ("text", e["transcript"].strip()), ("image", imf)) if ok]
st.markdown("#### How the modalities complement each other")
st.write(
    f"This memory has **{' + '.join(have)}**. The transcript carries the *content* (topics and keywords) "
    "which the audio signal cannot give without speech recognition; the audio adds *delivery* — length, "
    "loudness pattern, noisiness (ZCR) and spectral shape — that text alone loses; the photo adds *visual "
    "context* — colour, brightness and amount of detail — that neither of the others contain. "
    "Together they make a profile that can be searched and compared on any of the three axes.")

# ---- combined profile (percentile radar, relative to this diary)
if len(entries) >= 3 and have:
    st.markdown("#### Combined memory profile — relative to your other entries")
    axes = []
    for k, label in (("energy_mean", "Audio energy"), ("zcr_mean", "Audio ZCR"),
                     ("centroid_hz", "Spectral centroid")):
        axes.append((label, percentile(k, "audio_features")))
    others_len = [len(x["transcript"].split()) for x in entries if x["id"] != e["id"] and x["transcript"].strip()]
    my_len = len(e["transcript"].split())
    axes.append(("Text length", 100.0 * (sum(1 for t in others_len if t < my_len) +
                                         0.5 * sum(1 for t in others_len if t == my_len)) / len(others_len)
                 if e["transcript"].strip() and len(others_len) >= 3 else None))
    for k, label in (("brightness", "Image brightness"), ("contrast", "Image contrast"),
                     ("high_freq_ratio", "Image detail (HF)")):
        axes.append((label, percentile(k, "image_features")))
    axes = [(a, p) for a, p in axes if p is not None]
    if len(axes) >= 3:
        fig = go.Figure(go.Scatterpolar(r=[p for _, p in axes] + [axes[0][1]],
                                        theta=[a for a, _ in axes] + [axes[0][0]], fill="toself",
                                        line=dict(color="#818cf8"), fillcolor="rgba(129,140,248,.3)"))
        fig.update_layout(polar=dict(bgcolor="rgba(15,23,42,.35)", radialaxis=dict(range=[0, 100],
                          gridcolor="rgba(148,163,184,.2)"), angularaxis=dict(gridcolor="rgba(148,163,184,.2)")))
        st.plotly_chart(style(fig, "Rank among your other entries (descriptive percentile, 0–100)", 380),
                        width="stretch", theme=None)
else:
    st.caption("Add at least 4 memories to see how this one compares with the others in your diary.")

# ---- similar memories
st.markdown("### Similar memories")
w = st.columns(3)
wa = w[0].slider("Audio weight", 0.0, 1.0, 1.0, 0.1)
wt = w[1].slider("Text weight", 0.0, 1.0, 1.0, 0.1)
wi = w[2].slider("Image weight", 0.0, 1.0, 1.0, 0.1)
if len(entries) < 2:
    st.info("Save at least two memories to compare them.")
elif wa + wt + wi == 0:
    st.warning("Give at least one modality a non-zero weight.")
else:
    sims = similar_entries(idx, entries, (wa, wt, wi), top_k=3)
    if not sims:
        st.info("No comparable memories (they need at least one modality in common).")
    cols = st.columns(3)
    for col, s in zip(cols, sims):
        with col:
            entry_card(entries[s["index"]], s["score"])
            fig = go.Figure(go.Bar(x=[v for v in s["parts"].values()], y=list(s["parts"].keys()), orientation="h",
                                   marker_color=["#818cf8", "#38bdf8", "#f472b6"][:len(s["parts"])]))
            fig.update_xaxes(range=[-1, 1], title="per-modality cosine")
            st.plotly_chart(style(fig, None, 170), width="stretch", theme=None)
    st.caption("Late fusion: weighted average of per-modality scores — audio: cosine of z-scored descriptors; text: cosine "
               "of TF-IDF vectors; image: visual similarity (histogram + Sobel edges, 0–1; see Visual Search). "
               "TF-IDF, histograms, Sobel and the inner product/norm behind cosine are course concepts; the "
               "combination and weights are engineering design decisions.")

# ---- mood summary
st.markdown("### Measured characteristics by the moods you selected")
rows = mood_summary(entries)
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df.round(4), width="stretch", hide_index=True)
    if len(rows) >= 2 and df["Mean energy"].notna().any():
        d = df.dropna(subset=["Mean energy"])
        fig = go.Figure(go.Bar(x=d["Mood (user-selected)"], y=d["Mean energy"],
                               marker_color=[config.MOOD_COLORS[m] for m in d["Mood (user-selected)"]]))
        st.plotly_chart(style(fig, "Mean short-time energy per selected mood", 280), width="stretch", theme=None)
    st.caption("Descriptive averages over a handful of personal entries. They show how your own recordings "
               "differ, not that any mood can be detected from audio or text.")
