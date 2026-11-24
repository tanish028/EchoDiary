import os
from datetime import date

import streamlit as st

from components.cards import hero, note, pill
from modules import config
from modules.audio.preprocessing import AudioError, load_wav_full, validate
from modules.image.preprocessing import ImageError
from modules.pipeline import analyse_image, analyse_text, create_memory

log = config.get_logger("ui.create")

hero("Create a memory", "Add a voice note, its transcript, a photo and your mood. Preview, then save.")

if "form_gen" not in st.session_state:
    st.session_state.form_gen = 0
g = st.session_state.form_gen

saved = st.session_state.pop("just_saved", None)
if saved:
    st.success("Memory saved ✔")
    c = st.columns(4)
    c[0].page_link("views/my_memories.py", label="Open My Memories", icon="📚")
    c[1].page_link("views/analysis.py", label="View its analysis", icon="📊")
    c[2].page_link("views/visual_search.py", label="Visual search", icon="🔎")
    c[3].page_link("views/insights.py", label="Insights", icon="🔗")

col1, col2 = st.columns([1.1, 1])
with col1:
    title = st.text_input("Title", placeholder="e.g. Sunday walk by the lake", key=f"t{g}", max_chars=80)
    entry_date = st.date_input("Date", value=date.today(), max_value=date.today(), key=f"d{g}")
    mood = st.selectbox("How did you feel? (your choice)", config.MOODS, index=None,
                        placeholder="Select a mood", key=f"m{g}")
    st.markdown("**🎙️ Audio** — record, or upload a .wav file")
    tab_rec, tab_up = st.tabs(["Record", "Upload WAV"])
    with tab_rec:
        rec = st.audio_input("Record a voice note", key=f"rec{g}")
    with tab_up:
        up = st.file_uploader("WAV file (PCM)", type=["wav"], key=f"up{g}")
    transcript = st.text_area("📝 Transcript (type or paste what you said)", height=160, key=f"tx{g}",
                              placeholder="EchoDiary has no speech recognition, so write the transcript yourself. "
                                          "English, Hindi (देवनागरी) and mixed text all work.")
with col2:
    st.markdown("**🖼️ Photo**")
    img = st.file_uploader("Photo", type=["png", "jpg", "jpeg", "webp", "bmp"], key=f"im{g}",
                           label_visibility="collapsed")
    if img:
        st.image(img, width="stretch")
    notes = st.text_input("Short description (optional)", key=f"n{g}", max_chars=200)

audio_bytes = rec.getvalue() if rec else (up.getvalue() if up else None)
image_bytes = img.getvalue() if img else None

st.markdown("### Preview")
problems = []
pc = st.columns(3)
with pc[0]:
    st.markdown("**Audio**")
    if audio_bytes:
        try:
            ad = load_wav_full(audio_bytes)
            v = validate(ad.x, ad.fs)
            st.audio(audio_bytes, format="audio/wav")
            st.caption(f"{ad.original_duration_s:.1f} s · {ad.fs} Hz · peak {v['peak']:.2f}")
            if ad.truncated:
                st.warning(f"The recording is {ad.original_duration_s:.0f} s long. The original is saved in full, "
                           f"but analysis covers only the first {ad.analysed_duration_s:.0f} s.")
            if v["is_silent"]:
                st.warning("This recording looks silent.")
            if v["is_clipped"]:
                st.warning("The audio is clipped (too loud).")
        except AudioError as exc:
            problems.append(str(exc)); st.error(str(exc))
    else:
        st.caption("No audio added.")
with pc[1]:
    st.markdown("**Text**")
    if transcript.strip():
        s = analyse_text(transcript)
        st.caption(f"{s['n_tokens']} tokens · {s['n_unique']} unique")
        st.write(", ".join(s["top_terms"][:6]) or "–")
    else:
        st.caption("No transcript yet.")
with pc[2]:
    st.markdown("**Image**")
    if image_bytes:
        try:
            d, _ = analyse_image(image_bytes)
            st.caption(f"{d['width']}×{d['height']} · mean RGB "
                       f"({d['mean_r']:.0f}, {d['mean_g']:.0f}, {d['mean_b']:.0f})")
        except ImageError as exc:
            problems.append(str(exc)); st.error(str(exc))
    else:
        st.caption("No photo added.")
if mood:
    st.markdown(f"Mood you selected: {pill(mood)}", unsafe_allow_html=True)

st.write("")
ready = bool(title.strip()) and mood and not problems and (audio_bytes or image_bytes or transcript.strip())
if not ready and not problems:
    hint = []
    if not title.strip(): hint.append("a title")
    if not mood: hint.append("a mood")
    if not (audio_bytes or image_bytes or transcript.strip()): hint.append("audio, a transcript or a photo")
    if hint:
        st.caption("Still needed: " + ", ".join(hint) + ".")
if st.button("💾 Save memory", type="primary", disabled=not ready):
    try:
        with st.spinner("Analysing and saving..."):
            suffix = os.path.splitext(img.name)[1].lower() if img else ".png"
            create_memory(title, transcript, mood, entry_date, audio_bytes, image_bytes, notes,
                          image_suffix=suffix)
        st.session_state.form_gen += 1
        st.session_state.just_saved = True
        st.cache_data.clear()
        st.rerun()
    except (AudioError, ImageError, ValueError) as exc:
        st.error(str(exc))
    except Exception:
        log.exception("Saving memory failed")
        st.error("Something went wrong while saving. Nothing was stored; please try again.")
