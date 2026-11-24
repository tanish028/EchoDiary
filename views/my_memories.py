import os
from datetime import date

import streamlit as st

from components.cards import entry_card, esc, header_line, hero, note, pill
from components.common import empty_state
from modules import config
from modules.audio.preprocessing import AudioError, load_wav_full
from modules.database import db
from modules.image.preprocessing import ImageError
from modules.pipeline import analyse_image, update_memory
from modules.text.corpus import build_corpus

log = config.get_logger("ui.memories")

hero("My memories", "Browse, search, edit and manage everything you have saved.")
all_entries = db.list_entries()
if not all_entries:
    empty_state(); st.stop()


def go_analysis(eid):
    st.session_state["selected_id"] = eid
    st.switch_page("views/analysis.py")


# =============================================================== detail view
def detail_view(d: dict) -> None:
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)
    if st.button("← All memories"):
        st.session_state.pop("detail_id", None); st.session_state.pop("edit_mode", None); st.rerun()
    st.markdown(f"<h2 style='margin-bottom:2px'>{esc(d['title'])}</h2>", unsafe_allow_html=True)
    saved = f"saved {esc(d['created_at'])}" + (f" · edited {esc(d['updated_at'])}" if d.get("updated_at") else "")
    st.markdown(f"{esc(d['entry_date'])} · {saved} · mood you selected: {pill(d['mood'])}",
                unsafe_allow_html=True)
    miss = db.media_missing(d)
    l, r = st.columns([1.2, 1])
    with l:
        st.markdown("**Transcript**")
        st.write(d["transcript"] or "_No transcript_")
        if d["notes"]:
            st.caption(d["notes"])
        ap = db.abs_path(d["audio_path"])
        if miss["audio"]:
            st.warning("The audio file of this memory is missing from disk. Use Edit to attach a new recording.")
        elif ap:
            st.audio(str(ap))
            af = d["audio_features"] or {}
            if af.get("truncated"):
                st.caption(f"Recording is {af['duration_s']:.0f} s; analysis covers the first {af['analysed_duration_s']:.0f} s.")
        m = st.columns(3)
        tf = d["text_features"] or {}
        if d["audio_features"]:
            m[0].metric("Audio", f"{d['audio_features']['duration_s']:.1f} s")
        if tf:
            m[1].metric("Words", tf.get("n_tokens", "–"))
        if d["image_features"]:
            m[2].metric("Photo brightness", f"{d['image_features']['brightness']:.0f}/255")
    with r:
        ip = db.abs_path(d["image_path"])
        if miss["image"]:
            st.warning("The photo of this memory is missing from disk. Use Edit to attach a new photo.")
        elif ip:
            st.image(str(ip), width="stretch")

    b = st.columns(5)
    if b[0].button("✏️ Edit", width="stretch"):
        st.session_state["edit_mode"] = not st.session_state.get("edit_mode")
        st.rerun()
    if b[1].button("📊 Analysis", width="stretch"):
        go_analysis(d["id"])
    if b[2].button("🔎 Similar photos", width="stretch", disabled=not d["image_path"] or miss["image"]):
        st.session_state["vs_source_id"] = d["id"]
        st.switch_page("views/visual_search.py")
    if b[3].button("🔗 Insights", width="stretch"):
        st.session_state["selected_id"] = d["id"]; st.switch_page("views/insights.py")
    if b[4].button("🗑️ Delete", width="stretch"):
        st.session_state["confirm_del"] = d["id"]

    if st.session_state.get("confirm_del") == d["id"]:
        st.warning(f"Delete “{d['title']}” and its audio/photo files? This cannot be undone.")
        y, n = st.columns(2)
        if y.button("Yes, delete", type="primary"):
            db.delete_entry(d["id"])
            for k in ("confirm_del", "detail_id", "edit_mode"):
                st.session_state.pop(k, None)
            st.cache_data.clear(); st.session_state["flash_list"] = "Memory deleted."; st.rerun()
        if n.button("Cancel"):
            st.session_state.pop("confirm_del", None); st.rerun()

    if st.session_state.get("edit_mode"):
        edit_panel(d, miss)


def _reset_edit_widgets(entry_id: int) -> None:
    """Forget the edit form's widget state so it re-opens with the stored values."""
    for key in [k for k in st.session_state if str(k).startswith(f"e{entry_id}")]:
        del st.session_state[key]


def edit_panel(d: dict, miss: dict) -> None:
    st.divider()
    st.markdown("### ✏️ Edit this memory")
    k = f"e{d['id']}"
    c1, c2 = st.columns(2)
    title = c1.text_input("Title", d["title"], key=f"{k}t", max_chars=80)
    ed = c2.date_input("Date", date.fromisoformat(d["entry_date"]), max_value=date.today(), key=f"{k}d")
    mood = c1.selectbox("Mood (your choice)", config.MOODS, index=config.MOODS.index(d["mood"]), key=f"{k}m")
    notes = c2.text_input("Notes", d["notes"] or "", key=f"{k}n", max_chars=200)
    transcript = st.text_area("Transcript", d["transcript"], height=150, key=f"{k}x")

    a, i = st.columns(2)
    with a:
        has_a = bool(d["audio_path"])
        act_a = st.radio("Audio", (["Keep current", "Replace", "Remove"] if has_a else ["None", "Add"]),
                         key=f"{k}ra", horizontal=True)
        new_a = st.file_uploader("New WAV file", type=["wav"], key=f"{k}ua") if act_a in ("Replace", "Add") else None
    with i:
        has_i = bool(d["image_path"])
        act_i = st.radio("Photo", (["Keep current", "Replace", "Remove"] if has_i else ["None", "Add"]),
                         key=f"{k}ri", horizontal=True)
        new_i = (st.file_uploader("New photo", type=["png", "jpg", "jpeg", "webp", "bmp"], key=f"{k}ui")
                 if act_i in ("Replace", "Add") else None)
        if new_i:
            st.image(new_i, width=220)

    destructive = []
    if act_a == "Replace" and has_a and new_a: destructive.append("replace the current audio file")
    if act_a == "Remove": destructive.append("remove the current audio file")
    if act_i == "Replace" and has_i and new_i: destructive.append("replace the current photo")
    if act_i == "Remove": destructive.append("remove the current photo")
    confirmed = True
    if destructive:
        confirmed = st.checkbox("I understand this will permanently " + " and ".join(destructive) +
                                " (the old file is deleted once the change is saved).", key=f"{k}ok")
    pending_upload = (act_a in ("Replace", "Add") and not new_a) or (act_i in ("Replace", "Add") and not new_i)
    if pending_upload:
        st.caption("Choose the new file, or switch the option back to keep the current attachment.")
    s, c = st.columns([1, 1])
    if s.button("💾 Save changes", type="primary", disabled=not confirmed or pending_upload):
        try:
            with st.spinner("Re-analysing changed parts and saving…"):
                res = update_memory(
                    d["id"], title=title, transcript=transcript, mood=mood, entry_date=ed, notes=notes,
                    new_audio=new_a.getvalue() if new_a else None, remove_audio=act_a == "Remove",
                    new_image=new_i.getvalue() if new_i else None,
                    new_image_suffix=os.path.splitext(new_i.name)[1].lower() if new_i else ".png",
                    remove_image=act_i == "Remove")
            st.cache_data.clear()
            _reset_edit_widgets(d["id"])
            st.session_state.pop("edit_mode", None)
            what = ", ".join(res.recomputed) if res.recomputed else "no analysis needed"
            st.session_state["flash"] = f"Changes saved. Re-analysed: {what}."
            st.rerun()
        except (AudioError, ImageError, ValueError) as exc:
            st.error(f"Nothing was changed. {exc}")
        except Exception:
            log.exception("Editing memory %s failed", d["id"])
            st.error("Something went wrong while saving. Your original memory is unchanged.")
    if c.button("Cancel"):
        _reset_edit_widgets(d["id"])
        st.session_state.pop("edit_mode", None); st.rerun()


did = st.session_state.get("detail_id")
d = db.get_entry(did) if did else None
if did and not d:
    st.session_state.pop("detail_id", None)
    st.warning("That memory no longer exists.")
if d:
    detail_view(d)
    st.stop()

# =============================================================== list view
flash = st.session_state.pop("flash_list", None)
if flash:
    st.success(flash)
q = st.text_input("🔎 Search your diary by transcript (vector-space TF-IDF similarity)",
                  placeholder="e.g. lake sunrise walk  ·  खुश  ·  café")
f1, f2 = st.columns([1, 2])
mood = f1.selectbox("Mood", ["All"] + config.MOODS)
dates = [e["entry_date"] for e in all_entries]
dmin, dmax = date.fromisoformat(min(dates)), date.fromisoformat(max(dates))
if dmin < dmax:
    rng = f2.slider("Date range", dmin, dmax, (dmin, dmax))
else:
    rng = (dmin, dmax); f2.caption(f"All entries are from {dmin}.")

entries = db.list_entries(mood, rng[0].isoformat(), rng[1].isoformat())
scores: dict[int, float] = {}
if q.strip():
    corpus = build_corpus(all_entries)          # same corpus as Analysis and Insights
    ranking, info = corpus.search(q)
    scores = dict(ranking)
    entries = sorted([e for e in entries if e["id"] in scores], key=lambda e: -scores[e["id"]])
    if info["ignored_terms"]:
        st.caption("Not found in any transcript: " + ", ".join(info["ignored_terms"]))
    if info["mode"] == "tf-fallback":
        note("Every shared word appears in every transcript (or there is only one), so IDF = 0 and TF-IDF "
             "cannot rank. Ranking by term frequency instead <span class='eng'>engineering fallback</span>.", warn=True)
    if not entries:
        st.warning("No memories match that search. Try other words, or clear the mood/date filters.")

st.caption(f"{len(entries)} of {len(all_entries)} memories")
cols = st.columns(3)
for i, e in enumerate(entries):
    with cols[i % 3]:
        entry_card(e, scores.get(e["id"]))
        if st.button("Open", key=f"open_{e['id']}", width="stretch"):
            st.session_state["detail_id"] = e["id"]
            st.session_state["selected_id"] = e["id"]
            st.rerun()

if any(e["is_demo"] for e in all_entries):
    st.divider()
    if st.button("Remove all [DEMO] entries"):
        db.delete_demo_entries(); st.cache_data.clear(); st.rerun()
