"""Shared page helpers: entry selection and cached, error-tolerant media loaders."""
from __future__ import annotations

import streamlit as st

from modules.audio.preprocessing import AudioData, AudioError, load_wav_full
from modules.database import db
from modules.image.preprocessing import ImageError, load_image


@st.cache_data(show_spinner=False)
def _cached_audio(path: str, mtime: float) -> AudioData:
    with open(path, "rb") as f:
        return load_wav_full(f.read())


@st.cache_data(show_spinner=False)
def _cached_image(path: str, mtime: float):
    with open(path, "rb") as f:
        return load_image(f.read())


def load_entry_audio(e: dict, warn: bool = True) -> AudioData | None:
    """AudioData for an entry, or None (with a clear message) when there is no usable audio."""
    rel = e.get("audio_path")
    if not rel:
        return None
    p = db.abs_path(rel)
    if not p.exists():
        if warn:
            st.warning("The audio file of this memory is missing from disk, so audio analysis is unavailable. "
                       "You can attach a new recording from Edit.")
        return None
    try:
        return _cached_audio(str(p), p.stat().st_mtime)
    except (AudioError, OSError) as exc:
        if warn:
            st.warning(f"The stored audio file could not be analysed: {exc}")
        return None


def load_entry_image(e: dict, warn: bool = True):
    rel = e.get("image_path")
    if not rel:
        return None
    p = db.abs_path(rel)
    if not p.exists():
        if warn:
            st.warning("The photo of this memory is missing from disk, so image analysis is unavailable. "
                       "You can attach a new photo from Edit.")
        return None
    try:
        return _cached_image(str(p), p.stat().st_mtime)
    except (ImageError, OSError) as exc:
        if warn:
            st.warning(f"The stored photo could not be read: {exc}")
        return None


def pick_entry(entries: list[dict], key: str) -> dict | None:
    """Selectbox over entries; defaults to the one chosen elsewhere (session_state.selected_id)."""
    if not entries:
        return None
    ids = [e["id"] for e in entries]
    default = st.session_state.get("selected_id")
    idx = ids.index(default) if default in ids else 0
    label = lambda i: next(f'{e["title"]} · {e["entry_date"]} · {e["mood"]}' for e in entries if e["id"] == i)
    chosen = st.selectbox("Memory", ids, index=idx, format_func=label, key=key)
    st.session_state["selected_id"] = chosen
    return next(e for e in entries if e["id"] == chosen)


def open_entry(entry_id: int) -> None:
    """Navigate to the full diary entry (detail view)."""
    st.session_state["selected_id"] = entry_id
    st.session_state["detail_id"] = entry_id
    st.session_state.pop("edit_mode", None)
    st.switch_page("views/my_memories.py")


def empty_state() -> None:
    st.info("No memories yet. Create your first one to see the analysis.")
    st.page_link("views/create_memory.py", label="➕ Create a memory", icon="🎙️")
