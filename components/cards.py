"""HTML card helpers (all user text is escaped)."""
from __future__ import annotations

import base64
import html
import io
from functools import lru_cache

import streamlit as st
from PIL import Image

from modules import config
from modules.database import db


def esc(text) -> str:
    """Escape untrusted text (titles, notes, transcripts) before it goes into raw HTML."""
    return html.escape(str(text if text is not None else ""))


def header_line(e: dict) -> str:
    """HTML for 'title - date - mood' with every user-provided part escaped."""
    return (f'<b>{esc(e["title"])}</b> · {esc(e["entry_date"])} · mood you selected: {pill(e["mood"])}')


def pill(mood: str) -> str:
    c = config.MOOD_COLORS.get(mood, "#94a3b8")
    return f'<span class="pill" style="background:{c}">{html.escape(mood)}</span>'


@lru_cache(maxsize=256)
def _thumb_b64(path: str, mtime: float) -> str | None:
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((420, 300))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def thumb_html(entry: dict) -> str:
    p = db.abs_path(entry.get("image_path"))
    if p and p.exists():
        b = _thumb_b64(str(p), p.stat().st_mtime)
        if b:
            return f'<img class="thumb" src="data:image/jpeg;base64,{b}"/>'
    icon = "🎙️" if entry.get("audio_path") else "📝"
    return f'<div class="thumb-empty">{icon}</div>'


def entry_card(e: dict, score: float | None = None) -> None:
    snippet = html.escape((e.get("transcript") or "No transcript")[:110])
    demo = '<span class="pill ghost">demo data</span>' if e.get("is_demo") else ""
    sc = f'<span class="pill ghost">similarity {score:.2f}</span>' if score is not None else ""
    mods = " ".join(m for m, ok in (("🎙️", e.get("audio_path")), ("📝", e.get("transcript")),
                                    ("🖼️", e.get("image_path"))) if ok)
    st.markdown(
        f'<div class="card">{thumb_html(e)}<h4>{html.escape(e["title"])}</h4>'
        f'<div class="meta">{html.escape(e["entry_date"])} · {mods}</div>'
        f'<div style="margin-top:6px">{pill(e["mood"])}{demo}{sc}</div>'
        f'<div class="snippet">{snippet}{"…" if len(e.get("transcript") or "") > 110 else ""}</div></div>',
        unsafe_allow_html=True)


def metric(label: str, value) -> None:
    st.markdown(f'<div class="metric"><div class="v">{html.escape(str(value))}</div>'
                f'<div class="l">{html.escape(label)}</div></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def hero(title: str, sub: str) -> None:
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p></div>',
                unsafe_allow_html=True)
