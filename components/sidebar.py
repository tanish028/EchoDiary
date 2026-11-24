from __future__ import annotations

from pathlib import Path

import streamlit as st

CSS = Path(__file__).resolve().parent.parent / "assets" / "styles.css"


def inject_css() -> None:
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_sidebar(count: int) -> None:
    with st.sidebar:
        st.markdown('<div class="brand">🎙️ EchoDiary</div>'
                    '<div class="tag">Your Voice. Your Emotions. Your Story.</div>',
                    unsafe_allow_html=True)
        st.caption(f"{count} saved {'memory' if count == 1 else 'memories'}")
        st.divider()
        st.caption("DA321 · Multimodal Data Processing and Learning I · IIT Guwahati")
        st.caption("Mood = what you select. EchoDiary does not recognise emotions.")
