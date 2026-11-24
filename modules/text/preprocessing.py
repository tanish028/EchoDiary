"""Tokenisation (engineering decision; PDF2 pp.198-206 only says text is split into words).

Unicode-aware: a word is a run of letters, digits and combining marks (Devanagari vowel signs, virama,
accents), so English, Hindi (Devanagari), accented Latin and mixed Hindi-English text all work.
Punctuation, emoji, symbols and the Devanagari danda are separators; an apostrophe between letters stays
inside the word ("don't"). Text is NFC-normalised and lower-cased (no accent folding, no stemming, no
lemmatisation, no subword units - none of these is verified in the course material).
Stop-word removal follows PDF2 pp.203-206; the English and Hindi lists below are hand-written
engineering choices.
"""
from __future__ import annotations

import unicodedata

_APOSTROPHES = {"'", "’", "ʼ"}
_JOINERS = {"‌", "‍"}          # ZWNJ / ZWJ occur inside Devanagari conjuncts

STOPWORDS_EN = frozenset("""
a an the and or but if of to in on at by for with from as is are was were be been being am
i me my we our you your he she it its they them their this that these those there here
so than then too very can could would should will just not no do did does have has had
about into over after before up down out off again also what which who whom when where why how
all any both each few more most other some such only own same s t
""".split())

STOPWORDS_HI = frozenset("""
है हैं था थी थे हूँ हूं हो हुआ हुई हुए और का की के में को से पर यह वह ये वे मैं हम तुम आप
एक भी ही तो नहीं कि जो इस उस इन उन कर करना किया रहा रही रहे लिए साथ बहुत
""".split())

STOPWORDS = STOPWORDS_EN | STOPWORDS_HI


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or unicodedata.category(ch)[0] == "M"


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    if not text:
        return []
    s = unicodedata.normalize("NFC", text).lower()
    toks: list[str] = []
    cur: list[str] = []

    def flush():
        while cur and cur[-1] in _JOINERS:
            cur.pop()
        if cur:
            toks.append("".join(cur))
        cur.clear()

    n = len(s)
    for i, ch in enumerate(s):
        if _is_word_char(ch):
            cur.append(ch)
        elif ch in _APOSTROPHES and cur and i + 1 < n and _is_word_char(s[i + 1]):
            cur.append("'")
        elif ch in _JOINERS and cur and i + 1 < n and _is_word_char(s[i + 1]):
            cur.append(ch)
        else:
            flush()
    flush()
    toks = [t for t in toks if any(c.isalnum() for c in t)]       # drop marks-only fragments
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks
