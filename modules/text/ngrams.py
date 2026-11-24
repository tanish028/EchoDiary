"""Bag of n-grams (PDF2 pp.215-216) and character bigram counts (PDF2 pp.190-196, counts only;
no smoothing is taught so no probabilities with smoothing are computed)."""
from __future__ import annotations

from collections import Counter

from .preprocessing import tokenize


def word_ngrams(text: str, n: int = 2, remove_stopwords: bool = False) -> Counter:
    toks = tokenize(text, remove_stopwords)
    return Counter(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))


def char_bigrams(text: str) -> Counter:
    s = " ".join(tokenize(text))
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def text_stats(text: str) -> dict:
    toks = tokenize(text)
    return {
        "n_tokens": len(toks),
        "n_unique": len(set(toks)),
        "type_token_ratio": (len(set(toks)) / len(toks)) if toks else 0.0,
        "avg_token_len": (sum(map(len, toks)) / len(toks)) if toks else 0.0,
    }
