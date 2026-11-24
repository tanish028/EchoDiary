"""TF-IDF exactly as in the slides (PDF2 pp.210-213):

    TF(t, d)  = count(t, d) / total words in d
    IDF(t)    = log10(N / DF(t))            (base 10 inferred from the worked example, 0.48 = log10 3)
    TF-IDF    = TF * IDF

Terms present in every document therefore get weight 0 (as the slides note). With a corpus of one
entry every weight is 0; `search` then falls back to plain TF (engineering decision, flagged in UI).
"""
from __future__ import annotations

import math
from collections import Counter

from .bow import bag_of_words


def term_frequency(bow: Counter) -> dict[str, float]:
    total = sum(bow.values())
    return {t: c / total for t, c in bow.items()} if total else {}


def document_frequency(bows: list[Counter]) -> Counter:
    df: Counter = Counter()
    for b in bows:
        df.update(b.keys())
    return df


def inverse_document_frequency(bows: list[Counter]) -> dict[str, float]:
    n = sum(1 for b in bows if b)          # an entry with no tokens is not a document (engineering decision)
    if n == 0:
        return {}
    return {t: math.log10(n / d) for t, d in document_frequency(bows).items()}


def tfidf_vector(bow: Counter, idf: dict[str, float]) -> dict[str, float]:
    """Sparse TF-IDF vector; terms unseen in the corpus (no IDF) are dropped."""
    tf = term_frequency(bow)
    return {t: w * idf[t] for t, w in tf.items() if t in idf}


def corpus_tfidf(texts: list[str]):
    bows = [bag_of_words(t) for t in texts]
    idf = inverse_document_frequency(bows)
    return bows, idf, [tfidf_vector(b, idf) for b in bows]


def top_keywords(vec: dict[str, float], k: int = 10) -> list[tuple[str, float]]:
    return sorted(((t, w) for t, w in vec.items() if w > 0), key=lambda p: (-p[1], p[0]))[:k]
