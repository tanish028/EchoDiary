"""Vector-space diary search: rank entries by cosine similarity (PDF1 pp.152-225 inner product,
norm, angle; PDF2 p.213 TF-IDF ranking). Cosine as the ranking score is an engineering choice
built on the taught inner product / norm."""
from __future__ import annotations

import math

from .bow import bag_of_words
from .tfidf import corpus_tfidf, tfidf_vector, term_frequency


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def search_vectors(query: str, bows, idf, vecs, top_k: int | None = None):
    """Rank pre-computed corpus vectors against a query. Returns (ranking, info); ranking = [(doc_index, score)]
    with score > 0, best first. info['mode'] is 'tfidf' or 'tf-fallback' (TF-IDF all-zero, e.g. 1 entry)."""
    info = {"mode": "tfidf", "ignored_terms": []}
    if not bows or not query.strip():
        return [], info
    qbow = bag_of_words(query)
    info["ignored_terms"] = sorted(t for t in qbow if t not in idf)       # unseen query terms
    qvec = tfidf_vector(qbow, idf)
    scores = [cosine(qvec, v) for v in vecs]
    if not any(s > 0 for s in scores):
        # TF-IDF is all-zero when every shared term occurs in every entry (or N=1): fall back to TF.
        qtf = {t: w for t, w in term_frequency(qbow).items() if t in idf}
        scores = [cosine(qtf, term_frequency(b)) for b in bows]
        info["mode"] = "tf-fallback"
    ranking = sorted(((i, s) for i, s in enumerate(scores) if s > 1e-12), key=lambda p: -p[1])
    return (ranking[:top_k] if top_k else ranking), info


def search(query: str, texts: list[str], top_k: int | None = None):
    """Convenience wrapper: builds the corpus from `texts` and ranks it."""
    if not texts:
        return [], {"mode": "tfidf", "ignored_terms": []}
    bows, idf, vecs = corpus_tfidf(texts)
    return search_vectors(query, bows, idf, vecs, top_k)
