"""ONE corpus definition shared by Search, Analysis and Insights (fixes inconsistent TF-IDF).

Corpus = the transcripts of ALL saved diary entries (titles/notes are not part of it). An entry without
tokens is kept in the list (so indices line up with `entries`) but is not counted in N (see tfidf.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from .retrieval import search_vectors
from .tfidf import corpus_tfidf


@dataclass
class Corpus:
    ids: list[int]
    texts: list[str]
    bows: list
    idf: dict
    vecs: list

    def index_of(self, entry_id: int) -> int:
        return self.ids.index(entry_id)

    def search(self, query: str, top_k: int | None = None):
        """-> (list of (entry_id, score), info)."""
        ranking, info = search_vectors(query, self.bows, self.idf, self.vecs, top_k)
        return [(self.ids[i], s) for i, s in ranking], info


def build_corpus(entries: list[dict]) -> Corpus:
    texts = [(e.get("transcript") or "") for e in entries]
    bows, idf, vecs = corpus_tfidf(texts)
    return Corpus([e["id"] for e in entries], texts, bows, idf, vecs)
