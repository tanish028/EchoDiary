"""Bag-of-Words (PDF2 pp.198-206): unordered multiset of words, multiplicity kept."""
from __future__ import annotations

from collections import Counter

from .preprocessing import tokenize


def bag_of_words(text: str, remove_stopwords: bool = True) -> Counter:
    return Counter(tokenize(text, remove_stopwords))


def build_vocabulary(docs: list[str], remove_stopwords: bool = True) -> list[str]:
    vocab: set[str] = set()
    for d in docs:
        vocab.update(bag_of_words(d, remove_stopwords))
    return sorted(vocab)


def vocabulary_table(docs: list[str], remove_stopwords: bool = True) -> list[dict]:
    """Vocabulary construction (PDF2 pp.203-206): every distinct term with its total count and the number of
    documents containing it (DF). Sorted by DF, then count, then term."""
    total: Counter = Counter()
    df: Counter = Counter()
    for d in docs:
        b = bag_of_words(d, remove_stopwords)
        total.update(b)
        df.update(b.keys())
    return sorted(({"term": t, "count": total[t], "df": df[t]} for t in total),
                  key=lambda r: (-r["df"], -r["count"], r["term"]))
