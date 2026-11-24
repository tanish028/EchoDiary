import math

import pytest

from modules.text.bow import bag_of_words, build_vocabulary, vocabulary_table
from modules.text.corpus import build_corpus
from modules.text.ngrams import char_bigrams, word_ngrams
from modules.text.preprocessing import tokenize
from modules.text.retrieval import cosine, search
from modules.text.tfidf import corpus_tfidf, inverse_document_frequency, term_frequency, top_keywords


# ------------------------------------------------------------------ tokenisation
def test_english():
    assert tokenize("Hello, World! It's 5pm.") == ["hello", "world", "it's", "5pm"]
    assert tokenize("don't can’t") == ["don't", "can't"]


def test_hindi_devanagari():
    assert tokenize("मैं आज बहुत खुश हूँ।") == ["मैं", "आज", "बहुत", "खुश", "हूँ"]
    assert tokenize("क्षत्रिय ज़िंदगी") == ["क्षत्रिय", "ज़िंदगी"]          # conjuncts, nukta, anusvara kept whole


def test_mixed_hindi_english():
    assert tokenize("Aaj मौसम accha है, friends के साथ!") == ["aaj", "मौसम", "accha", "है", "friends", "के", "साथ"]


def test_accented_latin():
    assert tokenize("Café naïve Zoë Ünï") == ["café", "naïve", "zoë", "ünï"]
    assert tokenize("école") == ["école"]                          # NFC-normalised combining accent


def test_empty_punctuation_symbols():
    assert tokenize("") == [] and tokenize(None) == [] and tokenize("   ") == []
    assert tokenize("!!! ... --- ??? ।।") == []
    assert tokenize("😀 ♥ ©") == []
    assert tokenize("well-known state_of_art") == ["well", "known", "state", "of", "art"]


def test_stopwords_english_and_hindi():
    assert tokenize("the cat is on the mat", remove_stopwords=True) == ["cat", "mat"]
    assert tokenize("मैं आज खुश हूँ और है", remove_stopwords=True) == ["आज", "खुश"]


# ------------------------------------------------------------------ BoW / vocabulary / TF-IDF
def test_bow_and_vocabulary():
    assert bag_of_words("cat sat the cat")["cat"] == 2
    assert build_vocabulary(["cat sat", "dog sat"]) == ["cat", "dog", "sat"]
    vt = vocabulary_table(["cat sat", "dog sat", "sat"])
    assert vt[0] == {"term": "sat", "count": 3, "df": 3}
    assert {r["term"] for r in vt} == {"cat", "dog", "sat"}
    assert vocabulary_table([]) == [] and vocabulary_table(["", "the"]) == []


def test_tf_idf_formulas():
    docs = ["cat sat mat", "dog sat outside", "cat dog"]
    bows, idf, vecs = corpus_tfidf(docs)
    assert math.isclose(idf["mat"], math.log10(3))
    assert math.isclose(idf["sat"], math.log10(3 / 2))
    assert math.isclose(term_frequency(bows[0])["cat"], 1 / 3)
    assert math.isclose(vecs[0]["mat"], (1 / 3) * math.log10(3))


def test_common_word_scores_zero_and_keywords():
    _, idf, vecs = corpus_tfidf(["apple pie", "apple tart"])
    assert vecs[0]["apple"] == 0.0 and vecs[0]["pie"] > 0
    assert top_keywords(vecs[0])[0][0] == "pie"


def test_empty_documents_do_not_count_in_n():
    assert inverse_document_frequency([]) == {}
    bows, idf, _ = corpus_tfidf(["alpha beta", "", "alpha gamma"])
    assert math.isclose(idf["beta"], math.log10(2 / 1))                # N = 2 (the empty entry is not a document)


def test_ngrams():
    assert word_ngrams("a b c", 2) == {"a b": 1, "b c": 1}
    assert char_bigrams("ab")["ab"] == 1
    assert word_ngrams("one", 3) == {}
    assert word_ngrams("आज मौसम अच्छा है", 2)["आज मौसम"] == 1


# ------------------------------------------------------------------ search
def test_search_ranking_english():
    docs = ["I walked by the lake at sunrise", "exam stress and no sleep", "dinner with friends"]
    ranking, info = search("lake sunrise walk", docs)
    assert ranking[0][0] == 0 and info["mode"] == "tfidf"


def test_search_hindi_and_mixed_queries():
    docs = ["मैं आज बहुत खुश हूँ", "कल परीक्षा का तनाव था", "Dinner with friends और खुशी"]
    r, _ = search("खुश", docs)
    assert [i for i, _ in r] == [0]
    r, _ = search("परीक्षा तनाव", docs)
    assert r[0][0] == 1
    r, _ = search("friends खुशी", docs)
    assert r[0][0] == 2
    r, _ = search("café", ["Café au lait", "tea"])
    assert r[0][0] == 0


def test_search_edge_cases():
    assert search("anything", [])[0] == []
    assert search("   ", ["a b"])[0] == []
    r, info = search("lake", ["the lake was calm"])
    assert r and r[0][0] == 0 and info["mode"] == "tf-fallback"
    r, info = search("zzzz unseen", ["alpha beta", "gamma delta"])
    assert r == [] and "zzzz" in info["ignored_terms"]
    r, _ = search("hello", ["", "!!!", "hello world"])
    assert r[0][0] == 2


def test_cosine_zero_vectors():
    assert cosine({}, {"a": 1}) == 0.0
    assert cosine({"a": 0.0}, {"a": 1.0}) == 0.0
    assert math.isclose(cosine({"a": 1, "b": 1}, {"a": 1, "b": 1}), 1.0)


# ------------------------------------------------------------------ one corpus for every page
def test_corpus_is_the_single_source_of_truth():
    entries = [{"id": 5, "transcript": "lake sunrise walk"}, {"id": 9, "transcript": "exam stress"},
               {"id": 2, "transcript": ""}, {"id": 7, "transcript": "lake dinner"}]
    c = build_corpus(entries)
    assert c.ids == [5, 9, 2, 7] and c.index_of(2) == 2
    bows, idf, vecs = corpus_tfidf([e["transcript"] for e in entries])
    assert c.idf == idf and c.vecs == vecs
    hits, info = c.search("lake")
    assert {i for i, _ in hits} == {5, 7}
    # Search, Analysis (corpus.vecs) and Insights (corpus.vecs) agree on the weight of a term
    assert c.vecs[c.index_of(5)]["sunrise"] == pytest.approx(
        (1 / 3) * math.log10(3 / 1))                                  # TF 1/3, N=3 non-empty docs, DF=1
