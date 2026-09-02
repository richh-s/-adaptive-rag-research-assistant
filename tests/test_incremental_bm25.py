"""Tests for the incremental BM25 index.

The load-bearing test is equivalence with `rank_bm25.BM25Okapi` on randomised corpora. A
hand-rolled scorer replacing a well-tested library is only defensible if it can be shown to
rank identically — a subtle divergence here would silently change retrieval quality across the
whole system, and no other test in the suite would notice.
"""

import random

import pytest
from rank_bm25 import BM25Okapi

from rag_assistant.retrieval.incremental_bm25 import IncrementalBM25

VOCABULARY = [
    "anthropic",
    "mistral",
    "openai",
    "claude",
    "mixtral",
    "funding",
    "safety",
    "paris",
    "model",
    "research",
    "alignment",
    "series",
    "round",
    "open",
    "weights",
    "enterprise",
]


def random_corpus(rng: random.Random, documents: int, length: int = 25) -> list[list[str]]:
    return [
        [rng.choice(VOCABULARY) for _ in range(rng.randint(length // 2, length))]
        for _ in range(documents)
    ]


def reference_scores(corpus: list[list[str]], query: list[str]) -> list[float]:
    return list(BM25Okapi(corpus).get_scores(query))


def incremental_scores(corpus: list[list[str]], query: list[str]) -> list[float]:
    index = IncrementalBM25()
    for position, tokens in enumerate(corpus):
        index.add(str(position), tokens)
    scored = index.scores(query)
    return [scored[str(position)] for position in range(len(corpus))]


# ---- equivalence with the library it replaces ----


@pytest.mark.parametrize("seed", range(8))
def test_scores_match_rank_bm25_on_random_corpora(seed):
    rng = random.Random(seed)
    corpus = random_corpus(rng, documents=rng.randint(3, 25))
    query = [rng.choice(VOCABULARY) for _ in range(rng.randint(1, 4))]

    assert incremental_scores(corpus, query) == pytest.approx(
        reference_scores(corpus, query), rel=1e-9, abs=1e-12
    )


def test_scores_match_when_a_term_appears_in_most_documents():
    """The negative-IDF path: a term in more than half the corpus gets a negative raw IDF,
    which BM25Okapi replaces with an epsilon floor. Reproducing that exactly is the fiddliest
    part of the formula and the easiest to get silently wrong."""
    corpus = [["safety", "model"], ["safety", "research"], ["safety", "paris"], ["mistral"]]
    query = ["safety"]

    assert incremental_scores(corpus, query) == pytest.approx(reference_scores(corpus, query))


def test_scores_match_with_repeated_terms_in_one_document():
    corpus = [["claude"] * 12, ["claude", "mistral"], ["openai"]]

    assert incremental_scores(corpus, ["claude"]) == pytest.approx(
        reference_scores(corpus, ["claude"])
    )


def test_scores_match_with_widely_varying_document_lengths():
    """Length normalisation is the other place a reimplementation drifts."""
    corpus = [["funding"], ["funding"] + ["filler"] * 200, ["safety", "funding", "round"]]

    assert incremental_scores(corpus, ["funding"]) == pytest.approx(
        reference_scores(corpus, ["funding"])
    )


def test_an_index_matches_a_rebuild_after_adds_and_removes():
    """The actual claim: maintaining the index incrementally lands in the same place a full
    rebuild would."""
    rng = random.Random(99)
    corpus = random_corpus(rng, documents=12)
    index = IncrementalBM25()
    for position, tokens in enumerate(corpus):
        index.add(str(position), tokens)

    # Churn: drop three documents, add two, replace one.
    for removed in ("2", "5", "7"):
        index.remove(removed)
    surviving = [
        (str(i), tokens) for i, tokens in enumerate(corpus) if str(i) not in {"2", "5", "7"}
    ]
    new_docs = random_corpus(rng, documents=2)
    for offset, tokens in enumerate(new_docs):
        index.add(f"new{offset}", tokens)
        surviving.append((f"new{offset}", tokens))
    replacement = random_corpus(rng, documents=1)[0]
    index.add("0", replacement)
    surviving = [(doc_id, replacement if doc_id == "0" else tokens) for doc_id, tokens in surviving]

    query = ["funding", "safety"]
    expected = dict(
        zip(
            [doc_id for doc_id, _ in surviving],
            reference_scores([tokens for _, tokens in surviving], query),
        )
    )

    assert index.scores(query) == pytest.approx(expected)


# ---- maintenance semantics ----


def test_replacing_a_document_does_not_double_count_its_terms():
    index = IncrementalBM25()
    index.add("a", ["claude", "claude"])
    index.add("b", ["mistral"])

    index.add("a", ["claude"])

    assert len(index) == 2
    assert index.average_length == pytest.approx(1.0)


def test_removing_the_last_document_containing_a_term_drops_it_from_the_vocabulary():
    """A term left at df=0 would linger and skew the average IDF the epsilon floor comes
    from -- a leak that only shows up as gradually wrong scores."""
    index = IncrementalBM25()
    index.add("a", ["rare"])
    index.add("b", ["common"])

    index.remove("a")

    assert "rare" not in index._doc_freq


def test_removing_an_unknown_document_is_a_no_op():
    index = IncrementalBM25()
    index.add("a", ["claude"])

    index.remove("does-not-exist")

    assert len(index) == 1


def test_an_empty_index_scores_nothing():
    assert IncrementalBM25().scores(["claude"]) == {}


def test_documents_matching_no_query_term_score_zero_and_are_present():
    """Callers filter on `> 0`; omitting non-matching documents would change that contract."""
    index = IncrementalBM25()
    index.add("a", ["claude"])
    for position, term in enumerate(["mistral", "openai", "paris", "funding"]):
        index.add(f"other{position}", [term])

    scored = index.scores(["claude"])

    assert scored["a"] > 0.0
    assert all(scored[f"other{i}"] == 0.0 for i in range(4))


def test_a_term_in_half_a_tiny_corpus_scores_zero():
    """Not a bug in this implementation -- BM25's IDF is log((N-df+0.5)/(df+0.5)), which is
    exactly 0 when df is half of N. It is why bm25_search's `score > 0` filter returns
    nothing on two-document fixtures, and it matches rank_bm25 precisely."""
    corpus = [["claude"], ["mistral"]]

    assert incremental_scores(corpus, ["claude"]) == pytest.approx(
        reference_scores(corpus, ["claude"])
    )
    assert incremental_scores(corpus, ["claude"]) == [0.0, 0.0]


def test_an_unknown_query_term_contributes_nothing():
    index = IncrementalBM25()
    index.add("a", ["claude"])

    assert index.scores(["nonexistentterm"]) == {"a": 0.0}


def test_average_length_tracks_add_and_remove():
    index = IncrementalBM25()
    index.add("a", ["x"] * 10)
    index.add("b", ["y"] * 20)
    assert index.average_length == pytest.approx(15.0)

    index.remove("a")

    assert index.average_length == pytest.approx(20.0)
