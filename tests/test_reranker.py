"""Tests for cross-encoder reranking.

The backends themselves are a network call and a torch model, so they're stubbed. What's
worth testing is the contract around them: that reranking is off unless configured, that it
only reorders the shortlist, that its score becomes the ranking field everything downstream
reads, and — most importantly — that every failure mode degrades to fusion order instead of
breaking a request.
"""

import pytest

from rag_assistant.graph.nodes.fuse import fuse_results
from rag_assistant.retrieval import reranker as reranker_module
from rag_assistant.retrieval.reranker import Reranker, get_reranker, rerank_documents
from rag_assistant.schemas.models import FusedDocument, RetrievedDoc, SubQueryResult


class ScriptedReranker(Reranker):
    """Scores by position in a list of 'best' substrings, so the expected order is explicit."""

    def __init__(self, ranking: list[str]):
        self.ranking = ranking
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, len(documents)))
        scores = []
        for document in documents:
            match = next((i for i, term in enumerate(self.ranking) if term in document), None)
            scores.append(1.0 - (match / 10.0) if match is not None else 0.0)
        return scores


def doc(content: str, rrf_score: float = 0.1, source: str = "a.md") -> FusedDocument:
    return FusedDocument(content=content, source_id=source, rrf_score=rrf_score)


@pytest.fixture
def use_reranker(monkeypatch):
    def _install(instance):
        monkeypatch.setattr(reranker_module, "get_reranker", lambda: instance)
        return instance

    return _install


# ---- configuration ----


def test_reranking_is_off_by_default():
    assert get_reranker() is None


def test_documents_pass_through_untouched_when_disabled():
    documents = [doc("alpha", 0.3), doc("beta", 0.2)]

    assert rerank_documents("q", documents) == documents


def test_cohere_without_a_key_disables_reranking_rather_than_failing(monkeypatch):
    monkeypatch.setenv("RERANKER", "cohere")
    monkeypatch.setenv("COHERE_API_KEY", "")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    reranker_module.reset_reranker_cache()

    assert get_reranker() is None


def test_a_missing_optional_extra_disables_reranking_rather_than_failing(monkeypatch):
    """`RERANKER=cross_encoder` without the `rerank-local` extra installed must cost ranking
    quality, not availability."""
    monkeypatch.setenv("RERANKER", "cross_encoder")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    reranker_module.reset_reranker_cache()

    assert get_reranker() is None


# ---- reordering ----


def test_reranking_reorders_by_relevance_not_fusion_consensus(use_reranker):
    """The failure RRF has by construction: a document every retriever returned for lexical
    reasons outranks the one that actually answers the question."""
    use_reranker(ScriptedReranker(["answer", "popular"]))
    documents = [doc("popular but off topic", 0.9), doc("the actual answer", 0.1)]

    reranked = rerank_documents("what is the answer?", documents)

    assert [d.content for d in reranked] == ["the actual answer", "popular but off topic"]


def test_the_rerank_score_replaces_the_ranking_field(use_reranker):
    """Downstream consumers -- grading, citation order, the context budget -- read one field
    rather than each knowing whether reranking happened to run."""
    use_reranker(ScriptedReranker(["answer"]))
    documents = [doc("the actual answer", 0.01)]

    reranked = rerank_documents("q", documents)

    assert reranked[0].rrf_score == 1.0


def test_only_the_shortlist_is_scored(use_reranker):
    """Reranking is per-pair work; scoring everything fusion returned would scale cost with
    retrieval breadth instead of with what synthesis can use."""
    scripted = use_reranker(ScriptedReranker([]))
    documents = [doc(f"doc {i}", 1.0 - i / 100) for i in range(30)]

    rerank_documents("q", documents, top_n=5)

    assert scripted.calls == [("q", 5)]


def test_the_untouched_tail_keeps_its_fusion_order(use_reranker):
    use_reranker(ScriptedReranker([]))
    documents = [doc(f"doc {i}", 1.0 - i / 100) for i in range(6)]

    reranked = rerank_documents("q", documents, top_n=2)

    assert [d.content for d in reranked[2:]] == ["doc 2", "doc 3", "doc 4", "doc 5"]


def test_original_documents_are_not_mutated(use_reranker):
    use_reranker(ScriptedReranker(["answer"]))
    original = doc("the actual answer", 0.01)

    rerank_documents("q", [original])

    assert original.rrf_score == 0.01


# ---- degradation ----


def test_a_failing_reranker_falls_back_to_fusion_order(use_reranker):
    class Exploding(Reranker):
        def score(self, query, documents):
            raise RuntimeError("rerank endpoint down")

    use_reranker(Exploding())
    documents = [doc("alpha", 0.3), doc("beta", 0.2)]

    assert rerank_documents("q", documents) == documents


def test_a_wrong_length_score_list_falls_back_to_fusion_order(use_reranker):
    """Misaligned scores would attach one document's relevance to another -- worse than not
    reranking, because the result looks ordered."""

    class Misaligned(Reranker):
        def score(self, query, documents):
            return [1.0]

    use_reranker(Misaligned())
    documents = [doc("alpha", 0.3), doc("beta", 0.2)]

    assert rerank_documents("q", documents) == documents


def test_empty_input_is_handled(use_reranker):
    use_reranker(ScriptedReranker([]))

    assert rerank_documents("q", []) == []


# ---- graph wiring ----


def test_the_fuse_node_applies_reranking(use_reranker):
    use_reranker(ScriptedReranker(["answer"]))
    state = {
        "question": "what is the answer?",
        "vector_results": [
            SubQueryResult(
                sub_query="q",
                docs=[
                    RetrievedDoc(content="popular but off topic", source_id="a.md"),
                    RetrievedDoc(content="the actual answer", source_id="b.md"),
                ],
            )
        ],
    }

    result = fuse_results(state)

    assert result["fused_documents"][0].content == "the actual answer"


def test_the_fuse_node_skips_reranking_without_a_question(use_reranker):
    """A cross-encoder scored against an empty query returns meaningless relevance, and
    reordering by noise is worse than not reranking."""
    scripted = use_reranker(ScriptedReranker(["answer"]))
    state = {
        "vector_results": [
            SubQueryResult(sub_query="q", docs=[RetrievedDoc(content="a", source_id="a.md")])
        ]
    }

    fuse_results(state)

    assert scripted.calls == []
