"""The corrective loop's first escalation: re-ask the corpus before leaving it."""

from unittest.mock import AsyncMock

from langchain_core.documents import Document

from rag_assistant.config import get_settings
from rag_assistant.graph.nodes.refine import refine_retrieval
from rag_assistant.schemas.models import RefinedQueries, RetrievedDoc


class _Retriever:
    def __init__(self, calls):
        self._calls = calls

    def invoke(self, query):
        self._calls.append(query)
        return [Document(page_content=f"doc for {query}", metadata={"source": "corpus.md"})]


def _patch(monkeypatch, *, rewrite_to=None, fail_rewrite=False):
    """Stubs the rewrite call and both retrieval paths, returning what each one saw."""
    seen = {"vector_queries": [], "bm25_queries": [], "k": []}

    llm = AsyncMock()
    if fail_rewrite:
        llm.ainvoke.side_effect = RuntimeError("provider down")
    else:
        llm.ainvoke.return_value = RefinedQueries(sub_queries=rewrite_to or [])
    monkeypatch.setattr("rag_assistant.graph.nodes.refine.get_structured_llm", lambda *a, **k: llm)

    def fake_get_retriever(k, owner, filters, principals=None):
        seen["k"].append(k)
        seen["owner"] = owner
        seen["filters"] = filters
        return _Retriever(seen["vector_queries"])

    def fake_bm25(query, k, owner, filters, principals=None):
        seen["bm25_queries"].append(query)
        return [RetrievedDoc(content="kw", metadata={}, source_id="corpus.md")]

    monkeypatch.setattr("rag_assistant.graph.nodes.refine.get_retriever", fake_get_retriever)
    monkeypatch.setattr("rag_assistant.graph.nodes.refine.bm25_search", fake_bm25)
    return seen


async def test_refinement_searches_the_rewritten_queries_on_both_local_paths(monkeypatch):
    seen = _patch(monkeypatch, rewrite_to=["annual revenue figure", "reported turnover"])

    result = await refine_retrieval(
        {
            "question": "how much money did they make?",
            "sub_queries": ["how much money"],
            "owner": "public",
        }
    )

    assert seen["vector_queries"] == ["annual revenue figure", "reported turnover"]
    assert seen["bm25_queries"] == ["annual revenue figure", "reported turnover"]
    assert result["refined_sub_queries"] == ["annual revenue figure", "reported turnover"]
    assert result["refinement_attempted"] is True


async def test_refinement_widens_k_as_well_as_rewriting(monkeypatch):
    """Two different failures -- the right document ranked below the cutoff, and the right
    document not matching at all. A retry this expensive should address both."""
    monkeypatch.setenv("RETRIEVAL_K", "4")
    monkeypatch.setenv("REFINE_K_MULTIPLIER", "3")
    get_settings.cache_clear()
    seen = _patch(monkeypatch, rewrite_to=["rephrased"])

    await refine_retrieval({"question": "q", "sub_queries": ["q"], "owner": "public"})

    assert seen["k"] == [12]


async def test_a_failed_rewrite_still_retries_wider(monkeypatch):
    """A provider hiccup should cost the refinement its rewrite, not the whole second
    attempt -- the widened search is free of the LLM and still worth running."""
    seen = _patch(monkeypatch, fail_rewrite=True)

    result = await refine_retrieval(
        {"question": "q", "sub_queries": ["original one", "original two"], "owner": "public"}
    )

    assert seen["vector_queries"] == ["original one", "original two"]
    assert result["refinement_attempted"] is True


async def test_an_empty_rewrite_falls_back_to_the_originals(monkeypatch):
    """A model that returns [] or whitespace must not turn the retry into a no-op."""
    seen = _patch(monkeypatch, rewrite_to=["", "   "])

    await refine_retrieval({"question": "q", "sub_queries": ["original"], "owner": "public"})

    assert seen["vector_queries"] == ["original"]


async def test_refinement_stays_inside_the_tenant_and_its_filters(monkeypatch):
    """The node builds its own retrieval calls, so it carries the scope itself -- a retry
    that searched every tenant's documents would be a data-isolation bug reachable only by
    asking a question the corpus answers badly."""
    seen = _patch(monkeypatch, rewrite_to=["rephrased"])
    filters = object()

    await refine_retrieval(
        {"question": "q", "sub_queries": ["q"], "owner": "alice", "filters": filters}
    )

    assert seen["owner"] == "alice"
    assert seen["filters"] is filters


async def test_results_are_added_to_the_first_attempt_rather_than_replacing_it(monkeypatch):
    """`vector_results`/`bm25_results` carry an `operator.add` reducer, so returning the new
    documents appends them. Fusion then ranks across both attempts, and a document both
    passes found earns the rank consensus -- discarding the first attempt would throw away
    the evidence the grader was looking at."""
    _patch(monkeypatch, rewrite_to=["a", "b"])

    result = await refine_retrieval({"question": "q", "sub_queries": ["q"], "owner": "public"})

    assert [r.sub_query for r in result["vector_results"]] == ["a", "b"]
    assert [r.sub_query for r in result["bm25_results"]] == ["a", "b"]
