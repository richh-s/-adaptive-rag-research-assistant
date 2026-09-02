"""Tests for metadata filtering of local retrieval.

Filters are pushed into the query rather than applied to the results, for the same reason
tenancy is: `k` is applied by the store, so post-filtering silently returns fewer documents
than asked for — and the graph reads a short result as "the corpus has nothing" and falls
back to web search. So these check the *where clause* as well as the end-to-end behaviour.
"""

from datetime import UTC, datetime, timedelta

import pytest

from rag_assistant.ingestion.build_index import build_index
from rag_assistant.retrieval.bm25_store import bm25_search, invalidate_bm25_index
from rag_assistant.retrieval.vector_store import build_where_clause, get_retriever
from rag_assistant.schemas.api import ResearchRequest, RetrievalFilters


@pytest.fixture
def indexed_corpus(tmp_path, fake_embeddings):
    # More than a couple of documents on purpose: rank_bm25's IDF degenerates towards zero
    # for a term appearing in ~half a tiny corpus, and bm25_search drops zero-scored hits --
    # so a two-file fixture would return nothing regardless of whether filtering works.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "anthropic.md").write_text(
        "# Anthropic\n\n## Focus\n\nConstitutional AI and alignment research programmes.\n"
    )
    (corpus / "mistral.md").write_text(
        "# Mistral\n\n## Focus\n\nOpen weight Mixtral models built in Paris for Europe.\n"
    )
    (corpus / "openai.md").write_text(
        "# OpenAI\n\n## Focus\n\nThe GPT family of models and the ChatGPT product line.\n"
    )
    (corpus / "meta.md").write_text(
        "# Meta\n\n## Focus\n\nThe Llama family of open models and research tooling.\n"
    )
    (corpus / "cohere.md").write_text(
        "# Cohere\n\n## Focus\n\nEnterprise language models and retrieval products.\n"
    )
    persist = tmp_path / "chroma"
    build_index(source_dir=corpus, persist_dir=persist, embeddings=fake_embeddings)
    invalidate_bm25_index(persist)
    return corpus, persist


# ---- schema ----


def test_filters_default_to_empty():
    assert ResearchRequest(question="a real question here").filters.is_empty()


def test_an_inverted_date_range_is_rejected():
    with pytest.raises(ValueError):
        RetrievalFilters(
            ingested_after=datetime(2026, 2, 1, tzinfo=UTC),
            ingested_before=datetime(2026, 1, 1, tzinfo=UTC),
        )


# ---- where clause ----


def test_tenancy_alone_produces_a_single_clause():
    """Chroma rejects a one-element `$and`, so the shape has to depend on the count."""
    assert build_where_clause("public", None) == {"owner": {"$in": ["public"]}}


def test_filters_are_combined_with_tenancy_under_and():
    clause = build_where_clause("alice", RetrievalFilters(sources=["a.md"]))

    assert clause == {
        "$and": [{"owner": {"$in": ["alice", "public"]}}, {"source": {"$in": ["a.md"]}}]
    }


def test_dates_become_numeric_comparisons():
    """Stored as epoch floats because Chroma's operators compare numbers -- lexicographic
    date strings would only sort correctly by accident of ISO formatting."""
    after = datetime(2026, 1, 1, tzinfo=UTC)
    clause = build_where_clause("public", RetrievalFilters(ingested_after=after))

    assert clause["$and"][1] == {"ingested_at": {"$gte": after.timestamp()}}


def test_an_empty_filter_object_adds_no_clause():
    assert build_where_clause("public", RetrievalFilters()) == {"owner": {"$in": ["public"]}}


# ---- vector retrieval ----


def test_source_filter_restricts_vector_results(indexed_corpus, fake_embeddings):
    _, persist = indexed_corpus

    docs = get_retriever(
        k=10,
        embeddings=fake_embeddings,
        persist_dir=persist,
        filters=RetrievalFilters(sources=["anthropic.md"]),
    ).invoke("alignment research")

    assert docs
    assert {d.metadata["source"] for d in docs} == {"anthropic.md"}


def test_a_future_ingested_after_excludes_everything(indexed_corpus, fake_embeddings):
    _, persist = indexed_corpus
    tomorrow = datetime.now(UTC) + timedelta(days=1)

    docs = get_retriever(
        k=10,
        embeddings=fake_embeddings,
        persist_dir=persist,
        filters=RetrievalFilters(ingested_after=tomorrow),
    ).invoke("alignment research")

    assert docs == []


def test_a_past_ingested_after_includes_everything(indexed_corpus, fake_embeddings):
    _, persist = indexed_corpus
    yesterday = datetime.now(UTC) - timedelta(days=1)

    docs = get_retriever(
        k=10,
        embeddings=fake_embeddings,
        persist_dir=persist,
        filters=RetrievalFilters(ingested_after=yesterday),
    ).invoke("alignment research")

    assert {"anthropic.md", "mistral.md"} <= {d.metadata["source"] for d in docs}


# ---- bm25 retrieval ----


def test_source_filter_restricts_bm25_results(indexed_corpus):
    _, persist = indexed_corpus

    hits = bm25_search(
        "Mixtral Paris Europe",
        k=10,
        persist_dir=persist,
        filters=RetrievalFilters(sources=["mistral.md"]),
    )

    assert hits
    assert {h.source_id for h in hits} == {"mistral.md"}


def test_bm25_date_filter_matches_the_vector_behaviour(indexed_corpus):
    _, persist = indexed_corpus
    tomorrow = datetime.now(UTC) + timedelta(days=1)

    hits = bm25_search(
        "Mixtral Paris Europe",
        k=10,
        persist_dir=persist,
        filters=RetrievalFilters(ingested_after=tomorrow),
    )

    assert hits == []


def test_a_chunk_without_an_ingested_at_fails_a_date_filter(indexed_corpus):
    """A date filter is a claim about when a document was indexed; "unknown" can't satisfy
    it. Chunks predating the field are excluded until a re-index populates them."""
    from rag_assistant.retrieval.bm25_store import _passes_filters

    filters = RetrievalFilters(ingested_after=datetime(2020, 1, 1, tzinfo=UTC))

    assert _passes_filters({"source": "a.md"}, filters) is False
    assert _passes_filters({"source": "a.md", "ingested_at": 4.0e9}, filters) is True


def test_filters_do_not_shrink_k_below_what_is_available(indexed_corpus):
    """The post-filtering failure this design avoids: asking for one document from a filtered
    set must return one, not zero because the unfiltered top hit belonged to another source."""
    _, persist = indexed_corpus

    hits = bm25_search(
        "Mixtral Paris Europe",
        k=1,
        persist_dir=persist,
        filters=RetrievalFilters(sources=["mistral.md"]),
    )

    assert len(hits) == 1
    assert hits[0].source_id == "mistral.md"


# ---- end to end ----


def test_ingested_at_is_recorded_on_every_chunk(indexed_corpus, fake_embeddings):
    _, persist = indexed_corpus

    docs = get_retriever(k=10, embeddings=fake_embeddings, persist_dir=persist).invoke("focus")

    assert all(isinstance(d.metadata.get("ingested_at"), float) for d in docs)


def test_filters_reach_the_retrieval_nodes(monkeypatch):
    """The filter is only real if the API actually threads it down to the retriever."""
    from rag_assistant.graph.nodes.decompose import dispatch_retrieval

    filters = RetrievalFilters(sources=["anthropic.md"])
    sends = dispatch_retrieval(
        {"route": "vector", "sub_queries": ["q"], "owner": "public", "filters": filters}
    )

    assert all(send.arg["filters"] is filters for send in sends)
