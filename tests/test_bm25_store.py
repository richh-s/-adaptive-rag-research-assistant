from pathlib import Path

import pytest

from rag_assistant.ingestion.build_index import build_index
from rag_assistant.retrieval.bm25_store import bm25_search, invalidate_bm25_index


@pytest.fixture
def small_corpus_dir(tmp_path: Path) -> Path:
    # rank_bm25's IDF formula degenerates to ~0 for terms that appear in exactly half of a
    # 2-document corpus (see BM25Okapi.idf), so this needs more than the 2-file
    # `sample_corpus_dir` fixture from conftest.py to produce meaningful, non-zero scores.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "anthropic.md").write_text(
        "Anthropic was founded by Dario Amodei and Daniela Amodei. It builds the Claude "
        "model family, focused on Constitutional AI and safety research."
    )
    (corpus_dir / "mistral.md").write_text(
        "Mistral AI is a French company founded in Paris that builds open-weight models "
        "like Mixtral, emphasizing European AI sovereignty."
    )
    (corpus_dir / "openai.md").write_text(
        "OpenAI was founded in San Francisco and builds the GPT model family, known for "
        "ChatGPT and large scale deployment."
    )
    (corpus_dir / "meta.md").write_text(
        "Meta AI is a research lab within Meta Platforms that builds the Llama model "
        "family and open source research tools."
    )
    return corpus_dir


@pytest.fixture
def indexed(small_corpus_dir, fake_embeddings, tmp_path):
    """BM25 reads the indexed chunks from Chroma rather than the corpus directory, so a
    corpus has to actually be ingested before keyword search can see it -- the same
    precondition vector search always had."""
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=small_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    invalidate_bm25_index(persist_dir)
    return small_corpus_dir, persist_dir


def test_bm25_search_returns_matching_source(indexed):
    corpus_dir, persist_dir = indexed
    results = bm25_search("Dario Amodei Constitutional AI", k=4, persist_dir=persist_dir)

    assert results
    assert results[0].source_id == "anthropic.md"


def test_bm25_search_ranks_distinct_sources_for_distinct_queries(indexed):
    corpus_dir, persist_dir = indexed
    anthropic_results = bm25_search("Amodei Claude", k=1, persist_dir=persist_dir)
    mistral_results = bm25_search("Mixtral Paris sovereignty", k=1, persist_dir=persist_dir)

    assert anthropic_results[0].source_id == "anthropic.md"
    assert mistral_results[0].source_id == "mistral.md"


def test_bm25_search_returns_empty_for_no_keyword_overlap(indexed):
    corpus_dir, persist_dir = indexed
    results = bm25_search("zzzznonexistentqueryterm", k=4, persist_dir=persist_dir)

    assert results == []


def test_bm25_search_populates_raw_score(indexed):
    corpus_dir, persist_dir = indexed
    results = bm25_search("Dario Amodei Constitutional AI", k=1, persist_dir=persist_dir)

    assert results[0].score is not None
    assert results[0].score > 0


def test_a_file_on_disk_is_invisible_until_it_is_indexed(indexed, fake_embeddings):
    """BM25 now reflects what has been ingested rather than what is on disk. That is the
    honest behaviour -- an un-ingested file was always invisible to vector search, so having
    keyword search answer from it made the two paths disagree about what the corpus is."""
    corpus_dir, persist_dir = indexed

    (corpus_dir / "cohere.md").write_text(
        "Cohere is an enterprise-focused AI company building large language models for business."
    )

    assert bm25_search("Cohere enterprise", k=4, persist_dir=persist_dir) == []


def test_indexing_makes_a_new_file_keyword_searchable(indexed, fake_embeddings):
    corpus_dir, persist_dir = indexed
    bm25_search("anything", k=4, persist_dir=persist_dir)  # warm the lazy singleton

    (corpus_dir / "cohere.md").write_text(
        "Cohere is an enterprise-focused AI company building large language models for business."
    )
    build_index(source_dir=corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    # build_index invalidates *and* eagerly rebuilds, so this must not need a manual
    # invalidation call -- a query landing right after ingestion should already see the file.
    results = bm25_search("Cohere enterprise", k=4, persist_dir=persist_dir)
    assert results[0].source_id == "cohere.md"


def test_invalidate_drops_the_cached_index(indexed, fake_embeddings):
    corpus_dir, persist_dir = indexed
    bm25_search("anything", k=4, persist_dir=persist_dir)

    invalidate_bm25_index(persist_dir)

    # Rebuilds from the collection rather than returning an emptied cache.
    assert bm25_search("Dario Amodei Constitutional AI", k=4, persist_dir=persist_dir)
