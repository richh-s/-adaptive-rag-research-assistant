from pathlib import Path

import pytest

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


def test_bm25_search_returns_matching_source(small_corpus_dir):
    results = bm25_search("Dario Amodei Constitutional AI", k=4, source_dir=small_corpus_dir)

    assert results
    assert results[0].source_id == "anthropic.md"


def test_bm25_search_ranks_distinct_sources_for_distinct_queries(small_corpus_dir):
    anthropic_results = bm25_search("Amodei Claude", k=1, source_dir=small_corpus_dir)
    mistral_results = bm25_search("Mixtral Paris sovereignty", k=1, source_dir=small_corpus_dir)

    assert anthropic_results[0].source_id == "anthropic.md"
    assert mistral_results[0].source_id == "mistral.md"


def test_bm25_search_returns_empty_for_no_keyword_overlap(small_corpus_dir):
    results = bm25_search("zzzznonexistentqueryterm", k=4, source_dir=small_corpus_dir)

    assert results == []


def test_bm25_search_populates_raw_score(small_corpus_dir):
    results = bm25_search("Dario Amodei Constitutional AI", k=1, source_dir=small_corpus_dir)

    assert results[0].score is not None
    assert results[0].score > 0


def test_invalidate_bm25_index_picks_up_newly_added_file(small_corpus_dir):
    # Warm the lazily-built singleton cache for this corpus dir before adding a new file.
    bm25_search("anything", k=4, source_dir=small_corpus_dir)

    (small_corpus_dir / "cohere.md").write_text(
        "Cohere is an enterprise-focused AI company building large language models for business."
    )

    # Without invalidation the stale cached index would keep answering from the pre-write
    # corpus for the rest of the process's lifetime.
    stale_results = bm25_search("Cohere enterprise", k=4, source_dir=small_corpus_dir)
    assert all(r.source_id != "cohere.md" for r in stale_results)

    invalidate_bm25_index(small_corpus_dir)

    fresh_results = bm25_search("Cohere enterprise", k=4, source_dir=small_corpus_dir)
    assert fresh_results[0].source_id == "cohere.md"
