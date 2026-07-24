from unittest.mock import patch

from test_ingestion import _make_minimal_pdf

from rag_assistant.ingestion.build_index import build_index
from rag_assistant.retrieval.bm25_store import bm25_search, get_bm25_index
from rag_assistant.retrieval.vector_store import get_retriever


def test_unchanged_file_is_skipped_on_rerun(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"

    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    result = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )

    assert result.changed_files == 0
    assert result.skipped_files == 2
    assert result.indexed_chunks == 0


def test_editing_a_file_reindexes_only_that_file(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"

    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    (sample_corpus_dir / "anthropic.md").write_text(
        "Anthropic was founded by Dario Amodei and Daniela Amodei, and now also builds "
        "Claude Code, an agentic coding assistant, in addition to the core Claude model family."
    )

    result = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )

    assert result.changed_files == 1
    assert result.skipped_files == 1
    assert result.removed_files == 0

    retriever = get_retriever(k=10, embeddings=fake_embeddings, persist_dir=persist_dir)
    docs = retriever.invoke("Claude Code agentic coding assistant")

    assert any("Claude Code" in doc.page_content for doc in docs)


def test_deleting_a_file_removes_its_chunks(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"

    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    (sample_corpus_dir / "mistral.md").unlink()

    result = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )

    assert result.removed_files == 1
    assert result.changed_files == 0
    assert result.skipped_files == 1

    retriever = get_retriever(k=10, embeddings=fake_embeddings, persist_dir=persist_dir)
    docs = retriever.invoke("Mistral Mixtral France")

    assert all(doc.metadata["source"] != "mistral.md" for doc in docs)


def test_multi_page_pdf_indexes_every_page(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"
    (sample_corpus_dir / "cohere.pdf").write_bytes(
        _make_minimal_pdf(["Cohere page one about embeddings.", "Cohere page two about rerankers."])
    )

    result = build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    # 2 md files (1 chunk each) + 2 PDF pages (1 chunk each) -- proves both pages of the
    # single-source PDF survived grouping, not just the last one written into a dict.
    assert result.changed_files == 3
    assert result.indexed_chunks == 4

    retriever = get_retriever(k=10, embeddings=fake_embeddings, persist_dir=persist_dir)
    docs = retriever.invoke("Cohere rerankers")
    pdf_docs = [d for d in docs if d.metadata["source"] == "cohere.pdf"]
    assert {d.metadata["page"] for d in pdf_docs} == {1, 2}

    # BM25's lazily-built singleton cache must also have been invalidated by build_index,
    # not just Chroma -- otherwise local-corpus queries silently miss newly ingested PDFs.
    bm25_results = bm25_search("Cohere rerankers", k=10, source_dir=sample_corpus_dir)
    assert any(r.source_id == "cohere.pdf" for r in bm25_results)


def test_full_flag_forces_reset_and_full_reembed(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"

    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    result = build_index(
        source_dir=sample_corpus_dir,
        persist_dir=persist_dir,
        embeddings=fake_embeddings,
        incremental=False,
    )

    assert result.changed_files == 2
    assert result.skipped_files == 0


def test_on_stage_hook_fires_for_parsing_and_indexing(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"
    stages = []

    build_index(
        source_dir=sample_corpus_dir,
        persist_dir=persist_dir,
        embeddings=fake_embeddings,
        on_stage=lambda stage, message: stages.append(stage),
    )

    assert stages[0] == "parsing"
    assert "indexing" in stages
    assert stages[-1] == "indexing"  # the post-index BM25 hot-reload also reports "indexing"


def test_on_stage_hook_not_called_again_for_a_no_op_rerun(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    stages = []
    result = build_index(
        source_dir=sample_corpus_dir,
        persist_dir=persist_dir,
        embeddings=fake_embeddings,
        on_stage=lambda stage, message: stages.append(stage),
    )

    assert result.changed_files == 0
    # "parsing" always fires (loading the corpus is unconditional), but the hot-reload
    # "indexing" call after the loop is gated on changed_files/removed_sources -- a no-op
    # rerun shouldn't pay for an unnecessary BM25 rebuild.
    assert stages == ["parsing", "indexing"]


def test_bm25_index_is_eagerly_rebuilt_not_left_stale_after_build_index(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    # Warm the cache once so we can prove the *next* build_index() call refreshes it inline.
    get_bm25_index(sample_corpus_dir)

    (sample_corpus_dir / "anthropic.md").write_text("Anthropic ships Claude Opus 5 today.")

    with patch(
        "rag_assistant.ingestion.build_index.get_bm25_index", wraps=get_bm25_index
    ) as spy:
        build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
        # build_index() itself must trigger the rebuild -- not merely invalidate and leave it
        # for whichever request happens to call get_bm25_index/bm25_search next.
        spy.assert_called_once_with(sample_corpus_dir)

    # Assert against the rebuilt chunks directly rather than through bm25_search's score-filtered
    # results: with only 2 documents in this fixture, BM25Okapi's idf for a term appearing in
    # exactly 1 of 2 docs is log(1.5) - log(1.5) == 0, so even a perfectly-rebuilt index would
    # legitimately score every result 0 and get filtered out by bm25_search's `score > 0` guard.
    # That's a corpus-size artifact of BM25 scoring, not a signal about whether the rebuild
    # happened -- which is what this test is actually checking.
    _, chunks = get_bm25_index(sample_corpus_dir)
    assert any("Opus 5" in chunk.page_content for chunk in chunks)
