from rag_assistant.ingestion.build_index import build_index
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
