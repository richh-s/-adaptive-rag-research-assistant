from rag_assistant.ingestion.loaders import load_documents
from rag_assistant.ingestion.splitter import split_documents


def test_load_documents_reads_all_supported_files(sample_corpus_dir):
    docs = load_documents(sample_corpus_dir)

    assert len(docs) == 2
    sources = {d.metadata["source"] for d in docs}
    assert sources == {"anthropic.md", "mistral.md"}


def test_load_documents_ignores_unsupported_files(sample_corpus_dir):
    (sample_corpus_dir / "notes.json").write_text("{}")

    docs = load_documents(sample_corpus_dir)

    assert all(d.metadata["source"] != "notes.json" for d in docs)


def test_split_documents_produces_nonempty_chunks(sample_corpus_dir):
    docs = load_documents(sample_corpus_dir)

    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > len(docs)
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["source"] in {"anthropic.md", "mistral.md"} for chunk in chunks)
