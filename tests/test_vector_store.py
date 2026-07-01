from rag_assistant.ingestion.build_index import build_index
from rag_assistant.retrieval.vector_store import get_retriever


def test_build_index_and_retrieve_relevant_chunk(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"

    num_chunks = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )
    assert num_chunks > 0

    retriever = get_retriever(k=1, embeddings=fake_embeddings, persist_dir=persist_dir)
    results = retriever.invoke("Who founded Anthropic and what model do they build?")

    assert len(results) == 1
    assert results[0].metadata["source"] == "anthropic.md"


def test_build_index_is_idempotent_on_rerun(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"

    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    num_chunks_second_run = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )

    retriever = get_retriever(k=10, embeddings=fake_embeddings, persist_dir=persist_dir)
    all_docs = retriever.invoke("Anthropic Mistral")

    assert len(all_docs) == num_chunks_second_run
