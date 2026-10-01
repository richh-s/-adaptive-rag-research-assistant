from rag_assistant.ingestion.build_index import build_index
from rag_assistant.retrieval.vector_store import get_retriever


def test_build_index_and_retrieve_relevant_chunk(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"

    result = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )
    assert result.indexed_chunks > 0
    assert result.changed_files == 2
    assert result.skipped_files == 0
    assert result.removed_files == 0

    retriever = get_retriever(k=1, embeddings=fake_embeddings, persist_dir=persist_dir)
    results = retriever.invoke("Who founded Anthropic and what model do they build?")

    assert len(results) == 1
    assert results[0].metadata["source"] == "anthropic.md"


def test_build_index_is_idempotent_on_rerun(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"

    first_run = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )
    second_run = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )

    assert second_run.changed_files == 0
    assert second_run.skipped_files == 2

    retriever = get_retriever(k=10, embeddings=fake_embeddings, persist_dir=persist_dir)
    all_docs = retriever.invoke("Anthropic Mistral")

    assert len(all_docs) == first_run.indexed_chunks


def _index_docs(persist_dir, fake_embeddings, docs: dict[str, str]):
    """Indexes literal chunk texts, bypassing the splitter so a test controls exactly what
    the retriever has to choose between."""
    from langchain_core.documents import Document

    from rag_assistant.retrieval.vector_store import get_vector_store, reset_store_cache

    reset_store_cache()
    store = get_vector_store(embeddings=fake_embeddings, persist_dir=persist_dir)
    store.add_documents(
        [
            Document(page_content=text, metadata={"source": source, "owner": "public"})
            for source, text in docs.items()
        ],
        ids=list(docs),
    )
    return store


def test_mmr_returns_a_different_document_than_the_near_restatement(
    fake_embeddings, tmp_path, monkeypatch
):
    """With diversity off, the top 2 are a passage and its restatement; with it on, the
    second slot goes to the document covering different ground.

    The two near-identical documents are not byte-identical, so fusion's near-duplicate
    collapsing would not merge them either -- this is the redundancy that survives everything
    else in the pipeline.
    """
    from rag_assistant.config import get_settings
    from rag_assistant.retrieval.vector_store import reset_store_cache

    persist_dir = tmp_path / "chroma"
    _index_docs(
        persist_dir,
        fake_embeddings,
        {
            "safety_a.md": "Anthropic safety research safety research alignment",
            "safety_b.md": "Anthropic safety research safety research alignment work",
            "funding.md": "Anthropic safety funding Series C investors capital",
        },
    )

    get_settings.cache_clear()
    plain = get_retriever(k=2, embeddings=fake_embeddings, persist_dir=persist_dir)
    plain_sources = {d.metadata["source"] for d in plain.invoke("Anthropic safety research")}

    monkeypatch.setenv("RETRIEVAL_MMR", "true")
    monkeypatch.setenv("RETRIEVAL_MMR_LAMBDA", "0.5")
    get_settings.cache_clear()
    reset_store_cache()
    diverse = get_retriever(k=2, embeddings=fake_embeddings, persist_dir=persist_dir)
    diverse_sources = {d.metadata["source"] for d in diverse.invoke("Anthropic safety research")}

    assert plain_sources == {"safety_a.md", "safety_b.md"}
    assert diverse_sources == {"safety_a.md", "funding.md"}


def test_mmr_still_scopes_retrieval_to_the_tenant(fake_embeddings, tmp_path, monkeypatch):
    """The diversity path builds its own query against the collection, so it has to carry the
    tenant predicate itself -- a retriever that diversified across every tenant's documents
    would be a data-isolation bug introduced by a quality setting."""
    from langchain_core.documents import Document

    from rag_assistant.config import get_settings
    from rag_assistant.retrieval.vector_store import get_vector_store, reset_store_cache

    persist_dir = tmp_path / "chroma"
    reset_store_cache()
    store = get_vector_store(embeddings=fake_embeddings, persist_dir=persist_dir)
    store.add_documents(
        [
            Document(
                page_content="shared corpus text", metadata={"source": "p.md", "owner": "public"}
            ),
            Document(
                page_content="alice private text", metadata={"source": "a.md", "owner": "alice"}
            ),
        ],
        ids=["p", "a"],
    )

    monkeypatch.setenv("RETRIEVAL_MMR", "true")
    get_settings.cache_clear()
    reset_store_cache()
    docs = get_retriever(
        k=10, embeddings=fake_embeddings, persist_dir=persist_dir, owner="bob"
    ).invoke("text")

    assert {d.metadata["source"] for d in docs} == {"p.md"}
