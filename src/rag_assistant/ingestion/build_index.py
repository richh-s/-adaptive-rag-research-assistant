from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag_assistant.config import get_settings
from rag_assistant.ingestion.loaders import load_documents
from rag_assistant.ingestion.splitter import split_documents
from rag_assistant.retrieval.vector_store import get_vector_store


def build_index(
    source_dir: Path | None = None,
    persist_dir: Path | None = None,
    embeddings: Embeddings | None = None,
) -> int:
    """Load the corpus, chunk it, embed it, and (re)populate the Chroma collection.
    Resets the collection first so re-running ingestion doesn't duplicate chunks."""
    settings = get_settings()
    source_dir = source_dir or settings.corpus_dir
    persist_dir = persist_dir or settings.chroma_persist_dir

    documents = load_documents(source_dir)
    chunks = split_documents(documents)

    store = get_vector_store(embeddings=embeddings, persist_dir=persist_dir)
    store.reset_collection()
    if chunks:
        store.add_documents(chunks)
    return len(chunks)
