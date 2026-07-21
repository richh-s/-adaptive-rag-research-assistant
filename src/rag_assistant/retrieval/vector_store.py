import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from rag_assistant.config import get_settings
from rag_assistant.llm import get_embeddings_model

COLLECTION_NAME = "research_corpus"

# LangGraph's `Send` fan-out can invoke `retrieve_vector` for multiple sub-queries
# concurrently (via a thread pool). Two threads each opening a fresh `Chroma` client
# against the same on-disk directory races in its Rust binding teardown, so every
# persist directory gets exactly one cached client instance, built under a lock.
_store_cache: dict[str, Chroma] = {}
_store_lock = threading.Lock()


def get_vector_store(
    embeddings: Embeddings | None = None, persist_dir: Path | None = None
) -> Chroma:
    settings = get_settings()
    resolved_persist_dir = str(persist_dir or settings.chroma_persist_dir)
    if resolved_persist_dir not in _store_cache:
        with _store_lock:
            if resolved_persist_dir not in _store_cache:
                _store_cache[resolved_persist_dir] = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings or get_embeddings_model(),
                    persist_directory=resolved_persist_dir,
                    # Chroma defaults to l2 (squared Euclidean) if unset; Gemini's embeddings
                    # are meant to be compared by cosine similarity, so leaving this unset
                    # silently ranks documents by the wrong metric.
                    collection_metadata={"hnsw:space": "cosine"},
                )
    return _store_cache[resolved_persist_dir]


def get_retriever(
    k: int = 4, embeddings: Embeddings | None = None, persist_dir: Path | None = None
) -> VectorStoreRetriever:
    store = get_vector_store(embeddings=embeddings, persist_dir=persist_dir)
    return store.as_retriever(search_kwargs={"k": k})
