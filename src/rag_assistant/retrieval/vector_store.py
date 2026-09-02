import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.ingestion.ownership import visible_owners
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
    k: int = 4,
    embeddings: Embeddings | None = None,
    persist_dir: Path | None = None,
    owner: str = PUBLIC_OWNER,
) -> VectorStoreRetriever:
    """A retriever scoped to what `owner` is allowed to see: their own documents plus the
    shared public corpus.

    The filter is applied by Chroma during search rather than by dropping results afterwards.
    That is not just efficiency -- post-filtering would silently shrink k, so a tenant whose
    top hits belong to someone else would get fewer documents (sometimes none) with no
    indication why, and the graph would read that as "the corpus has nothing" and fall back
    to web search.
    """
    store = get_vector_store(embeddings=embeddings, persist_dir=persist_dir)
    return store.as_retriever(
        search_kwargs={"k": k, "filter": {"owner": {"$in": visible_owners(owner)}}}
    )
