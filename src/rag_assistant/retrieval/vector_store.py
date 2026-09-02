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
    # Server mode is keyed separately so a process can hold both (tests pass explicit
    # persist dirs while the app may be pointed at a server).
    cache_key = (
        f"server:{settings.chroma_server_host}:{settings.chroma_server_port}"
        if settings.chroma_server_host and persist_dir is None
        else resolved_persist_dir
    )
    if cache_key not in _store_cache:
        with _store_lock:
            if cache_key not in _store_cache:
                common = {
                    "collection_name": COLLECTION_NAME,
                    "embedding_function": embeddings or get_embeddings_model(),
                    # Chroma defaults to l2 (squared Euclidean) if unset; Gemini's embeddings
                    # are meant to be compared by cosine similarity, so leaving this unset
                    # silently ranks documents by the wrong metric.
                    "collection_metadata": {"hnsw:space": "cosine"},
                }
                if cache_key.startswith("server:"):
                    import chromadb

                    _store_cache[cache_key] = Chroma(
                        client=chromadb.HttpClient(
                            host=settings.chroma_server_host,
                            port=settings.chroma_server_port,
                            ssl=settings.chroma_server_ssl,
                        ),
                        **common,
                    )
                else:
                    _store_cache[cache_key] = Chroma(
                        persist_directory=resolved_persist_dir, **common
                    )
    return _store_cache[cache_key]


def get_retriever(
    k: int = 4,
    embeddings: Embeddings | None = None,
    persist_dir: Path | None = None,
    owner: str = PUBLIC_OWNER,
    filters=None,
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
    return store.as_retriever(search_kwargs={"k": k, "filter": build_where_clause(owner, filters)})


def build_where_clause(owner: str, filters=None) -> dict:
    """Chroma `where` combining tenant scope with the caller's metadata filters.

    Chroma requires `$and` for more than one condition, and rejects a single-clause `$and`,
    so the shape depends on how many conditions there actually are.
    """
    clauses: list[dict] = [{"owner": {"$in": visible_owners(owner)}}]
    if filters is not None and not filters.is_empty():
        if filters.sources:
            clauses.append({"source": {"$in": list(filters.sources)}})
        if filters.ingested_after is not None:
            clauses.append({"ingested_at": {"$gte": filters.ingested_after.timestamp()}})
        if filters.ingested_before is not None:
            clauses.append({"ingested_at": {"$lte": filters.ingested_before.timestamp()}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
