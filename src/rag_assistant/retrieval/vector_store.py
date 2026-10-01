import hashlib
import logging
import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.ingestion.acl import META_ACL, META_RESTRICTED
from rag_assistant.ingestion.generations import (
    active_index_dir,
    collection_suffix,
    generation_of_dir,
    index_root,
)
from rag_assistant.ingestion.ownership import owner_of_relative_path, visible_owners
from rag_assistant.llm import get_embeddings_model
from rag_assistant.retrieval.mmr import maximal_marginal_relevance

logger = logging.getLogger(__name__)

COLLECTION_NAME = "research_corpus"

# LangGraph's `Send` fan-out can invoke `retrieve_vector` for multiple sub-queries
# concurrently (via a thread pool). Two threads each opening a fresh `Chroma` client
# against the same on-disk directory races in its Rust binding teardown, so every
# persist directory gets exactly one cached client instance, built under a lock.
_store_cache: dict[str, object] = {}
_store_lock = threading.RLock()

_COSINE = {"hnsw:space": "cosine"}


def index_embeddings(persist_dir: Path) -> Embeddings:
    """The embedding model that must read and write the index at `persist_dir`: the one it
    recorded when it was built, or the configured one for an index that has recorded nothing
    yet (a fresh deployment, or an index older than the record).

    Following the index rather than the configuration is what makes a model change safe. The
    configured model decides what the *next* generation is built with (see
    ingestion/generations.py); an existing index keeps being queried in the space its vectors
    actually occupy, so changing EMBEDDING_PROVIDER can no longer make it return plausible
    nonsense.
    """
    from rag_assistant.ingestion.index_metadata import index_embedding_model

    return get_embeddings_model(index_embedding_model(persist_dir))


def _uses_server(persist_dir: Path) -> bool:
    """Whether this index lives on the configured Chroma server rather than on local disk.

    The configured index root and every generation under it are served from the server;
    any other directory is embedded. That second case is not hypothetical -- tests and the
    backup drill point the store at scratch directories while a server is configured, and
    those must never write into the shared server's collections.
    """
    settings = get_settings()
    if not settings.chroma_server_host:
        return False
    try:
        Path(persist_dir).resolve().relative_to(index_root().resolve())
    except ValueError:
        return False
    return True


def _chroma_client(persist_dir: Path):
    import chromadb

    settings = get_settings()
    if _uses_server(persist_dir):
        return chromadb.HttpClient(
            host=settings.chroma_server_host,
            port=settings.chroma_server_port,
            ssl=settings.chroma_server_ssl,
        )
    return chromadb.PersistentClient(path=str(persist_dir))


def get_vector_store(embeddings: Embeddings | None = None, persist_dir: Path | None = None):
    """The process-wide vector store for the configured backend and index generation.

    `persist_dir` selects the generation (see ingestion/generations.py) and defaults to the
    one currently serving. For pgvector it names a schema rather than a directory, and it
    still selects the rest of the index -- manifest, parent store and BM25 cache are all keyed
    on it -- so callers keep passing it either way.
    """
    settings = get_settings()
    persist_dir = Path(persist_dir) if persist_dir is not None else active_index_dir()
    generation = generation_of_dir(persist_dir)

    if settings.vector_backend == "pgvector":
        cache_key = f"pgvector:{generation}"
        if cache_key not in _store_cache:
            with _store_lock:
                if cache_key not in _store_cache:
                    from rag_assistant.retrieval.pgvector_store import PgVectorStore

                    _store_cache[cache_key] = PgVectorStore(
                        embeddings=embeddings or index_embeddings(persist_dir),
                        generation=generation,
                    )
        return _store_cache[cache_key]

    server = _uses_server(persist_dir)
    # Server mode is keyed separately so a process can hold both (tests pass explicit
    # persist dirs while the app may be pointed at a server).
    cache_key = (
        f"server:{settings.chroma_server_host}:{settings.chroma_server_port}:{generation}"
        if server
        else str(persist_dir)
    )
    if cache_key not in _store_cache:
        with _store_lock:
            if cache_key not in _store_cache:
                from rag_assistant.ingestion.index_metadata import index_layout

                # The layout the index was *written* with, not the configured one: an index
                # built as one shared collection is read as one shared collection until a new
                # generation replaces it (see generations.py), so changing TENANT_ISOLATION
                # never hides existing documents. Read once per construction; a full rebuild
                # that changes it evicts this store.
                layout = index_layout(persist_dir)
                function = embeddings or index_embeddings(persist_dir)
                base_name = COLLECTION_NAME + collection_suffix(generation)
                if layout == "strict":
                    _store_cache[cache_key] = TenantCollections(
                        _chroma_client(persist_dir), function, base_name
                    )
                else:
                    _store_cache[cache_key] = Chroma(
                        client=_chroma_client(persist_dir),
                        collection_name=base_name,
                        embedding_function=function,
                        # Chroma defaults to l2 (squared Euclidean) if unset; Gemini's
                        # embeddings are meant to be compared by cosine similarity, so leaving
                        # this unset silently ranks documents by the wrong metric.
                        collection_metadata=_COSINE,
                    )
    return _store_cache[cache_key]


def get_retriever(
    k: int = 4,
    embeddings: Embeddings | None = None,
    persist_dir: Path | None = None,
    owner: str = PUBLIC_OWNER,
    filters=None,
    principals: frozenset[str] | None = None,
) -> VectorStoreRetriever:
    """A retriever scoped to what `owner` is allowed to see: their own documents plus the
    shared public corpus -- and, within those, only documents whose ACL admits one of
    `principals` (None means the caller bypasses document ACLs; see ingestion/acl.py).

    The filter is applied by Chroma during search rather than by dropping results afterwards.
    That is not just efficiency -- post-filtering would silently shrink k, so a tenant whose
    top hits belong to someone else would get fewer documents (sometimes none) with no
    indication why, and the graph would read that as "the corpus has nothing" and fall back
    to web search.
    """
    settings = get_settings()
    store = get_vector_store(embeddings=embeddings, persist_dir=persist_dir)
    if settings.vector_backend == "pgvector":
        # The pgvector backend translates owner/filters into a SQL predicate itself, so it
        # takes them directly rather than a Chroma `where` dict it would have to parse back.
        return store.as_retriever(k=k, owner=owner, filters=filters, principals=principals)
    where = build_where_clause(owner, filters, principals)
    if isinstance(store, TenantCollections):
        return _ChromaMultiCollectionRetriever(
            [store.collection_for(o) for o in visible_owners(owner)],
            k=k,
            where=where,
            embeddings=embeddings or store.embeddings,
            mmr=settings.retrieval_mmr,
        )
    if settings.retrieval_mmr:
        return _ChromaMultiCollectionRetriever(
            [store], k=k, where=where, embeddings=embeddings or store.embeddings, mmr=True
        )
    return store.as_retriever(search_kwargs={"k": k, "filter": where})


class _ChromaMultiCollectionRetriever:
    """`.invoke(query) -> list[Document]` over one or more Chroma collections, with optional
    diversity selection. Mirrors `_PgVectorRetriever`'s shape.

    One collection is the MMR path over the shared collection. Several is strict tenant
    isolation, where a tenant's own collection and the public one are searched separately and
    merged by distance. Distances from different collections are comparable because every
    collection in a generation is embedded by the same model into the same cosine space --
    that is a property of the generation, not an assumption about the collections.

    Chroma ships its own MMR via `max_marginal_relevance_search`, and this deliberately does
    not use it. Both backends have to select the same documents from the same corpus -- there
    is a test asserting exactly that for the similarity path -- and two implementations with
    two tie-breaking rules would make that hold for plain retrieval and quietly stop holding
    the moment MMR was switched on. So both call `retrieval/mmr.py`, and the only thing that
    differs is how the candidate vectors are fetched.

    That fetch reaches through to `_collection` because LangChain's Chroma wrapper has no
    public call that returns embeddings alongside documents -- the same accommodation
    `readiness.py` and `index_metadata.py` already make for `count()` and `peek()`.
    """

    def __init__(
        self, stores: list, k: int, where: dict, embeddings: Embeddings, mmr: bool = False
    ):
        self._stores = stores
        self._k = k
        self._where = where
        self._embeddings = embeddings
        self._mmr = mmr

    def invoke(self, query: str) -> list[Document]:
        settings = get_settings()
        query_vector = self._embeddings.embed_query(query)
        # fetch_k is a ceiling on candidates, never on results: a corpus smaller than it
        # simply returns everything, and MMR selects from what it is given.
        n_results = max(settings.retrieval_fetch_k, self._k) if self._mmr else self._k
        include = ["documents", "metadatas", "distances"]
        if self._mmr:
            include.append("embeddings")

        candidates: list[tuple[float, str, dict, list[float] | None]] = []
        for store in self._stores:
            collection = store._collection
            if collection.count() == 0:
                continue
            result = collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=self._where,
                include=include,
            )
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            vectors = (result.get("embeddings") or [[]])[0] if self._mmr else []
            for index, content in enumerate(documents):
                candidates.append(
                    (
                        float(distances[index]) if index < len(distances) else 0.0,
                        content,
                        dict(metadatas[index] or {}) if index < len(metadatas) else {},
                        # `vectors` is a numpy array on current Chroma; `len()` works on both
                        # it and a list, but a bare truthiness check on an array raises.
                        list(vectors[index]) if index < len(vectors) else None,
                    )
                )
        candidates.sort(key=lambda c: c[0])
        candidates = candidates[:n_results]
        if not candidates:
            return []
        if not self._mmr or any(c[3] is None for c in candidates):
            return [Document(page_content=c[1], metadata=c[2]) for c in candidates[: self._k]]
        picked = maximal_marginal_relevance(
            query_vector,
            [c[3] for c in candidates],
            k=self._k,
            lambda_mult=settings.retrieval_mmr_lambda,
        )
        return [Document(page_content=candidates[i][1], metadata=candidates[i][2]) for i in picked]


def build_where_clause(owner: str, filters=None, principals: frozenset[str] | None = None) -> dict:
    """Chroma `where` combining tenant scope, document ACLs and the caller's metadata filters.

    Chroma requires `$and` for more than one condition, and rejects a single-clause `$and`,
    so the shape depends on how many conditions there actually are. The same goes for `$or`,
    which is why the ACL clause collapses to a single comparison when the caller holds no
    principals at all.

    The tenant clause stays under strict isolation too, where each collection already holds
    one tenant: it costs nothing, and it keeps the query correct on its own rather than
    correct only because of where it happened to be sent.
    """
    clauses: list[dict] = [{"owner": {"$in": visible_owners(owner)}}]
    if principals is not None:
        # `$ne: True` rather than `== False` so chunks indexed before ACLs existed -- which
        # have no `acl_restricted` key at all -- stay visible to their tenant.
        open_clause = {META_RESTRICTED: {"$ne": True}}
        member_clauses = [{META_ACL: {"$contains": p}} for p in sorted(principals)]
        clauses.append({"$or": [open_clause, *member_clauses]} if member_clauses else open_clause)
    if filters is not None and not filters.is_empty():
        if filters.sources:
            clauses.append({"source": {"$in": list(filters.sources)}})
        if filters.ingested_after is not None:
            clauses.append({"ingested_at": {"$gte": filters.ingested_after.timestamp()}})
        if filters.ingested_before is not None:
            clauses.append({"ingested_at": {"$lte": filters.ingested_before.timestamp()}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def update_chunk_metadata(store, ids: list[str], patch: dict) -> None:
    """Merges `patch` into the stored metadata of `ids` without re-embedding them. How a
    permission change reaches the index: the text and vectors did not change, so paying for
    an embedding call per chunk to rewrite two metadata fields would be pure waste."""
    if not ids:
        return
    if hasattr(store, "update_metadata"):
        store.update_metadata(ids, patch)
        return
    # Chroma merges the given keys into existing metadata rather than replacing it.
    store._collection.update(ids=list(ids), metadatas=[dict(patch) for _ in ids])


def add_embedded(
    store, documents: list[Document], ids: list[str], vectors: list[list[float]]
) -> None:
    """Inserts chunks whose vectors the caller already computed. A re-embed into a new index
    generation batches its own embedding calls, and going through `add_documents` would
    embed every chunk a second time."""
    if not documents:
        return
    if hasattr(store, "add_embedded"):
        store.add_embedded(documents, ids, vectors)
        return
    store._collection.upsert(
        ids=list(ids),
        documents=[d.page_content for d in documents],
        metadatas=[dict(d.metadata or {}) for d in documents],
        embeddings=[list(v) for v in vectors],
    )


def owner_of_chunk_id(chunk_id: str) -> str:
    """Chunk ids are `<source>::<n>` and a source's path names its owner (ownership.py), so a
    chunk can be routed to its tenant's collection from the id alone -- which is what lets a
    delete by id work without first reading the chunk back."""
    source = chunk_id.rsplit("::", 1)[0]
    return owner_of_relative_path(Path(source))


def tenant_collection_name(base_name: str, owner: str) -> str:
    """One Chroma collection per tenant. The public corpus keeps the base name, so a strict
    deployment's public collection is the same collection a shared one used.

    The hash suffix keeps names unique and valid: Chroma requires names to end in an
    alphanumeric, and owner labels may end in `-`.
    """
    if owner == PUBLIC_OWNER:
        return base_name
    digest = hashlib.sha256(owner.encode()).hexdigest()[:8]
    return f"{base_name}__t_{owner}_{digest}"


class _AggregateCollection:
    """The `_collection` calls made against a single Chroma collection elsewhere --
    readiness's `count()`, the dimension check's `peek()`, tests' `get()` -- answered across
    every tenant collection."""

    def __init__(self, owner: "TenantCollections"):
        self._owner = owner

    @property
    def metadata(self):
        return self._owner.collection_for(PUBLIC_OWNER)._collection.metadata

    def count(self) -> int:
        return sum(store._collection.count() for store in self._owner.all_collections())

    def peek(self, limit: int = 1) -> dict:
        for store in self._owner.all_collections():
            peeked = store._collection.peek(limit)
            embeddings = peeked.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                return peeked
        return {"embeddings": []}

    def get(self, **kwargs) -> dict:
        return self._owner.get(**kwargs)

    def update(self, ids: list[str], metadatas: list[dict]) -> None:
        for owner, pairs in _group(zip(ids, metadatas), key=lambda p: owner_of_chunk_id(p[0])):
            self._owner.collection_for(owner)._collection.update(
                ids=[p[0] for p in pairs], metadatas=[p[1] for p in pairs]
            )


def _group(items, key) -> list[tuple[str, list]]:
    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(key(item), []).append(item)
    return list(grouped.items())


class TenantCollections:
    """Strict tenant isolation on Chroma: one collection per tenant, behind the same
    interface a single collection presents.

    Writes are routed by the owner a chunk already carries, deletes by the owner its id
    encodes, and searches go only to the collections the caller may see -- their own and the
    public one. Another tenant's vectors are therefore not merely filtered out of the search;
    they are not in the index being searched at all, so a bug in the `where` clause cannot
    leak them.

    Everything that reads "the whole index" -- the BM25 build, readiness, the dimension check,
    a re-embed into a new generation -- goes through `get()` and `_collection`, which union
    every tenant's collection.
    """

    def __init__(self, client, embeddings: Embeddings, base_name: str):
        self._client = client
        self.embeddings = embeddings
        self._embedding_function = embeddings
        self._base_name = base_name
        self._collections: dict[str, Chroma] = {}
        self._lock = threading.Lock()
        self._collection = _AggregateCollection(self)

    def collection_for(self, owner: str) -> Chroma:
        if owner not in self._collections:
            with self._lock:
                if owner not in self._collections:
                    self._collections[owner] = Chroma(
                        client=self._client,
                        collection_name=tenant_collection_name(self._base_name, owner),
                        embedding_function=self.embeddings,
                        collection_metadata=_COSINE,
                    )
        return self._collections[owner]

    def _existing_collection_names(self) -> set[str]:
        try:
            listed = self._client.list_collections()
        except Exception:
            logger.debug("Could not list Chroma collections", exc_info=True)
            return set()
        return {getattr(c, "name", c) for c in listed}

    def all_collections(self) -> list[Chroma]:
        """Every tenant collection that exists, including ones this process has not touched
        yet -- a replica that did not perform an ingest still has to see it."""
        prefix = f"{self._base_name}__t_"
        names = self._existing_collection_names()
        stores = [self.collection_for(PUBLIC_OWNER)]
        for name in sorted(names):
            if not name.startswith(prefix):
                continue
            owner = name[len(prefix) :].rsplit("_", 1)[0]
            if tenant_collection_name(self._base_name, owner) == name:
                stores.append(self.collection_for(owner))
        return stores

    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        for owner, pairs in _group(
            zip(documents, ids), key=lambda p: (p[0].metadata or {}).get("owner", PUBLIC_OWNER)
        ):
            self.collection_for(owner).add_documents(
                [p[0] for p in pairs], ids=[p[1] for p in pairs]
            )
        return list(ids)

    def add_embedded(
        self, documents: list[Document], ids: list[str], vectors: list[list[float]]
    ) -> list[str]:
        for owner, rows in _group(
            zip(documents, ids, vectors),
            key=lambda r: (r[0].metadata or {}).get("owner", PUBLIC_OWNER),
        ):
            self.collection_for(owner)._collection.upsert(
                ids=[r[1] for r in rows],
                documents=[r[0].page_content for r in rows],
                metadatas=[dict(r[0].metadata or {}) for r in rows],
                embeddings=[list(r[2]) for r in rows],
            )
        return list(ids)

    def delete(self, ids: list[str]) -> None:
        for owner, owned in _group(ids, key=owner_of_chunk_id):
            self.collection_for(owner).delete(ids=owned)

    def update_metadata(self, ids: list[str], patch: dict) -> None:
        self._collection.update(ids=list(ids), metadatas=[dict(patch) for _ in ids])

    def get(self, ids: list[str] | None = None, include: list[str] | None = None, **kwargs):
        merged: dict[str, list] = {"ids": [], "documents": [], "metadatas": []}
        want_embeddings = bool(include and "embeddings" in include)
        if want_embeddings:
            merged["embeddings"] = []
        if ids is not None:
            targets = [
                (self.collection_for(owner), owned)
                for owner, owned in _group(ids, key=owner_of_chunk_id)
            ]
        else:
            targets = [(store, None) for store in self.all_collections()]
        for store, owned in targets:
            call = dict(kwargs)
            if include is not None:
                call["include"] = include
            if owned is not None:
                call["ids"] = owned
            got = store._collection.get(**call)
            merged["ids"].extend(got.get("ids") or [])
            merged["documents"].extend(got.get("documents") or [])
            merged["metadatas"].extend(got.get("metadatas") or [])
            if want_embeddings:
                embeddings = got.get("embeddings")
                merged["embeddings"].extend(
                    [list(v) for v in embeddings] if embeddings is not None else []
                )
        return merged

    def reset_collection(self) -> None:
        for store in self.all_collections():
            store.reset_collection()


def reset_store_cache() -> None:
    """Drops the cached store(s). Tests that switch backend or persist directory mid-process
    need this; so does the pgvector pool, which otherwise outlives a changed DATABASE_URL."""
    global _store_cache
    with _store_lock:
        _store_cache = {}
    try:
        from rag_assistant.retrieval.pgvector_store import reset_pool

        reset_pool()
    except Exception:
        logger.debug("pgvector pool reset skipped", exc_info=True)


def evict_store(persist_dir: Path) -> None:
    """Forgets the cached store for one index, so the next `get_vector_store` rebuilds it --
    with whatever embedding model the index now records."""
    generation = generation_of_dir(persist_dir)
    settings = get_settings()
    candidates = {
        f"pgvector:{generation}",
        f"server:{settings.chroma_server_host}:{settings.chroma_server_port}:{generation}",
        str(Path(persist_dir)),
    }
    with _store_lock:
        for key in candidates:
            _store_cache.pop(key, None)
