"""pgvector-backed vector store, interchangeable with the embedded Chroma default.

`VECTOR_BACKEND=pgvector` plus `DATABASE_URL` switches to it; nothing that calls
`retrieval.vector_store` knows which is running. This is the same argument the Postgres
conversations backend makes, applied to the index: embedded Chroma is SQLite-backed and locks
its file to one process, which is the single hardest constraint on running more than one
worker. Chroma server mode removes that too, but it means operating another service; if
Postgres is already in the deployment for conversations, pgvector makes the index a table in
a database that is already backed up, replicated and monitored.

Two things are deliberately different from the Chroma path:

* **The column's dimension configures itself on first write.** The table is created with an
  unconstrained `vector`, and the first `add_documents` ALTERs it to the width it actually
  observes and builds the HNSW index. There is no dimension setting to get wrong, and once
  set, pgvector itself rejects a vector of the wrong width. That turns the embedding-model
  swap -- the one dependency whose failure is otherwise silent, because a same-dimension
  model returns plausible nonsense rather than an error -- into a loud failure at insert.

* **Filtering is a SQL predicate, not a post-filter.** Same reasoning as the Chroma path:
  post-filtering silently shrinks k, so a tenant whose top hits belong to someone else gets
  fewer documents with no indication why, and the graph reads that as "the corpus has
  nothing" and falls back to web search.

Cosine distance (`<=>`) throughout, matching the Chroma collection's `hnsw:space: cosine` --
Gemini's embeddings are meant to be compared that way, and a store that ranked by L2 while
the other ranked by cosine would not be the interchangeable backend this claims to be.
"""

import logging
import threading
import time

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.ingestion.ownership import visible_owners

logger = logging.getLogger(__name__)

TABLE_NAME = "corpus_chunks"

# Distinct from the conversations backend's lock id: the two migration chains are
# independent and may run against the same database at the same time, so sharing a lock id
# would serialise unrelated startups and, worse, make a failure in one look like a hang in
# the other.
_MIGRATION_LOCK_ID = hash("rag_assistant_pgvector_migrations") % 2**31

_LOCK = threading.Lock()
_pool = None


def _migration_001_baseline(cur) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # `vector` without a width on purpose -- see the module docstring. _ensure_dimension()
    # narrows it on first write.
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            chunk_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector NOT NULL,
            owner TEXT NOT NULL DEFAULT 'public',
            source TEXT NOT NULL DEFAULT '',
            ingested_at DOUBLE PRECISION,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
        )
        """
    )
    # Tenant scope is on every single query, so it leads the composite. `source` and
    # `ingested_at` follow because they are the two optional filters the API exposes.
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_owner_source ON {TABLE_NAME}(owner, source)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_ingested_at ON {TABLE_NAME}(ingested_at)"
    )


def _migration_002_index_state(cur) -> None:
    """The rest of the index's state, so the whole index moves together.

    Vectors alone are not the index. `ingestion/manifest.py` records what is currently
    indexed and is what makes re-ingest incremental; `retrieval/parent_store.py` holds the
    section bodies small-to-big retrieval hands to synthesis. Leaving those two on local disk
    while the vectors move to Postgres produces the worst of both: replica B decides what to
    re-index from a manifest that never saw replica A's ingest, and serves chunks whose parent
    sections it does not have. A manifest that disagrees with the vectors it describes is
    worse than either being missing.
    """
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS corpus_manifest (
            source TEXT PRIMARY KEY,
            owner TEXT NOT NULL DEFAULT 'public',
            -- The entry is stored whole rather than as columns because its shape is
            -- versioned by the code that writes it: an entry written by an older build
            -- simply lacks the newer fields, compares unequal, and re-indexes. Columns would
            -- turn every added field into a migration for no gain.
            entry JSONB NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS corpus_parents (
            parent_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            owner TEXT NOT NULL DEFAULT 'public',
            content TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_corpus_parents_source ON corpus_parents(source)")
    # A single-row counter bumped on every ingest that changed anything. This is what lets a
    # replica that did not perform an ingest discover that its in-memory BM25 index is stale
    # -- see bm25_store.py. One row, enforced by the CHECK, so "the version" is unambiguous.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS corpus_index_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version BIGINT NOT NULL DEFAULT 0
        )
        """
    )
    cur.execute("INSERT INTO corpus_index_state (id, version) VALUES (1, 0) ON CONFLICT DO NOTHING")


def _migration_003_index_metadata(cur) -> None:
    """The embedding model the index was built with.

    Left on local disk when the manifest and parents moved, which quietly disabled the guard
    it exists to be. `check_embedding_model` treats a missing record as "cannot verify" and
    reports ready -- correct for a fresh deployment, wrong for a replica that simply did not
    perform the ingest. Every such replica therefore skipped the check entirely.

    That matters most for the failure this record is the *only* defence against: a different
    embedding model of the *same* width. A different width is now rejected by pgvector at
    insert; a same-width model produces plausible neighbours from a space the stored vectors
    do not occupy, with no error anywhere.
    """
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS corpus_index_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            embedding_model TEXT NOT NULL,
            embedding_dimension INTEGER,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )


_MIGRATIONS: list = [
    _migration_001_baseline,
    _migration_002_index_state,
    _migration_003_index_metadata,
]


def _get_pool():
    """One connection pool per process, mirroring the conversations backend.

    A pool rather than a shared connection because LangGraph's `Send` fan-out runs
    `retrieve_vector` for several sub-queries concurrently on a thread pool, and Postgres
    connections are not safe to share across threads.
    """
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool

        settings = get_settings()
        if not settings.database_url:
            raise RuntimeError("VECTOR_BACKEND=pgvector requires DATABASE_URL to be set.")
        _pool = ConnectionPool(settings.database_url, min_size=1, open=True)
        _migrate()
    return _pool


def reset_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
    _pool = None


def _migrate() -> None:
    """Applies pending migrations, each in its own transaction, under an advisory lock so
    several replicas can start at once without racing each other onto the same migration."""
    with _pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
            try:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS pgvector_schema_migrations "
                    "(version INTEGER PRIMARY KEY, applied_at DOUBLE PRECISION NOT NULL)"
                )
                conn.commit()
                cur.execute("SELECT COALESCE(MAX(version), 0) FROM pgvector_schema_migrations")
                current = cur.fetchone()[0]
                for version in range(current, len(_MIGRATIONS)):
                    migration = _MIGRATIONS[version]
                    logger.info(
                        "applying pgvector migration %d (%s)", version + 1, migration.__name__
                    )
                    migration(cur)
                    cur.execute(
                        "INSERT INTO pgvector_schema_migrations (version, applied_at) "
                        "VALUES (%s, %s)",
                        (version + 1, time.time()),
                    )
                    conn.commit()
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))
                conn.commit()


def _to_vector_literal(values) -> str:
    """pgvector's text input format. Sent as a string and cast with `::vector` rather than
    via a registered type adapter, so the backend needs no import beyond psycopg, which is
    already a dependency for the conversations store."""
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def _column_dimension(cur) -> int | None:
    """The width the embedding column is currently constrained to, or None while it is still
    unconstrained. pgvector stores it in `atttypmod`, which is -1 for a bare `vector`."""
    cur.execute(
        "SELECT a.atttypmod FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid "
        "WHERE c.relname = %s AND a.attname = 'embedding'",
        (TABLE_NAME,),
    )
    row = cur.fetchone()
    if row is None or row[0] is None or row[0] < 0:
        return None
    return row[0]


def _ensure_dimension(conn, cur, dimension: int) -> None:
    """Narrows the embedding column to `dimension` and builds the HNSW index, once.

    Deferred to the first write because that is the first moment the width is actually known
    -- it is a property of the configured embedding model, not something worth asking an
    operator to restate in an environment variable where it can disagree with reality. HNSW
    cannot be built on an unconstrained `vector`, so the index arrives with the constraint.
    """
    if _column_dimension(cur) is not None:
        return
    logger.info("pgvector: fixing embedding dimension at %d and building HNSW index", dimension)
    cur.execute(f"ALTER TABLE {TABLE_NAME} ALTER COLUMN embedding TYPE vector({dimension})")
    # vector_cosine_ops to match the Chroma collection's cosine space. An index built for a
    # different operator class is simply not used by a `<=>` query -- it would degrade to a
    # sequential scan silently rather than fail, which is why the operator class is pinned
    # here next to the query that depends on it.
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_embedding_hnsw "
        f"ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops)"
    )
    conn.commit()


class _Collection:
    """The two private-`_collection` calls the rest of the codebase makes against Chroma
    (`readiness.count()` and `index_metadata.peek(1)`), served from Postgres so those modules
    work against either backend without a branch."""

    def __init__(self, store: "PgVectorStore"):
        self._store = store

    def count(self) -> int:
        with _get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            return cur.fetchone()[0]

    def peek(self, limit: int = 1) -> dict:
        with _get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT embedding FROM {TABLE_NAME} LIMIT %s", (limit,))
            rows = cur.fetchall()
        # Returned in Chroma's peek shape so read_embedding_dimension() -- which only ever
        # takes len(embeddings[0]) -- needs no knowledge of which backend answered.
        return {"embeddings": [_parse_vector(row[0]) for row in rows]}


def _parse_vector(raw) -> list[float]:
    if isinstance(raw, str):
        return [float(v) for v in raw.strip("[]").split(",") if v]
    return list(raw)


class PgVectorStore:
    """The subset of Chroma's interface this codebase actually uses, over pgvector.

    Deliberately not a LangChain `VectorStore` subclass: implementing the full abstract
    surface would mean writing several methods nothing here calls, and each one would be
    untested code that looks supported. The methods below are exactly the ones
    `build_index`, `bm25_store`, `readiness` and `index_metadata` reach for.
    """

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self._collection = _Collection(self)

    # ---- writes ----

    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        if not documents:
            return []
        vectors = self.embeddings.embed_documents([d.page_content for d in documents])
        rows = []
        for chunk_id, doc, vector in zip(ids, documents, vectors):
            metadata = dict(doc.metadata or {})
            rows.append(
                (
                    chunk_id,
                    doc.page_content,
                    _to_vector_literal(vector),
                    metadata.get("owner", PUBLIC_OWNER),
                    metadata.get("source", ""),
                    metadata.get("ingested_at"),
                    _json_dumps(metadata),
                )
            )
        pool = _get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            _ensure_dimension(conn, cur, len(vectors[0]))
            # ON CONFLICT rather than delete-then-insert: build_index already deletes a
            # source's previous chunk ids before re-adding, but chunk ids are derived from
            # (source, index) and so collide by design when a file is re-chunked into the
            # same count. An upsert makes a re-ingest idempotent either way.
            cur.executemany(
                f"""
                INSERT INTO {TABLE_NAME}
                    (chunk_id, content, embedding, owner, source, ingested_at, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    owner = EXCLUDED.owner,
                    source = EXCLUDED.source,
                    ingested_at = EXCLUDED.ingested_at,
                    metadata = EXCLUDED.metadata
                """,
                rows,
            )
            conn.commit()
        return list(ids)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        with _get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE_NAME} WHERE chunk_id = ANY(%s)", (list(ids),))
            conn.commit()

    def reset_collection(self) -> None:
        with _get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(f"TRUNCATE {TABLE_NAME}")
            conn.commit()

    # ---- reads ----

    def get(self, ids: list[str] | None = None, include: list[str] | None = None) -> dict:
        """Chroma's `get` shape: parallel lists under `ids`/`documents`/`metadatas`.

        `include` is accepted and ignored -- the three columns cost the same one row fetch
        here, and honouring it would only let a caller receive fewer fields than it asked
        for on one backend and not the other.
        """
        sql = f"SELECT chunk_id, content, metadata FROM {TABLE_NAME}"
        params: tuple = ()
        if ids is not None:
            sql += " WHERE chunk_id = ANY(%s)"
            params = (list(ids),)
        with _get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [dict(r[2] or {}) for r in rows],
        }

    def similarity_search(
        self, query: str, k: int = 4, owner: str = PUBLIC_OWNER, filters=None
    ) -> list[Document]:
        vector = _to_vector_literal(self.embeddings.embed_query(query))
        where, params = build_where_sql(owner, filters)
        with _get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT content, metadata FROM {TABLE_NAME} "
                f"WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s",
                (*params, vector, k),
            )
            rows = cur.fetchall()
        return [Document(page_content=r[0], metadata=dict(r[1] or {})) for r in rows]

    def as_retriever(self, k: int = 4, owner: str = PUBLIC_OWNER, filters=None):
        return _PgVectorRetriever(self, k=k, owner=owner, filters=filters)


class _PgVectorRetriever:
    """`.invoke(query) -> list[Document]`, the only part of LangChain's retriever protocol
    the graph uses (see graph/nodes/retrieve.py)."""

    def __init__(self, store: PgVectorStore, k: int, owner: str, filters):
        self._store = store
        self._k = k
        self._owner = owner
        self._filters = filters

    def invoke(self, query: str) -> list[Document]:
        return self._store.similarity_search(
            query, k=self._k, owner=self._owner, filters=self._filters
        )


def build_where_sql(owner: str, filters=None) -> tuple[str, tuple]:
    """The SQL counterpart of vector_store.build_where_clause -- tenant scope plus the
    caller's metadata filters, as a predicate applied during the search.

    Returns (sql, params) rather than an interpolated string so every value reaches Postgres
    as a bound parameter; `source` in particular is caller-supplied.
    """
    clauses = ["owner = ANY(%s)"]
    params: list = [visible_owners(owner)]
    if filters is not None and not filters.is_empty():
        if filters.sources:
            clauses.append("source = ANY(%s)")
            params.append(list(filters.sources))
        if filters.ingested_after is not None:
            clauses.append("ingested_at >= %s")
            params.append(filters.ingested_after.timestamp())
        if filters.ingested_before is not None:
            clauses.append("ingested_at <= %s")
            params.append(filters.ingested_before.timestamp())
    return " AND ".join(clauses), tuple(params)


def _json_dumps(value) -> str:
    import json

    return json.dumps(value, default=str)


# ---- the rest of the index's state (see _migration_002_index_state) ----


def load_manifest_rows() -> dict[str, dict]:
    with _get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT source, entry FROM corpus_manifest")
        return {row[0]: dict(row[1]) for row in cur.fetchall()}


def save_manifest_rows(manifest: dict[str, dict]) -> None:
    """Replaces the manifest with `manifest`, as one transaction.

    The file-backed manifest is rewritten whole on every save, and callers rely on that: a
    source removed from the dict is a source that is no longer indexed. Reproducing those
    semantics means the delete and the upserts have to land together, or a crash between them
    leaves rows describing chunks that are gone.
    """
    rows = [
        (source, entry.get("owner", PUBLIC_OWNER), _json_dumps(entry), time.time())
        for source, entry in manifest.items()
    ]
    with _get_pool().connection() as conn, conn.cursor() as cur:
        if rows:
            cur.execute(
                "DELETE FROM corpus_manifest WHERE source <> ALL(%s)", ([r[0] for r in rows],)
            )
            cur.executemany(
                """
                INSERT INTO corpus_manifest (source, owner, entry, updated_at)
                VALUES (%s, %s, %s::jsonb, %s)
                ON CONFLICT (source) DO UPDATE SET
                    owner = EXCLUDED.owner,
                    entry = EXCLUDED.entry,
                    updated_at = EXCLUDED.updated_at
                """,
                rows,
            )
        else:
            cur.execute("DELETE FROM corpus_manifest")
        conn.commit()


def replace_parents(source: str, owner: str, parents: dict[str, str]) -> None:
    """Delete-then-insert, for the same reason the SQLite implementation does it: a re-indexed
    file may produce *fewer* sections than before, and an upsert would leave the vanished ones
    behind as orphans nothing points at and nothing cleans up."""
    with _get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_parents WHERE source = %s", (source,))
        if parents:
            cur.executemany(
                "INSERT INTO corpus_parents (parent_id, source, owner, content) "
                "VALUES (%s, %s, %s, %s)",
                [(pid, source, owner, content) for pid, content in parents.items()],
            )
        conn.commit()


def delete_parents(source: str) -> None:
    with _get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_parents WHERE source = %s", (source,))
        conn.commit()


def get_parent_contents(parent_ids: list[str]) -> dict[str, str]:
    if not parent_ids:
        return {}
    with _get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT parent_id, content FROM corpus_parents WHERE parent_id = ANY(%s)",
            (list(parent_ids),),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def count_parent_rows() -> int:
    with _get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM corpus_parents")
        return cur.fetchone()[0]


def current_index_version() -> int:
    with _get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT version FROM corpus_index_state WHERE id = 1")
        row = cur.fetchone()
        return int(row[0]) if row else 0


def bump_index_version() -> int:
    """Called once per ingest that changed anything. Atomic in the database rather than
    read-increment-write in Python, so two replicas ingesting at once cannot both read the
    same version and write the same successor."""
    with _get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE corpus_index_state SET version = version + 1 WHERE id = 1 RETURNING version"
        )
        row = cur.fetchone()
        conn.commit()
        return int(row[0]) if row else 0


def load_index_metadata_row() -> dict | None:
    with _get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT embedding_model, embedding_dimension, updated_at "
            "FROM corpus_index_metadata WHERE id = 1"
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "embedding_model": row[0],
        "embedding_dimension": row[1],
        "updated_at": float(row[2]),
    }


def save_index_metadata_row(
    embedding_model: str, embedding_dimension: int | None, updated_at: float
) -> None:
    with _get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO corpus_index_metadata (id, embedding_model, embedding_dimension, updated_at)
            VALUES (1, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding_model = EXCLUDED.embedding_model,
                embedding_dimension = EXCLUDED.embedding_dimension,
                updated_at = EXCLUDED.updated_at
            """,
            (embedding_model, embedding_dimension, updated_at),
        )
        conn.commit()
