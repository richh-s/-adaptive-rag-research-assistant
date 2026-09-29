"""pgvector-backed vector store, interchangeable with the embedded Chroma default.

`VECTOR_BACKEND=pgvector` plus `DATABASE_URL` switches to it; nothing that calls
`retrieval.vector_store` knows which is running. This is the same argument the Postgres
conversations backend makes, applied to the index: embedded Chroma is SQLite-backed and locks
its file to one process, which is the single hardest constraint on running more than one
worker. Chroma server mode removes that too, but it means operating another service; if
Postgres is already in the deployment for conversations, pgvector makes the index a table in
a database that is already backed up, replicated and monitored.

Things that are deliberately different from the Chroma path:

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

* **Tenant isolation is enforced by the database as well as by the query.** The chunks table
  carries a row-level security policy keyed on a per-connection setting, `rag.visible_owners`,
  that every connection this module hands out sets before use. A search that forgot its
  tenant predicate -- the one-bug-away leak a shared table otherwise is -- still sees only the
  caller's rows. The policy fails closed: a connection that never set the value sees nothing.
  Two caveats are real and reported by `row_security_status()`: Postgres superusers and roles
  with BYPASSRLS skip every policy, so the application must connect as an ordinary role for
  this layer to exist at all.

* **Index generations are schemas.** The legacy index lives in `public`; each generation
  built by `rag-assistant reindex` lives in its own `rag_idx_<id>` schema with the same table
  names (see ingestion/generations.py). Every connection sets `search_path` to the generation
  it serves, so the SQL below is written once and runs unchanged against any of them.

Cosine distance (`<=>`) throughout, matching the Chroma collection's `hnsw:space: cosine` --
Gemini's embeddings are meant to be compared that way, and a store that ranked by L2 while
the other ranked by cosine would not be the interchangeable backend this claims to be.
"""

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.advisory_lock import advisory_lock_id
from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.ingestion.acl import META_ACL, META_RESTRICTED
from rag_assistant.ingestion.generations import LEGACY, schema_for_generation
from rag_assistant.ingestion.ownership import visible_owners
from rag_assistant.retrieval.mmr import maximal_marginal_relevance

logger = logging.getLogger(__name__)

TABLE_NAME = "corpus_chunks"
POINTER_TABLE = "rag_active_index"
_GENERATION_SCHEMA_PREFIX = "rag_idx_"

# The value of `rag.visible_owners` that lets a connection see every tenant's rows. Only the
# paths that genuinely operate on the whole index use it: ingestion writes, the BM25 build,
# readiness counts, backup. Everything that answers a caller's question is scoped.
ALL_OWNERS = "*"

# Distinct from the conversations backend's lock id: the two migration chains are
# independent and may run against the same database at the same time, so sharing a lock id
# would serialise unrelated startups and, worse, make a failure in one look like a hang in
# the other.
# Computed by `advisory_lock_id` rather than `hash()`, which is salted per process:
# every replica would otherwise take a different lock and contend with nobody.
_MIGRATION_LOCK_ID = advisory_lock_id("rag_assistant_pgvector_migrations")

# Re-entrant: `_ensure_migrated` holds it while `_migrate` borrows from the pool, and the
# pool's own lazy construction takes it too.
_LOCK = threading.RLock()
_pool = None
# Schemas whose migration chain this process has already run.
_migrated: set[str] = set()


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


def _migration_004_fulltext(cur) -> None:
    """A full-text index on the chunks, for keyword search that does not live in RAM.

    The in-memory BM25 index is built by reading *every* chunk in the collection into the
    process and keeping it there, once per replica. That is correct and free for a corpus of
    a few thousand chunks and it is the first thing to break as one grows: build time and
    resident memory both scale with the whole collection, on every replica, and a restart
    pays for it again before the first keyword query can be served. pgvector removed the
    vector index's single-process constraint; this removes the keyword index's memory one.

    A stored generated column rather than a trigger or an expression index: the tsvector is
    then maintained by Postgres on every insert and update with no application code that can
    forget to run, and `ts_rank_cd` can read it without recomputing `to_tsvector` per row.

    'english' is a fixed configuration, which is a real limitation and the honest one to
    state: stemming and stop words are language-specific, so a non-English corpus is stemmed
    by the wrong rules here. The in-memory scorer has the mirror-image property -- it does no
    stemming at all -- so neither is right for a multilingual corpus, and the two are wrong in
    different directions.
    """
    cur.execute(
        f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS content_tsv tsvector "
        f"GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_content_tsv "
        f"ON {TABLE_NAME} USING GIN (content_tsv)"
    )


def _migration_005_row_level_security(cur) -> None:
    """Tenant isolation enforced by Postgres, not only by each query's WHERE clause.

    FORCE as well as ENABLE, because the application usually connects as the table's owner,
    and an owner is exempt from its own table's policies unless forced. The policy reads a
    connection setting rather than `current_user`: tenants are not database roles, and
    minting one per tenant would turn onboarding into DDL.

    `current_setting(..., true)` returns NULL when the setting was never made, and NULL
    satisfies neither branch -- so a connection that skipped `_connection()` sees no rows at
    all rather than every row. Failing closed is the whole point of the layer.

    The policy only filters reads and checks writes; the manifest, parents and state tables
    are left without one. They hold no text a query returns to a caller except parent
    sections, which are fetched by the ids of chunks the caller was already allowed to see.
    """
    cur.execute(f"ALTER TABLE {TABLE_NAME} ENABLE ROW LEVEL SECURITY")
    cur.execute(f"ALTER TABLE {TABLE_NAME} FORCE ROW LEVEL SECURITY")
    cur.execute(f"DROP POLICY IF EXISTS tenant_scope ON {TABLE_NAME}")
    cur.execute(
        f"""
        CREATE POLICY tenant_scope ON {TABLE_NAME}
        USING (
            current_setting('rag.visible_owners', true) = '*'
            OR owner = ANY(string_to_array(current_setting('rag.visible_owners', true), ','))
        )
        """
    )


def _migration_006_generation_pointer(cur) -> None:
    """Which index generation serves (see ingestion/generations.py).

    Schema-qualified on purpose: the migration chain runs once per generation schema, and the
    pointer must exist exactly once, in `public`, where every replica looks for it.
    """
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS public.{POINTER_TABLE} (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            generation TEXT NOT NULL,
            previous TEXT,
            switched_at DOUBLE PRECISION NOT NULL
        )
        """
    )


_MIGRATIONS: list = [
    _migration_001_baseline,
    _migration_002_index_state,
    _migration_003_index_metadata,
    _migration_004_fulltext,
    _migration_005_row_level_security,
    _migration_006_generation_pointer,
]


def _get_pool():
    """One connection pool per process, mirroring the conversations backend.

    A pool rather than a shared connection because LangGraph's `Send` fan-out runs
    `retrieve_vector` for several sub-queries concurrently on a thread pool, and Postgres
    connections are not safe to share across threads.
    """
    global _pool
    if _pool is None:
        with _LOCK:
            if _pool is None:
                from psycopg_pool import ConnectionPool

                settings = get_settings()
                if not settings.database_url:
                    raise RuntimeError("VECTOR_BACKEND=pgvector requires DATABASE_URL to be set.")
                _pool = ConnectionPool(settings.database_url, min_size=1, open=True)
    return _pool


def reset_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
    _pool = None
    _migrated.clear()


def _migration_lock_id(schema: str) -> int:
    # `public` keeps the lock id it always had, so a replica still running the previous build
    # contends on the same lock during a rolling deploy instead of racing it.
    if schema == "public":
        return _MIGRATION_LOCK_ID
    return advisory_lock_id(f"rag_assistant_pgvector_migrations:{schema}")


def _migrate(schema: str) -> None:
    """Applies pending migrations to one schema, each in its own transaction, under an
    advisory lock so several replicas can start at once without racing each other onto the
    same migration."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            lock_id = _migration_lock_id(schema)
            cur.execute("SELECT pg_advisory_lock(%s)", (lock_id,))
            try:
                if schema != "public":
                    cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                cur.execute("SELECT set_config('search_path', %s, false)", (f"{schema}, public",))
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
                        "applying pgvector migration %d (%s) to schema %s",
                        version + 1,
                        migration.__name__,
                        schema,
                    )
                    migration(cur)
                    cur.execute(
                        "INSERT INTO pgvector_schema_migrations (version, applied_at) "
                        "VALUES (%s, %s)",
                        (version + 1, time.time()),
                    )
                    conn.commit()
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
                conn.commit()


def _ensure_migrated(schema: str) -> None:
    if schema in _migrated:
        return
    with _LOCK:
        if schema in _migrated:
            return
        # The pointer table lives in `public`, so a generation schema is only usable once
        # `public` has been migrated too.
        if schema != "public" and "public" not in _migrated:
            _migrate("public")
            _migrated.add("public")
        _migrate(schema)
        _migrated.add(schema)


def _scope_value(owners: list[str] | str) -> str:
    if owners == ALL_OWNERS:
        return ALL_OWNERS
    # Owner labels are filesystem-safe by construction (see ownership.safe_owner_dirname) and
    # can therefore never contain the comma this is split on.
    return ",".join(owners)


@contextmanager
def _connection(generation: str = LEGACY, owners: list[str] | str = ALL_OWNERS) -> Iterator:
    """A pooled connection pointed at one generation's schema and scoped to `owners`.

    Every connection in this module goes through here, and that is the invariant row-level
    security depends on: the scope is set on each borrow, so a connection returned to the pool
    by one request cannot carry its scope into the next. Session-level rather than
    transaction-local, because several functions commit midway and a transaction-local value
    would silently reset to "nothing visible" after the first commit.
    """
    schema = schema_for_generation(generation)
    _ensure_migrated(schema)
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('search_path', %s, false), "
                "set_config('rag.visible_owners', %s, false)",
                (f"{schema}, public", _scope_value(owners)),
            )
        yield conn


def _to_vector_literal(values) -> str:
    """pgvector's text input format. Sent as a string and cast with `::vector` rather than
    via a registered type adapter, so the backend needs no import beyond psycopg, which is
    already a dependency for the conversations store."""
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def _column_dimension(cur) -> int | None:
    """The width the embedding column is currently constrained to, or None while it is still
    unconstrained. pgvector stores it in `atttypmod`, which is -1 for a bare `vector`.

    Resolved through `to_regclass`, which honours the connection's search_path, so this reads
    the table of the generation being written rather than whichever schema's table of the same
    name happens to be found first in the catalog.
    """
    cur.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
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
        with _connection(self._store.generation) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")  # nosec B608  # table name is a module constant; every value is a bound parameter
            return cur.fetchone()[0]

    def peek(self, limit: int = 1) -> dict:
        with _connection(self._store.generation) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT embedding FROM {TABLE_NAME} LIMIT %s", (limit,))  # nosec B608  # table name is a module constant; every value is a bound parameter
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

    def __init__(self, embeddings: Embeddings, generation: str = LEGACY):
        self.embeddings = embeddings
        self.generation = generation
        self._collection = _Collection(self)

    # ---- writes ----

    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        if not documents:
            return []
        vectors = self.embeddings.embed_documents([d.page_content for d in documents])
        return self.add_embedded(documents, ids, vectors)

    def add_embedded(
        self, documents: list[Document], ids: list[str], vectors: list[list[float]]
    ) -> list[str]:
        """Inserts chunks whose vectors are already computed -- what a re-embed into a new
        generation does after batching the embedding calls itself."""
        if not documents:
            return []
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
        with _connection(self.generation) as conn, conn.cursor() as cur:
            _ensure_dimension(conn, cur, len(vectors[0]))
            # ON CONFLICT rather than delete-then-insert: build_index already deletes a
            # source's previous chunk ids before re-adding, but chunk ids are derived from
            # (source, index) and so collide by design when a file is re-chunked into the
            # same count. An upsert makes a re-ingest idempotent either way.
            # Implicit concatenation rather than a triple-quoted block so the reviewed-and-
            # cleared suppression marker can sit on the interpolating line. Inside a string
            # literal it would not be a comment at all -- it would be part of the SQL.
            cur.executemany(
                f"INSERT INTO {TABLE_NAME} "  # nosec B608  # table name is a module constant; every value is a bound parameter
                "(chunk_id, content, embedding, owner, source, ingested_at, metadata) "
                "VALUES (%s, %s, %s::vector, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (chunk_id) DO UPDATE SET "
                "content = EXCLUDED.content, "
                "embedding = EXCLUDED.embedding, "
                "owner = EXCLUDED.owner, "
                "source = EXCLUDED.source, "
                "ingested_at = EXCLUDED.ingested_at, "
                "metadata = EXCLUDED.metadata",
                rows,
            )
            conn.commit()
        return list(ids)

    def update_metadata(self, ids: list[str], patch: dict) -> None:
        """Merges `patch` into the metadata of the given chunks without touching their
        vectors -- how a permission change reaches the index without re-embedding."""
        if not ids:
            return
        with _connection(self.generation) as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TABLE_NAME} SET metadata = metadata || %s::jsonb "  # nosec B608  # table name is a module constant; every value is a bound parameter
                "WHERE chunk_id = ANY(%s)",
                (_json_dumps(patch), list(ids)),
            )
            conn.commit()

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        with _connection(self.generation) as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE_NAME} WHERE chunk_id = ANY(%s)", (list(ids),))  # nosec B608  # table name is a module constant; every value is a bound parameter
            conn.commit()

    def reset_collection(self) -> None:
        with _connection(self.generation) as conn, conn.cursor() as cur:
            cur.execute(f"TRUNCATE {TABLE_NAME}")
            conn.commit()

    # ---- reads ----

    def get(self, ids: list[str] | None = None, include: list[str] | None = None) -> dict:
        """Chroma's `get` shape: parallel lists under `ids`/`documents`/`metadatas`, plus
        `embeddings` when asked for.

        Other fields in `include` are accepted and ignored -- the three columns cost the same
        one row fetch here, and honouring it would only let a caller receive fewer fields than
        it asked for on one backend and not the other. Embeddings are the exception because
        they are large, and only a re-embed-free copy between generations wants them.
        """
        with_embeddings = bool(include and "embeddings" in include)
        columns = "chunk_id, content, metadata" + (", embedding" if with_embeddings else "")
        sql = f"SELECT {columns} FROM {TABLE_NAME}"  # nosec B608  # table and column names are module constants; every value is a bound parameter
        params: tuple = ()
        if ids is not None:
            sql += " WHERE chunk_id = ANY(%s)"
            params = (list(ids),)
        with _connection(self.generation) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        result = {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [dict(r[2] or {}) for r in rows],
        }
        if with_embeddings:
            result["embeddings"] = [_parse_vector(r[3]) for r in rows]
        return result

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        owner: str = PUBLIC_OWNER,
        filters=None,
        principals: frozenset[str] | None = None,
    ) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        settings = get_settings()
        if settings.retrieval_mmr:
            return self._mmr_search(
                query_vector, k=k, owner=owner, filters=filters, principals=principals
            )
        vector = _to_vector_literal(query_vector)
        where, params = build_where_sql(owner, filters, principals)
        with _connection(self.generation, visible_owners(owner)) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT content, metadata FROM {TABLE_NAME} "  # nosec B608  # table name is a module constant; every value is a bound parameter
                f"WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s",
                (*params, vector, k),
            )
            rows = cur.fetchall()
        return [Document(page_content=r[0], metadata=dict(r[1] or {})) for r in rows]

    def _mmr_search(
        self,
        query_vector: list[float],
        k: int,
        owner: str,
        filters,
        principals: frozenset[str] | None = None,
    ) -> list[Document]:
        """Diversity selection over the nearest `fetch_k`.

        The shortlist is still chosen by the HNSW index -- MMR reorders what similarity
        already found rather than replacing it, so the expensive part stays in Postgres and
        only tens of vectors cross the wire. Selection itself runs through the same
        `retrieval/mmr.py` the Chroma path uses, which is what keeps the two backends
        returning the same documents for the same corpus.
        """
        settings = get_settings()
        vector = _to_vector_literal(query_vector)
        where, params = build_where_sql(owner, filters, principals)
        with _connection(self.generation, visible_owners(owner)) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT content, metadata, embedding FROM {TABLE_NAME} "  # nosec B608  # table name is a module constant; every value is a bound parameter
                f"WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s",
                (*params, vector, max(settings.retrieval_fetch_k, k)),
            )
            rows = cur.fetchall()
        if not rows:
            return []
        picked = maximal_marginal_relevance(
            query_vector,
            [_parse_vector(row[2]) for row in rows],
            k=k,
            lambda_mult=settings.retrieval_mmr_lambda,
        )
        return [Document(page_content=rows[i][0], metadata=dict(rows[i][1] or {})) for i in picked]

    def as_retriever(
        self,
        k: int = 4,
        owner: str = PUBLIC_OWNER,
        filters=None,
        principals: frozenset[str] | None = None,
    ):
        return _PgVectorRetriever(self, k=k, owner=owner, filters=filters, principals=principals)


class _PgVectorRetriever:
    """`.invoke(query) -> list[Document]`, the only part of LangChain's retriever protocol
    the graph uses (see graph/nodes/retrieve.py)."""

    def __init__(self, store: PgVectorStore, k: int, owner: str, filters, principals=None):
        self._store = store
        self._k = k
        self._owner = owner
        self._filters = filters
        self._principals = principals

    def invoke(self, query: str) -> list[Document]:
        return self._store.similarity_search(
            query,
            k=self._k,
            owner=self._owner,
            filters=self._filters,
            principals=self._principals,
        )


def build_where_sql(
    owner: str, filters=None, principals: frozenset[str] | None = None
) -> tuple[str, tuple]:
    """The SQL counterpart of vector_store.build_where_clause -- tenant scope, document ACLs
    and the caller's metadata filters, as a predicate applied during the search.

    Returns (sql, params) rather than an interpolated string so every value reaches Postgres
    as a bound parameter; `source` in particular is caller-supplied.

    The tenant predicate stays even though row-level security enforces the same boundary:
    a superuser connection skips RLS, and the query must be correct on its own.
    """
    clauses = ["owner = ANY(%s)"]
    params: list = [visible_owners(owner)]
    if principals is not None:
        # A restricted chunk is visible when its ACL array shares any element with the
        # caller's principals. `?|` is jsonb's "contains any of these strings" operator.
        clauses.append(
            f"(COALESCE((metadata->>'{META_RESTRICTED}')::boolean, false) = false "
            f"OR (metadata->'{META_ACL}') ?| %s)"
        )
        params.append(sorted(principals))
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


def load_manifest_rows(generation: str = LEGACY) -> dict[str, dict]:
    with _connection(generation) as conn, conn.cursor() as cur:
        cur.execute("SELECT source, entry FROM corpus_manifest")
        return {row[0]: dict(row[1]) for row in cur.fetchall()}


def save_manifest_rows(manifest: dict[str, dict], generation: str = LEGACY) -> None:
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
    with _connection(generation) as conn, conn.cursor() as cur:
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


def replace_parents(
    source: str, owner: str, parents: dict[str, str], generation: str = LEGACY
) -> None:
    """Delete-then-insert, for the same reason the SQLite implementation does it: a re-indexed
    file may produce *fewer* sections than before, and an upsert would leave the vanished ones
    behind as orphans nothing points at and nothing cleans up."""
    with _connection(generation) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_parents WHERE source = %s", (source,))
        if parents:
            cur.executemany(
                "INSERT INTO corpus_parents (parent_id, source, owner, content) "
                "VALUES (%s, %s, %s, %s)",
                [(pid, source, owner, content) for pid, content in parents.items()],
            )
        conn.commit()


def delete_parents(source: str, generation: str = LEGACY) -> None:
    with _connection(generation) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_parents WHERE source = %s", (source,))
        conn.commit()


def get_parent_contents(parent_ids: list[str], generation: str = LEGACY) -> dict[str, str]:
    if not parent_ids:
        return {}
    with _connection(generation) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT parent_id, content FROM corpus_parents WHERE parent_id = ANY(%s)",
            (list(parent_ids),),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def parents_for_source(source: str, generation: str = LEGACY) -> dict[str, str]:
    with _connection(generation) as conn, conn.cursor() as cur:
        cur.execute("SELECT parent_id, content FROM corpus_parents WHERE source = %s", (source,))
        return {row[0]: row[1] for row in cur.fetchall()}


def count_parent_rows(generation: str = LEGACY) -> int:
    with _connection(generation) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM corpus_parents")
        return cur.fetchone()[0]


def current_index_version(generation: str = LEGACY) -> int:
    with _connection(generation) as conn, conn.cursor() as cur:
        cur.execute("SELECT version FROM corpus_index_state WHERE id = 1")
        row = cur.fetchone()
        return int(row[0]) if row else 0


def bump_index_version(generation: str = LEGACY) -> int:
    """Called once per ingest that changed anything. Atomic in the database rather than
    read-increment-write in Python, so two replicas ingesting at once cannot both read the
    same version and write the same successor."""
    with _connection(generation) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE corpus_index_state SET version = version + 1 WHERE id = 1 RETURNING version"
        )
        row = cur.fetchone()
        conn.commit()
        return int(row[0]) if row else 0


def load_index_metadata_row(generation: str = LEGACY) -> dict | None:
    with _connection(generation) as conn, conn.cursor() as cur:
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
    embedding_model: str,
    embedding_dimension: int | None,
    updated_at: float,
    generation: str = LEGACY,
) -> None:
    with _connection(generation) as conn, conn.cursor() as cur:
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


def keyword_search(
    query: str,
    k: int,
    owner: str,
    filters=None,
    principals: frozenset[str] | None = None,
    generation: str = LEGACY,
) -> list[tuple[str, dict, float]]:
    """Full-text keyword search, ranked by `ts_rank_cd`, as (content, metadata, score).

    Not BM25, and the difference is worth stating rather than hiding behind a shared function
    name. `ts_rank_cd` is cover-density ranking: it rewards query terms appearing close
    together and does not model document length or term saturation the way BM25's k1/b do. So
    the two backends will not return identical orderings for the same corpus -- unlike the
    vector backends, which do, and are tested for it.

    That is tolerable here specifically because of what consumes this. Fusion combines
    retrieval paths by Reciprocal Rank Fusion, which votes on *rank position* and never
    compares scores across paths -- precisely so that incomparable scoring schemes can be
    merged. A keyword path that ranks somewhat differently is the same kind of input RRF
    already accepts from vector search and the web.

    `websearch_to_tsquery` rather than `plainto_tsquery`: it tolerates whatever a user types,
    including quotes and `or`, without raising on syntax the way `to_tsquery` does -- and a
    keyword path that can be made to error by a question mark is worse than one that ranks
    imperfectly.
    """
    where, params = build_where_sql(owner, filters, principals)
    with _connection(generation, visible_owners(owner)) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT content, metadata, ts_rank_cd(content_tsv, q) AS rank "  # nosec B608  # table name is a module constant; every value is a bound parameter
            f"FROM {TABLE_NAME}, websearch_to_tsquery('english', %s) q "
            f"WHERE {where} AND content_tsv @@ q "
            f"ORDER BY rank DESC, chunk_id LIMIT %s",
            (query, *params, k),
        )
        rows = cur.fetchall()
    return [(row[0], dict(row[1] or {}), float(row[2])) for row in rows]


# ---- generations and the pointer (see ingestion/generations.py) ----


def load_active_pointer() -> tuple[str, str | None, float] | None:
    with _connection(LEGACY) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT generation, previous, switched_at FROM public.{POINTER_TABLE} WHERE id = 1"  # nosec B608  # table name is a module constant
        )
        row = cur.fetchone()
    if row is None:
        return None
    return row[0], row[1], float(row[2])


def save_active_pointer(generation: str, previous: str | None, switched_at: float) -> None:
    # Migrated before the flip, so the first request after it does not pay for a migration
    # chain -- and so a generation whose schema is somehow incomplete fails here, loudly,
    # rather than on the first query after every replica has switched to it.
    _ensure_migrated(schema_for_generation(generation))
    with _connection(LEGACY) as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO public.{POINTER_TABLE} (id, generation, previous, switched_at) "  # nosec B608  # table name is a module constant; every value is a bound parameter
            "VALUES (1, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET generation = EXCLUDED.generation, "
            "previous = EXCLUDED.previous, switched_at = EXCLUDED.switched_at",
            (generation, previous, switched_at),
        )
        conn.commit()


def list_generation_schemas() -> list[str]:
    with _connection(LEGACY) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT nspname FROM pg_namespace WHERE starts_with(nspname, %s)",
            (_GENERATION_SCHEMA_PREFIX,),
        )
        return [row[0].removeprefix(_GENERATION_SCHEMA_PREFIX) for row in cur.fetchall()]


def drop_generation(generation: str) -> None:
    """Removes a generation's schema and everything in it. The caller refuses the active
    generation and the legacy one; this refuses them again, because it cannot be undone."""
    if generation == LEGACY:
        raise ValueError("The legacy generation lives in `public` and is never dropped.")
    schema = schema_for_generation(generation)
    with _connection(LEGACY) as conn, conn.cursor() as cur:
        # The schema name is validated by schema_for_generation against a strict pattern and
        # cannot be a bound parameter in DDL.
        cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.commit()
    _migrated.discard(schema)


def row_security_status() -> tuple[bool, str | None]:
    """(enforced, detail). Row-level security is inert for superusers and BYPASSRLS roles,
    and a deployment connecting as one has only the query predicate between tenants. That is
    not a readiness failure -- the predicate is still correct -- but it is the kind of
    silently-absent control that should be visible to whoever is looking."""
    with _connection(LEGACY) as conn, conn.cursor() as cur:
        cur.execute("SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user")
        row = cur.fetchone()
    if row and row[0]:
        return False, (
            "connected as a superuser or BYPASSRLS role, so row-level tenant isolation is "
            "not enforced; connect as an ordinary role to enable it"
        )
    return True, None
