"""What the current index was built with, recorded next to the collection.

Chunking and loader changes are caught per file by the manifest's version fields. The
embedding model is different in kind: it isn't a property of any one file, it's a property of
the whole vector space, and getting it wrong doesn't fail — it silently returns nonsense.

Point `GEMINI_EMBEDDING_MODEL` at a different model and restart without re-indexing, and every
query is embedded into a space the stored vectors don't live in. Chroma will happily compute
cosine distances between them and return the four nearest of the wrong thing. Retrieval looks
like it worked, grading scores the results, synthesis cites them, and the answer is confidently
wrong with no error anywhere in the logs. A dimension change is caught by Chroma; a *same
dimension, different model* change is not caught by anything.

So the model is recorded at index time, and every reader of the index embeds with the model
the index recorded rather than the one currently configured (see vector_store.index_embeddings).
Changing the configured model therefore no longer changes what an existing index is queried
with; it chooses the model the *next* index generation is built with, and `rag-assistant
reindex` is how an index moves to it without downtime (see generations.py). Readiness reports
the difference as a pending migration, and fails only when the recorded model cannot be
constructed at all -- its provider's credentials or server are missing.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_assistant.config import get_settings

logger = logging.getLogger(__name__)


def _shared_backend() -> bool:
    """See ingestion/manifest.py. This record has to live wherever the vectors do: left on
    local disk while the index moved to Postgres, a replica that never ingested has no record
    to compare against, reports "cannot verify", and is therefore never checked at all."""
    return get_settings().vector_backend == "pgvector"


INDEX_METADATA_FILENAME = "index_metadata.json"


@dataclass
class IndexMetadata:
    embedding_model: str
    # Best-effort: read back from the collection rather than by embedding a probe, so
    # recording it costs no API call. None when the collection was empty or unreadable.
    embedding_dimension: int | None = None
    updated_at: float = 0.0
    # How tenants are laid out in this index's Chroma collections: "filter" (one shared
    # collection) or "strict" (one per tenant). Recorded per index for the same reason the
    # model is: it is a property of what was written, and reading an index with the other
    # layout finds nothing. None for an index recorded before the field existed -- which was
    # necessarily the shared layout.
    tenant_isolation: str | None = None


def index_metadata_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / INDEX_METADATA_FILENAME


def load_index_metadata(persist_dir: Path) -> IndexMetadata | None:
    """None when nothing has been indexed yet, or when the file predates this feature --
    both mean "no recorded model", which callers must treat as "cannot verify", never as
    "verified fine"."""
    if _shared_backend():
        from rag_assistant.ingestion.generations import generation_of_dir
        from rag_assistant.retrieval.pgvector_store import load_index_metadata_row

        try:
            row = load_index_metadata_row(generation_of_dir(persist_dir))
        except Exception:
            logger.warning("Could not read index metadata from Postgres", exc_info=True)
            return None
        return IndexMetadata(**row) if row else None
    path = index_metadata_path(persist_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return IndexMetadata(
            embedding_model=payload["embedding_model"],
            embedding_dimension=payload.get("embedding_dimension"),
            updated_at=payload.get("updated_at", 0.0),
            tenant_isolation=payload.get("tenant_isolation"),
        )
    except Exception:
        logger.warning("Unreadable index metadata at %s; treating as absent", path, exc_info=True)
        return None


def save_index_metadata(
    persist_dir: Path,
    embedding_model: str,
    embedding_dimension: int | None = None,
    tenant_isolation: str | None = None,
) -> None:
    """`tenant_isolation` omitted keeps whatever the index already recorded -- every ingest
    re-saves this record, and an ingest must never change an index's layout -- or, for an
    index recording one for the first time, the configured layout."""
    if tenant_isolation is None:
        tenant_isolation = index_layout(persist_dir)
    metadata = IndexMetadata(
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        updated_at=time.time(),
        tenant_isolation=tenant_isolation,
    )
    if _shared_backend():
        from rag_assistant.retrieval.pgvector_store import save_index_metadata_row

        from rag_assistant.ingestion.generations import generation_of_dir

        save_index_metadata_row(
            embedding_model=metadata.embedding_model,
            embedding_dimension=metadata.embedding_dimension,
            updated_at=metadata.updated_at,
            generation=generation_of_dir(persist_dir),
        )
        return
    path = index_metadata_path(persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n")


def read_embedding_dimension(store) -> int | None:
    """Dimension of the stored vectors, read from the collection itself.

    Uses Chroma's private `_collection` the same way readiness.py's `_collection.count()`
    does. Best-effort throughout: an empty collection, a Chroma version that shapes `peek`
    differently, or anything else returns None, because failing to record a dimension must
    never fail an ingest that otherwise succeeded.
    """
    try:
        peeked = store._collection.peek(1)
        embeddings = peeked.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])
    except Exception:
        logger.debug("Could not read embedding dimension from the collection", exc_info=True)
        return None


def index_layout(persist_dir: Path) -> str:
    """The tenant layout an index was written with (see IndexMetadata.tenant_isolation).
    A fresh index takes the configured one; an index recorded before layouts existed is the
    shared layout, whatever is configured now."""
    from rag_assistant.config import get_settings

    if _shared_backend():
        # pgvector has one layout; its isolation is row-level security on one table.
        return get_settings().tenant_isolation
    metadata = load_index_metadata(persist_dir)
    if metadata is None:
        return get_settings().tenant_isolation
    return metadata.tenant_isolation or "filter"


def index_embedding_model(persist_dir: Path) -> str | None:
    """The embedding model an index recorded, or None when it has recorded nothing -- in
    which case its readers and writers use the configured model, and the first ingest records
    it."""
    metadata = load_index_metadata(persist_dir)
    return metadata.embedding_model if metadata else None


def check_embedding_model(persist_dir: Path, configured_model: str) -> tuple[bool, str | None]:
    """(ok, error). Unverifiable states report ok -- a fresh deployment with nothing indexed
    yet is not misconfigured, and reporting it as such would keep a healthy replica out of
    the load balancer forever."""
    metadata = load_index_metadata(persist_dir)
    if metadata is None:
        return True, None
    if metadata.embedding_model == configured_model:
        return True, None
    return False, (
        f"Index was built with embedding model {metadata.embedding_model!r} but "
        f"{configured_model!r} is configured. Queries would be embedded into a different "
        f"vector space than the stored documents, which returns plausible-looking but "
        f"meaningless results rather than an error. Re-index with "
        f"`rag-assistant ingest --full`, or restore the previous embedding model."
    )
