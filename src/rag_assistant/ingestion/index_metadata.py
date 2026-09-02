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

So the model is recorded at index time and compared at readiness time, which turns an
invisible corruption into a replica that reports itself unable to serve.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

INDEX_METADATA_FILENAME = "index_metadata.json"


@dataclass
class IndexMetadata:
    embedding_model: str
    # Best-effort: read back from the collection rather than by embedding a probe, so
    # recording it costs no API call. None when the collection was empty or unreadable.
    embedding_dimension: int | None = None
    updated_at: float = 0.0


def index_metadata_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / INDEX_METADATA_FILENAME


def load_index_metadata(persist_dir: Path) -> IndexMetadata | None:
    """None when nothing has been indexed yet, or when the file predates this feature --
    both mean "no recorded model", which callers must treat as "cannot verify", never as
    "verified fine"."""
    path = index_metadata_path(persist_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return IndexMetadata(
            embedding_model=payload["embedding_model"],
            embedding_dimension=payload.get("embedding_dimension"),
            updated_at=payload.get("updated_at", 0.0),
        )
    except Exception:
        logger.warning("Unreadable index metadata at %s; treating as absent", path, exc_info=True)
        return None


def save_index_metadata(
    persist_dir: Path, embedding_model: str, embedding_dimension: int | None = None
) -> None:
    path = index_metadata_path(persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = IndexMetadata(
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        updated_at=time.time(),
    )
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
