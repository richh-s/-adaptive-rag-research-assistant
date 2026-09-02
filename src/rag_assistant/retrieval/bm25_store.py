"""BM25 keyword retrieval over the same chunks Chroma holds.

The index is built from the Chroma collection rather than by re-reading and re-splitting the
corpus from disk, which fixes two separate problems.

The first is cost. Ingestion parses only changed files (see build_index), but BM25 previously
rebuilt by parsing *everything* -- so one tenant uploading a 2KB note re-parsed every PDF in
every tenant's corpus, and with PDF_VISION on that is a vision API call per figure and per
scanned page, paid again on every upload. Reading chunks that are already stored removes the
second parse pass entirely.

The second is correctness, and it is the more interesting one. RRF deduplicates by
SHA256(content), so a passage retrieved by both paths only collapses into one entry if the
two paths produced byte-identical text. That used to hold by convention -- both called the
same splitter with the same defaults -- and would have broken silently the moment either side
drifted, showing the same passage twice, splitting its own rank votes, and citing it twice.
Reading the indexed chunks makes the two sets identical by construction instead.

The tradeoff: BM25 now reflects what has been *indexed*, not what is on disk. A file dropped
into the corpus directory is invisible to keyword search until `ingest` runs -- which is the
honest behaviour, since it was already invisible to vector search.
"""

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.ingestion.ownership import visible_owners
from rag_assistant.retrieval.incremental_bm25 import IncrementalBM25
from rag_assistant.retrieval.vector_store import get_vector_store
from rag_assistant.schemas.models import RetrievedDoc

# Deliberately simple: no stemming, no stopword removal. BM25 is sensitive to exact token
# overlap, so e.g. "founded" vs "founding" won't match -- an accepted simplification for a
# small, low-vocabulary-variance corpus, not a production-grade tokenizer.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# rank_bm25 has no persistence API, so the index is rebuilt in-memory once per process and
# cached -- same lazy-singleton shape as vector_store.py's Chroma cache, now keyed by the
# persist directory it reads from rather than by a corpus directory.
@dataclass
class Bm25State:
    """The in-memory keyword index plus the chunks it scores, keyed by Chroma chunk id."""

    index: IncrementalBM25 = field(default_factory=IncrementalBM25)
    documents: dict[str, Document] = field(default_factory=dict)


_index_cache: dict[str, Bm25State] = {}
_index_lock = threading.Lock()


def _fetch_chunks(persist_dir: Path, ids: list[str] | None = None) -> list[tuple[str, Document]]:
    """Chunks from the collection: all of them, or just the given ids."""
    store = get_vector_store(persist_dir=persist_dir)
    try:
        stored = store.get(ids=ids, include=["documents", "metadatas"])
    except Exception:
        # A collection that doesn't exist yet (nothing ingested) is not an error condition --
        # keyword search simply has nothing to match, exactly as before ingestion.
        return []

    fetched_ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    # Sorted by chunk id so index order -- and therefore tie-breaking among equal BM25 scores
    # -- is stable across rebuilds; Chroma makes no ordering guarantee.
    return sorted(
        (
            (chunk_id, Document(page_content=content, metadata=dict(metadata or {})))
            for chunk_id, content, metadata in zip(fetched_ids, documents, metadatas)
        ),
        key=lambda row: row[0] or "",
    )


def _build_index(persist_dir: Path) -> Bm25State:
    state = Bm25State()
    for chunk_id, document in _fetch_chunks(persist_dir):
        state.index.add(chunk_id, _tokenize(document.page_content))
        state.documents[chunk_id] = document
    return state


def get_bm25_index(persist_dir: Path | None = None) -> Bm25State:
    settings = get_settings()
    resolved = str(persist_dir or settings.chroma_persist_dir)
    if resolved not in _index_cache:
        with _index_lock:
            if resolved not in _index_cache:
                _index_cache[resolved] = _build_index(Path(resolved))
    return _index_cache[resolved]


def apply_bm25_delta(
    persist_dir: Path | None = None,
    added_ids: list[str] | None = None,
    removed_ids: list[str] | None = None,
) -> bool:
    """Updates the cached index for exactly the chunks that changed.

    This is what makes an ingest cost the changed documents rather than the collection: only
    the added ids are fetched and tokenized, and removals are pure bookkeeping. Returns False
    when there is no cached index to update -- in that case the next query builds a fresh one
    from the collection anyway, so there is nothing to do and nothing has been missed.
    """
    settings = get_settings()
    resolved = str(persist_dir or settings.chroma_persist_dir)
    with _index_lock:
        state = _index_cache.get(resolved)
        if state is None:
            return False
        for chunk_id in removed_ids or []:
            state.index.remove(chunk_id)
            state.documents.pop(chunk_id, None)
        if added_ids:
            for chunk_id, document in _fetch_chunks(Path(resolved), ids=list(added_ids)):
                state.index.add(chunk_id, _tokenize(document.page_content))
                state.documents[chunk_id] = document
    return True


def invalidate_bm25_index(persist_dir: Path | None = None) -> None:
    """Drop the cached in-memory index so the next `get_bm25_index`/`bm25_search` call
    rebuilds it from the collection. Call this after ingestion adds, changes, or removes
    chunks -- the cache above is a lazy singleton with no other way to learn the collection
    changed. Popping under the same lock used to build it means a concurrent reader either
    sees the old index (a plain dict-key read, safe to interleave under the GIL) or triggers
    a fresh build; it never observes a torn/half-rebuilt one.
    """
    settings = get_settings()
    resolved = str(persist_dir or settings.chroma_persist_dir)
    with _index_lock:
        _index_cache.pop(resolved, None)


def _passes_filters(metadata: dict, filters) -> bool:
    """Mirrors vector_store.build_where_clause in Python.

    A chunk with no `ingested_at` (indexed before the field existed) fails any date filter
    rather than passing it: a date filter is a claim about when a document was indexed, and
    "unknown" cannot satisfy it. Re-indexing populates the field.
    """
    if filters is None or filters.is_empty():
        return True
    if filters.sources and metadata.get("source") not in set(filters.sources):
        return False
    ingested_at = metadata.get("ingested_at")
    if filters.ingested_after is not None:
        if ingested_at is None or ingested_at < filters.ingested_after.timestamp():
            return False
    if filters.ingested_before is not None:
        if ingested_at is None or ingested_at > filters.ingested_before.timestamp():
            return False
    return True


def bm25_search(
    sub_query: str,
    k: int = 4,
    persist_dir: Path | None = None,
    owner: str = PUBLIC_OWNER,
    filters=None,
) -> list[RetrievedDoc]:
    """Keyword search over the chunks `owner` may see.

    One index spans the whole collection and is filtered per query, rather than one index per
    tenant: BM25's IDF term is corpus-wide statistics, so per-tenant indexes would both
    duplicate memory and make a document's score depend on who is asking. The candidate set is
    narrowed *before* the top-k cut so a tenant always gets k of their own documents rather
    than k minus however many of someone else's outranked them.
    """
    state = get_bm25_index(persist_dir)
    if not state.documents:
        return []

    allowed = set(visible_owners(owner))
    scores = state.index.scores(_tokenize(sub_query))
    visible_ids = [
        chunk_id
        for chunk_id, document in state.documents.items()
        if document.metadata.get("owner", PUBLIC_OWNER) in allowed
        and _passes_filters(document.metadata, filters)
    ]
    ranked_ids = sorted(visible_ids, key=lambda i: scores.get(i, 0.0), reverse=True)[:k]

    return [
        RetrievedDoc(
            content=state.documents[chunk_id].page_content,
            metadata=state.documents[chunk_id].metadata,
            source_id=state.documents[chunk_id].metadata.get("source", ""),
            score=float(scores.get(chunk_id, 0.0)),
        )
        for chunk_id in ranked_ids
        # no keyword overlap at all -- don't pad results with noise
        if scores.get(chunk_id, 0.0) > 0
    ]
