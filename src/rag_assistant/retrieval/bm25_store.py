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
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.ingestion.ownership import visible_owners
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
_index_cache: dict[str, tuple[BM25Okapi | None, list]] = {}
_index_lock = threading.Lock()


def _build_index(persist_dir: Path) -> tuple[BM25Okapi | None, list[Document]]:
    store = get_vector_store(persist_dir=persist_dir)
    try:
        stored = store.get(include=["documents", "metadatas"])
    except Exception:
        # A collection that doesn't exist yet (nothing ingested) is not an error condition --
        # keyword search simply has nothing to match, exactly as before ingestion.
        return None, []

    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    if not documents:
        return None, []

    # Sorted by chunk id so the index order (and therefore tie-breaking among equal BM25
    # scores) is stable across rebuilds; Chroma makes no ordering guarantee.
    rows = sorted(
        zip(ids, documents, metadatas),
        key=lambda row: row[0] or "",
    )
    chunks = [
        Document(page_content=content, metadata=dict(metadata or {}))
        for _, content, metadata in rows
    ]
    return BM25Okapi([_tokenize(chunk.page_content) for chunk in chunks]), chunks


def get_bm25_index(persist_dir: Path | None = None) -> tuple[BM25Okapi | None, list]:
    settings = get_settings()
    resolved = str(persist_dir or settings.chroma_persist_dir)
    if resolved not in _index_cache:
        with _index_lock:
            if resolved not in _index_cache:
                _index_cache[resolved] = _build_index(Path(resolved))
    return _index_cache[resolved]


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


def bm25_search(
    sub_query: str,
    k: int = 4,
    persist_dir: Path | None = None,
    owner: str = PUBLIC_OWNER,
) -> list[RetrievedDoc]:
    """Keyword search over the chunks `owner` may see.

    One index spans the whole collection and is filtered per query, rather than one index per
    tenant: BM25's IDF term is corpus-wide statistics, so per-tenant indexes would both
    duplicate memory and make a document's score depend on who is asking. The candidate set is
    narrowed *before* the top-k cut so a tenant always gets k of their own documents rather
    than k minus however many of someone else's outranked them.
    """
    bm25, chunks = get_bm25_index(persist_dir)
    if bm25 is None:
        return []

    allowed = set(visible_owners(owner))
    scores = bm25.get_scores(_tokenize(sub_query))
    visible_indices = [
        i for i in range(len(chunks)) if chunks[i].metadata.get("owner", PUBLIC_OWNER) in allowed
    ]
    ranked_indices = sorted(visible_indices, key=lambda i: scores[i], reverse=True)[:k]

    return [
        RetrievedDoc(
            content=chunks[i].page_content,
            metadata=chunks[i].metadata,
            source_id=chunks[i].metadata.get("source", ""),
            score=float(scores[i]),
        )
        for i in ranked_indices
        if scores[i] > 0  # no keyword overlap at all -- don't pad results with noise
    ]
