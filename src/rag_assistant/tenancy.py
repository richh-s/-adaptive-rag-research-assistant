"""Erasing everything belonging to one tenant.

Retention bounds how long data lives; this answers a different question -- "delete my data
now" -- and it has to reach every store that holds any, or it is not an erasure. A tenant's
footprint spans five places, and four of them are easy to forget:

    corpus files          the uploaded documents themselves
    vector chunks         embeddings, in Chroma or pgvector
    parent sections       small-to-big bodies, keyed by source
    manifest entries      the record of what is indexed
    conversations         transcripts, messages and feedback

Order matters. Files go last, because every earlier step is driven from the manifest and a
deleted file makes the record of what to delete unrecoverable. The manifest entry for a
source is removed only after that source's chunks and parents are gone, so an interrupted
purge leaves work still described and a re-run finishes it -- the operation is resumable
rather than atomic, which is the honest trade for something touching five stores with no
shared transaction.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.config import get_settings
from rag_assistant.conversations import store as conversations_store
from rag_assistant.ingestion.manifest import load_manifest, save_manifest
from rag_assistant.ingestion.ownership import owner_corpus_dir
from rag_assistant.retrieval.bm25_store import invalidate_bm25_index
from rag_assistant.retrieval.parent_store import delete_parents_for_source
from rag_assistant.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class PurgeResult:
    sources: int = 0
    chunks: int = 0
    conversations: int = 0
    feedback: int = 0
    files_removed: bool = False


def purge_tenant(owner: str, persist_dir: Path | None = None) -> PurgeResult:
    """Removes every trace of `owner`. Returns what was removed, for the audit log."""
    settings = get_settings()
    persist_dir = persist_dir or settings.chroma_persist_dir
    result = PurgeResult()

    manifest = load_manifest(persist_dir)
    owned = {source: entry for source, entry in manifest.items() if entry.get("owner") == owner}
    if owned:
        store = get_vector_store(persist_dir=persist_dir)
        for source, entry in owned.items():
            chunk_ids = entry.get("chunk_ids") or []
            if chunk_ids:
                store.delete(ids=chunk_ids)
                result.chunks += len(chunk_ids)
            delete_parents_for_source(persist_dir, source)
            del manifest[source]
            result.sources += 1
        save_manifest(persist_dir, manifest)
        # The in-memory keyword index is built from the chunks that just disappeared. Without
        # this, BM25 keeps serving the purged tenant's text until the process restarts.
        invalidate_bm25_index(persist_dir)
        if settings.vector_backend == "pgvector":
            from rag_assistant.retrieval.pgvector_store import bump_index_version

            bump_index_version()

    result.conversations, result.feedback = conversations_store.delete_all_for_owner(owner)

    corpus_dir = owner_corpus_dir(settings.corpus_dir, owner)
    # The public tenant's "directory" is the shared corpus root, which holds every other
    # tenant's baseline documents. Deleting it would erase the corpus rather than one
    # tenant's slice of it.
    if corpus_dir != Path(settings.corpus_dir) and corpus_dir.exists():
        shutil.rmtree(corpus_dir)
        result.files_removed = True

    logger.info(
        "tenant purged",
        extra={
            "owner": owner,
            "sources": result.sources,
            "chunks": result.chunks,
            "conversations": result.conversations,
            "feedback": result.feedback,
        },
    )
    return result
