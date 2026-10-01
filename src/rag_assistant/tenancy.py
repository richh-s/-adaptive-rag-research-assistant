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

from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.ingestion.generations import active_index_dir
from rag_assistant.conversations import store as conversations_store
from rag_assistant.ingestion.acl import entry_readable, sidecar_path
from rag_assistant.ingestion.manifest import load_manifest, save_manifest
from rag_assistant.ingestion.ownership import owner_corpus_dir
from rag_assistant.retrieval.bm25_store import invalidate_bm25_index
from rag_assistant.retrieval.parent_store import delete_parents_for_source
from rag_assistant.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class SourceNotFound(LookupError):
    """No such source, or not one this tenant owns.

    One exception for both, and deliberately so: distinguishing them would answer "does
    tenant B have a file called payroll.xlsx" for anyone willing to ask, which is the same
    leak the retrieval filter exists to prevent.
    """


@dataclass
class SourcePurgeResult:
    """What removing one document actually removed."""

    source: str
    chunks: int = 0
    file_removed: bool = False


def purge_source(
    source: str,
    owner: str,
    persist_dir: Path | None = None,
    principals: frozenset[str] | None = None,
) -> SourcePurgeResult:
    """Removes one document: its chunks, its parent sections, its manifest entry and the
    uploaded file itself.

    Erasure used to be all-or-nothing per tenant, which meant a takedown request, an expired
    document or one bad file had no answer short of deleting everything the tenant owned and
    re-uploading the rest. The stores are the same five `purge_tenant` walks; the difference
    is the unit.

    Same ordering as `purge_tenant`, for the same reason: the manifest entry is what says
    which chunks and parents belong to this source, so it is removed last and the operation
    is resumable rather than atomic. An interrupted purge leaves work still described, and
    running it again finishes it.

    `principals` is the caller's (see ingestion/acl.py): a document they may not read is
    reported as absent, exactly like another tenant's. Being able to delete a document is
    being able to learn it exists, and to destroy it.
    """
    settings = get_settings()
    persist_dir = persist_dir or active_index_dir()
    manifest = load_manifest(persist_dir)
    entry = manifest.get(source)
    if (
        entry is None
        or entry.get("owner", PUBLIC_OWNER) != owner
        or not entry_readable(entry, principals)
    ):
        raise SourceNotFound(source)

    result = SourcePurgeResult(source=source)
    chunk_ids = entry.get("chunk_ids") or []
    if chunk_ids:
        get_vector_store(persist_dir=persist_dir).delete(ids=chunk_ids)
        result.chunks = len(chunk_ids)
    delete_parents_for_source(persist_dir, source)
    del manifest[source]
    save_manifest(persist_dir, manifest)
    invalidate_bm25_index(persist_dir)
    if settings.vector_backend == "pgvector":
        from rag_assistant.retrieval.pgvector_store import bump_index_version

        bump_index_version()

    result.file_removed = _remove_corpus_file(settings.corpus_dir, source)
    logger.info(
        "source purged",
        extra={"owner": owner, "route": source, "node": f"chunks={result.chunks}"},
    )
    return result


def corpus_path(source: str) -> Path | None:
    """The file behind a source key, or None if the key would resolve outside the corpus."""
    corpus_dir = Path(get_settings().corpus_dir).resolve()
    try:
        target = (corpus_dir / source).resolve()
        target.relative_to(corpus_dir)
    except (ValueError, OSError):
        return None
    return target


def _remove_corpus_file(corpus_dir: Path, source: str) -> bool:
    """Deletes the uploaded file behind a source, if it is still there.

    `source` is a manifest key rather than request input, but it is *derived* from an
    uploaded filename, and the cost of confirming it resolves inside the corpus is one
    comparison against the cost of a traversal deleting something else entirely.
    """
    corpus_dir = Path(corpus_dir).resolve()
    try:
        target = (corpus_dir / source).resolve()
        target.relative_to(corpus_dir)
    except (ValueError, OSError):
        logger.warning("refusing to delete a source resolving outside the corpus: %r", source)
        return False
    # The permission sidecar goes with the document: left behind, it would silently apply to
    # a different file uploaded later under the same name.
    sidecar_path(target).unlink(missing_ok=True)
    if not target.is_file():
        return False
    target.unlink()
    return True


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
    persist_dir = persist_dir or active_index_dir()
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


@dataclass
class TenantUsage:
    """What one tenant currently occupies and has spent today."""

    sources: int
    chunks: int
    corpus_bytes: int
    tokens_used_today: int
    daily_token_budget: int


def tenant_usage(owner: str, persist_dir: Path | None = None) -> TenantUsage:
    """Index footprint and today's token spend for one tenant.

    Spend was already metered per tenant and index size was not, which left the two halves of
    "what is this tenant costing me" in different places -- one queryable, one only knowable by
    reading the manifest by hand. Both are cheap to read: the manifest already records an owner
    and chunk ids per source, and the corpus subtree is a stat() walk.
    """
    from rag_assistant import budget
    from rag_assistant.auth import PUBLIC_OWNER
    from rag_assistant.ingestion.manifest import load_manifest
    from rag_assistant.ingestion.ownership import owner_corpus_dir

    settings = get_settings()
    persist_dir = persist_dir or active_index_dir()
    manifest = load_manifest(persist_dir)
    mine = [entry for entry in manifest.values() if entry.get("owner", PUBLIC_OWNER) == owner]
    corpus_dir = owner_corpus_dir(settings.corpus_dir, owner)
    # Only this tenant's own subtree: for the public tenant that is the corpus root, which is
    # also where a single-tenant deployment keeps everything.
    corpus_bytes = (
        sum(p.stat().st_size for p in corpus_dir.rglob("*") if p.is_file())
        if corpus_dir.exists()
        else 0
    )
    return TenantUsage(
        sources=len(mine),
        chunks=sum(len(entry.get("chunk_ids", [])) for entry in mine),
        corpus_bytes=corpus_bytes,
        tokens_used_today=budget.used_tokens(owner),
        daily_token_budget=settings.tenant_daily_token_budget,
    )
