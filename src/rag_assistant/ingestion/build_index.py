import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.config import get_settings
from rag_assistant.ingestion.loaders import load_documents
from rag_assistant.ingestion.manifest import hash_content, load_manifest, save_manifest
from rag_assistant.ingestion.splitter import split_documents
from rag_assistant.retrieval.bm25_store import get_bm25_index, invalidate_bm25_index
from rag_assistant.retrieval.vector_store import get_vector_store

# Serializes full build_index() runs. Needed for two reasons: (1) the manifest is a plain
# JSON file with a read-modify-write cycle and no locking of its own -- two ingestion runs
# overlapping (e.g. two near-simultaneous uploads via /api/v1/ingest) would race on it and
# could drop each other's changes; (2) it gives BM25 cache invalidation a clear
# happens-after relationship with the Chroma writes it follows, so a concurrent /research
# query never observes Chroma and BM25 disagreeing about what's indexed for longer than
# necessary. Chroma's own client cache (vector_store.py) and the BM25 index cache
# (bm25_store.py) each already guard their *own* construction with a lock; this lock is
# specifically for the higher-level ingest-pipeline sequence that calls into both.
INGEST_LOCK = threading.Lock()


@dataclass
class IndexResult:
    indexed_chunks: int
    changed_files: int
    skipped_files: int
    removed_files: int


def _chunk_ids(source: str, chunks: list[Document]) -> list[str]:
    return [f"{source}::{i}" for i in range(len(chunks))]


def _group_by_source(documents: list[Document]) -> dict[str, list[Document]]:
    # Most loaders (.md/.txt) yield exactly one Document per file, but PDFs yield one per
    # page sharing the same `source` -- group rather than assume 1:1 so multi-page PDFs
    # don't silently lose every page but the last.
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        grouped[doc.metadata["source"]].append(doc)
    return grouped


def build_index(
    source_dir: Path | None = None,
    persist_dir: Path | None = None,
    embeddings: Embeddings | None = None,
    incremental: bool = True,
    on_stage: Callable[[str, str], None] | None = None,
) -> IndexResult:
    """Load the corpus, chunk it, embed it, and (re)populate the Chroma collection.

    Incremental by default: a content-hash manifest alongside the Chroma collection tracks
    what was last indexed per source file, so unchanged files are skipped, changed files have
    their old chunks deleted and replaced, and files removed from the corpus have their chunks
    deleted too -- only new/changed content pays for embedding calls. Pass `incremental=False`
    to reset the collection and manifest and rebuild everything from scratch.

    `on_stage(stage, message)` is an optional hook fired at each phase transition (currently
    "parsing" and "indexing") -- callers that expose ingestion progress externally (e.g. the
    `/api/v1/ingest` background task) use it to update a task-status record without this
    function needing to know anything about tasks, HTTP, or polling.
    """
    settings = get_settings()
    source_dir = source_dir or settings.corpus_dir
    persist_dir = persist_dir or settings.chroma_persist_dir

    with INGEST_LOCK:
        if on_stage:
            on_stage("parsing", "Loading and parsing corpus files...")
        documents = load_documents(source_dir)
        store = get_vector_store(embeddings=embeddings, persist_dir=persist_dir)

        if not incremental:
            store.reset_collection()
            manifest: dict[str, dict] = {}
        else:
            manifest = load_manifest(persist_dir)

        docs_by_source = _group_by_source(documents)

        removed_sources = set(manifest) - set(docs_by_source)
        for source in removed_sources:
            store.delete(ids=manifest[source]["chunk_ids"])
            del manifest[source]

        if on_stage:
            on_stage("indexing", "Embedding and indexing changed files...")

        indexed_chunks = 0
        changed_files = 0
        skipped_files = 0
        for source, docs in docs_by_source.items():
            # Hash every page/Document's content together (order-sensitive) so a change to
            # any single page of a multi-page PDF is detected, not just whole-file changes.
            content_hash = hash_content("\x00".join(doc.page_content for doc in docs))
            existing = manifest.get(source)
            if existing and existing["hash"] == content_hash:
                skipped_files += 1
                continue

            if existing:
                store.delete(ids=existing["chunk_ids"])

            chunks = split_documents(docs)
            chunk_ids = _chunk_ids(source, chunks)
            if chunks:
                store.add_documents(chunks, ids=chunk_ids)
            manifest[source] = {"hash": content_hash, "chunk_ids": chunk_ids}
            indexed_chunks += len(chunks)
            changed_files += 1

        save_manifest(persist_dir, manifest)

        # Hot-reload, part 1/2 -- BM25: the index is a lazily-built in-memory singleton (see
        # bm25_store.py) with no awareness of when the corpus on disk changes. Rebuilding it
        # *eagerly* here (not just invalidating and leaving it to the next query to rebuild
        # lazily) means that by the time this function returns -- and callers like
        # `/api/v1/ingest`'s background task mark the job "indexed" -- BM25 has already
        # absorbed the new corpus. Without this, a query landing in the gap between
        # invalidation and the next lazy rebuild would pay the rebuild latency inline, and a
        # "indexed" status would be very slightly ahead of what BM25 could actually serve.
        #
        # Hot-reload, part 2/2 -- Chroma: no equivalent step is needed. `store` above came
        # from `get_vector_store()`'s process-wide cache keyed by `persist_dir`, and
        # `/research`'s retrieval node fetches that exact same cached client instance --
        # ingestion's `store.add_documents(...)` calls above already wrote directly into the
        # live object every query reads from, so there is no separate client to refresh. (A
        # multi-worker deployment would break this assumption -- see tasks.py's module
        # docstring -- but this project runs a single worker.)
        if changed_files or removed_sources:
            invalidate_bm25_index(source_dir)
            get_bm25_index(source_dir)
            if on_stage:
                on_stage("indexing", "Refreshing in-memory search indices...")

    return IndexResult(
        indexed_chunks=indexed_chunks,
        changed_files=changed_files,
        skipped_files=skipped_files,
        removed_files=len(removed_sources),
    )
