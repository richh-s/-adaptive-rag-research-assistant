import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.config import get_settings
from rag_assistant.ingestion.loaders import LOADER_VERSION, iter_corpus_files, load_corpus_file
from rag_assistant.ingestion.index_metadata import read_embedding_dimension, save_index_metadata
from rag_assistant.ingestion.manifest import load_manifest, save_manifest
from rag_assistant.ingestion.splitter import CHUNKING_VERSION, split_with_parents
from rag_assistant.ingestion.ownership import owner_of_relative_path
from rag_assistant.retrieval.bm25_store import apply_bm25_delta, get_bm25_index
from rag_assistant.retrieval.parent_store import (
    delete_parents_for_source,
    replace_parents_for_source,
)
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
    # How many files were actually parsed. Distinct from `changed_files` only when a parse
    # produced no indexable content -- it exists so the expensive step is measurable, since
    # "skipped 40 files" used to be true of embedding while every one of them was still
    # parsed (and, for PDFs with vision on, paid for) first.
    parsed_files: int = 0


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
    owner: str | None = None,
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

    if owner is not None and not incremental:
        # A full rebuild resets the whole collection, which would delete every other
        # tenant's chunks while only re-indexing this one's. Refuse rather than corrupt.
        raise ValueError(
            "A non-incremental rebuild cannot be scoped to one owner -- it resets the "
            "entire collection. Run build_index(incremental=False) without an owner."
        )

    with INGEST_LOCK:
        if on_stage:
            on_stage("parsing", "Scanning corpus for changes...")
        # Enumerate and fingerprint first, parse second. Fingerprinting reads raw bytes;
        # parsing runs pymupdf4llm and, with PDF_VISION on, a vision API call per figure.
        # Deciding what to re-index from the cheap signal is the entire point -- it makes an
        # upload cost the changed files rather than the whole corpus, twice over.
        corpus_files = iter_corpus_files(source_dir, owner=owner)
        store = get_vector_store(embeddings=embeddings, persist_dir=persist_dir)

        if not incremental:
            store.reset_collection()
            manifest: dict[str, dict] = {}
        else:
            manifest = load_manifest(persist_dir)

        files_by_source = {f.source: f for f in corpus_files}

        # Removal detection is scoped to the same slice that was scanned. Comparing a
        # single tenant's scan against the whole manifest would read every other tenant's
        # documents as deleted and drop their chunks.
        tracked_sources = {
            source
            for source in manifest
            if owner is None or owner_of_relative_path(Path(source)) == owner
        }
        # Collected so the keyword index can be updated for exactly these chunks instead of
        # rebuilt over the whole collection.
        added_chunk_ids: list[str] = []
        removed_chunk_ids: list[str] = []

        removed_sources = tracked_sources - set(files_by_source)
        for source in removed_sources:
            removed_chunk_ids.extend(manifest[source]["chunk_ids"])
            store.delete(ids=manifest[source]["chunk_ids"])
            delete_parents_for_source(persist_dir, source)
            del manifest[source]

        if on_stage:
            on_stage("indexing", "Embedding and indexing changed files...")

        indexed_chunks = 0
        changed_files = 0
        skipped_files = 0
        parsed_files = 0
        for source, corpus_file in files_by_source.items():
            existing = manifest.get(source)
            # Three independent reasons to re-index, all checkable without parsing: the bytes
            # changed, the chunking strategy changed, or a loader changed. The version checks
            # are what make those changes self-applying migrations -- without them an
            # unchanged file's fingerprint still matches, the file is skipped, and the
            # collection keeps serving chunks built by code that no longer exists. Silent,
            # and visible only as quietly worse retrieval.
            if (
                existing
                and existing.get("file_hash") == corpus_file.fingerprint
                and existing.get("chunking_version") == CHUNKING_VERSION
                and existing.get("loader_version") == LOADER_VERSION
            ):
                skipped_files += 1
                continue

            docs = load_corpus_file(corpus_file)
            parsed_files += 1

            if existing:
                removed_chunk_ids.extend(existing["chunk_ids"])
                store.delete(ids=existing["chunk_ids"])

            split = split_with_parents(docs, embeddings=embeddings or store.embeddings)
            chunks = split.chunks
            # Stored as an epoch float rather than a string: Chroma's `$gte`/`$lte` operators
            # compare numbers, and lexicographic date strings would only work by accident of
            # ISO formatting.
            indexed_at = time.time()
            for chunk in chunks:
                chunk.metadata["ingested_at"] = indexed_at
            chunk_ids = _chunk_ids(source, chunks)
            if chunks:
                store.add_documents(chunks, ids=chunk_ids)
                added_chunk_ids.extend(chunk_ids)
            # Written even when PARENT_CONTEXT is off, so enabling it later doesn't require a
            # re-index -- the sections are cheap to store and useless to reconstruct after
            # the fact without re-parsing.
            replace_parents_for_source(persist_dir, source, corpus_file.owner, split.parents)
            manifest[source] = {
                "file_hash": corpus_file.fingerprint,
                "chunk_ids": chunk_ids,
                "chunking_version": CHUNKING_VERSION,
                "loader_version": LOADER_VERSION,
                # Recorded for the router's corpus description, which must list only what the
                # asking tenant can actually retrieve. Chunk metadata carries the owner too
                # (that is what filters retrieval); this copy just saves reading Chroma to
                # answer "what is in this corpus for me".
                "owner": corpus_file.owner,
            }
            indexed_chunks += len(chunks)
            changed_files += 1

        save_manifest(persist_dir, manifest)
        # Recorded on every run, not only when something changed: the point is to describe
        # what the collection currently holds, and a run that changed nothing still confirms
        # the configured model matches what is stored.
        save_index_metadata(
            persist_dir,
            embedding_model=get_settings().gemini_embedding_model,
            embedding_dimension=read_embedding_dimension(store),
        )

        # Hot-reload, part 1/2 -- BM25: the index is a lazily-built in-memory singleton (see
        # bm25_store.py) with no awareness of when the collection changes. Rebuilding it
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
            # Apply just the delta. `apply_bm25_delta` reports False when no index is cached
            # yet, in which case building one now keeps the eager-refresh guarantee: by the
            # time this returns, keyword search already reflects the new corpus, so a query
            # landing immediately after an ingest doesn't pay a rebuild inline.
            if not apply_bm25_delta(
                persist_dir, added_ids=added_chunk_ids, removed_ids=removed_chunk_ids
            ):
                get_bm25_index(persist_dir)
            if on_stage:
                on_stage("indexing", "Refreshing in-memory search indices...")

    return IndexResult(
        indexed_chunks=indexed_chunks,
        changed_files=changed_files,
        skipped_files=skipped_files,
        removed_files=len(removed_sources),
        parsed_files=parsed_files,
    )
