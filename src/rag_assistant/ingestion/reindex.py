"""Rebuilding the index without taking it offline (see generations.py for the model).

    rag-assistant reindex build [--embedding-model openai/text-embedding-3-large]
    rag-assistant reindex activate <generation>
    rag-assistant reindex rollback
    rag-assistant reindex gc

`build` writes a complete new generation beside the serving one. By default it **re-embeds
what is already indexed** rather than re-parsing the corpus: chunk text, parent sections,
document labels and permissions are copied from the serving generation, and only the vectors
are recomputed. A model change is a change of vector space, not of text, and re-parsing would
repeat every vision call and every document-description call the corpus ever cost -- on the
real 30-report corpus that is ~370 vision transcriptions, for text that is byte-identical to
what is already stored. `--from-corpus` re-parses instead, for when the *text* must change
too (a loader or chunking change that should not wait for files to change).

Ingestion keeps writing to the serving generation throughout, so a finished build is behind
by whatever was uploaded, changed, deleted or re-permissioned while it ran. `build` ends, and
`activate` begins, with an incremental catch-up pass against the corpus on disk -- the same
fingerprint comparison every ingest makes -- so the new generation converges on exactly what
the corpus holds. The final catch-up and the pointer flip happen under the ingest lock, so no
ingest in this process can land between them.

What is not covered, stated rather than hidden: with several replicas, the ingest lock is
per-process, so an ingest on *another* replica can reach the old generation in the few
seconds before that replica notices the flip. `activate` therefore runs one more catch-up
after replicas have had INDEX_POINTER_POLL_SECONDS to converge, which picks up any file that
landed on disk in that window. A file whose ingest is still in flight after that is indexed by
the next ingest in its tenant.
"""

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from rag_assistant.config import get_settings
from rag_assistant.ingestion import generations
from rag_assistant.ingestion.build_index import INGEST_LOCK, IndexResult, build_index
from rag_assistant.ingestion.index_metadata import (
    index_embedding_model,
    load_index_metadata,
    read_embedding_dimension,
    save_index_metadata,
)
from rag_assistant.ingestion.manifest import load_manifest, save_manifest
from rag_assistant.llm import get_embeddings_model
from rag_assistant.retrieval.bm25_store import get_bm25_index, invalidate_bm25_index
from rag_assistant.retrieval.parent_store import parents_for_source, replace_parents_for_source
from rag_assistant.retrieval.vector_store import add_embedded, evict_store, get_vector_store

logger = logging.getLogger(__name__)


class ReindexError(RuntimeError):
    pass


@dataclass
class BuildResult:
    generation: str
    embedding_model: str
    mode: str
    sources: int = 0
    chunks: int = 0
    catch_up: IndexResult | None = None
    seconds: float = 0.0


@dataclass
class GenerationInfo:
    generation: str
    active: bool
    previous: bool
    embedding_model: str | None
    sources: int
    chunks: int
    updated_at: float | None = None
    notes: list[str] = field(default_factory=list)


def _progress(on_progress: Callable[[str], None] | None, message: str) -> None:
    logger.info("reindex: %s", message)
    if on_progress:
        on_progress(message)


def build_generation(
    embedding_model: str | None = None,
    from_corpus: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> BuildResult:
    """Builds a new generation and returns its id. Does not activate it."""
    started = time.monotonic()
    settings = get_settings()
    target_model = embedding_model or settings.embedding_model_name
    # Constructed first so a model this deployment has no credentials for fails before any
    # work is done, not after a half-built generation has been written.
    embeddings = get_embeddings_model(target_model)
    source_dir = generations.active_index_dir()
    generation = generations.new_generation_id()
    target_dir = generations.generation_dir(generation)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Recorded before anything is written: the store for this generation reads the model to
    # embed with from this record, and a half-built generation must still say what it is.
    save_index_metadata(
        target_dir, embedding_model=target_model, tenant_isolation=settings.tenant_isolation
    )
    target_store = get_vector_store(embeddings=embeddings, persist_dir=target_dir)
    result = BuildResult(
        generation=generation,
        embedding_model=target_model,
        mode="corpus" if from_corpus else "re-embed",
    )
    _progress(
        on_progress,
        f"building generation {generation} with {target_model} "
        f"({'re-parsing the corpus' if from_corpus else 're-embedding the serving index'})",
    )

    if not from_corpus:
        result.sources, result.chunks = _copy_reembedding(
            source_dir, target_dir, target_store, embeddings, on_progress
        )

    # The catch-up pass. For a re-embed it reconciles the copy with the corpus on disk; for a
    # corpus build it *is* the build.
    _progress(on_progress, "catching up with changes made to the corpus during the build")
    result.catch_up = build_index(persist_dir=target_dir, embeddings=embeddings)
    if from_corpus:
        result.sources = len(load_manifest(target_dir))
        result.chunks = result.catch_up.indexed_chunks
    result.seconds = round(time.monotonic() - started, 1)
    _progress(
        on_progress,
        f"generation {generation} built: {result.sources} source(s), {result.chunks} chunk(s) "
        f"in {result.seconds}s -- activate with `rag-assistant reindex activate {generation}`",
    )
    return result


def _copy_reembedding(
    source_dir: Path,
    target_dir: Path,
    target_store,
    embeddings,
    on_progress: Callable[[str], None] | None,
) -> tuple[int, int]:
    """Copies every indexed document into the target generation, recomputing only vectors.

    Batched by chunk count, so memory holds one batch of text and vectors regardless of
    corpus size, and progress is visible on a corpus that takes an hour. The manifest is
    written once at the end: a build that dies midway leaves a generation with vectors and no
    manifest, which the catch-up pass would treat as "nothing indexed" and redo -- slow, but
    never wrong, and `gc` removes the wreck.
    """
    settings = get_settings()
    source_manifest = load_manifest(source_dir)
    source_store = get_vector_store(persist_dir=source_dir)
    batch_size = max(1, settings.reindex_batch_size)

    copied: dict[str, dict] = {}
    pending_sources: list[str] = []
    pending_ids: list[str] = []
    chunks = 0

    def flush() -> None:
        nonlocal chunks
        if not pending_ids:
            pending_sources.clear()
            return
        got = source_store.get(ids=list(pending_ids), include=["documents", "metadatas"])
        by_id = {
            chunk_id: (text, metadata)
            for chunk_id, text, metadata in zip(
                got.get("ids", []), got.get("documents", []), got.get("metadatas", [])
            )
        }
        ids = [i for i in pending_ids if i in by_id]
        documents = [
            Document(page_content=by_id[i][0], metadata=dict(by_id[i][1] or {})) for i in ids
        ]
        if documents:
            vectors = embeddings.embed_documents([d.page_content for d in documents])
            add_embedded(target_store, documents, ids, vectors)
            chunks += len(documents)
        for source in pending_sources:
            entry = source_manifest[source]
            replace_parents_for_source(
                target_dir,
                source,
                entry.get("owner", "public"),
                parents_for_source(source_dir, source),
            )
            # A source whose chunks were not all found in the serving index is left out of
            # the copied manifest, so the catch-up pass re-indexes it from the file rather
            # than carrying a partial document into the new generation.
            if all(i in by_id for i in entry.get("chunk_ids", [])):
                copied[source] = dict(entry, chunk_ids=list(entry.get("chunk_ids", [])))
        pending_sources.clear()
        pending_ids.clear()
        _progress(on_progress, f"re-embedded {chunks} chunk(s) from {len(copied)} source(s)")

    for source in sorted(source_manifest):
        pending_sources.append(source)
        pending_ids.extend(source_manifest[source].get("chunk_ids", []))
        if len(pending_ids) >= batch_size:
            flush()
    flush()
    save_manifest(target_dir, copied)
    save_index_metadata(
        target_dir,
        embedding_model=index_embedding_model(target_dir) or get_settings().embedding_model_name,
        embedding_dimension=read_embedding_dimension(target_store),
    )
    return len(copied), chunks


def activate(generation: str, settle: bool = True) -> generations.Pointer:
    """Catches `generation` up one last time, makes it the serving generation, and -- with
    `settle` -- catches it up again once every replica has had time to notice."""
    generations.validate_generation(generation)
    if generation not in generations.list_generations():
        raise ReindexError(f"No such index generation: {generation!r}")
    target_dir = generations.generation_dir(generation)
    if generation != generations.LEGACY and load_index_metadata(target_dir) is None:
        raise ReindexError(
            f"Generation {generation!r} has no index metadata; it was never built. "
            "Run `rag-assistant reindex build`."
        )
    # One hold across both steps, so no ingest in this process can land in the old
    # generation after the final catch-up has run. build_index takes the (re-entrant) lock
    # itself as well.
    with INGEST_LOCK:
        build_index(persist_dir=target_dir)
        pointer = generations.set_active_generation(generation)
    # Warm the keyword index for the new generation now, so the first query after the flip
    # does not pay for building it.
    try:
        get_bm25_index(target_dir)
    except Exception:
        logger.warning("Could not pre-build the keyword index for %s", generation, exc_info=True)
    if settle:
        time.sleep(get_settings().index_pointer_poll_seconds + 1.0)
        build_index(persist_dir=target_dir)
    return pointer


def rollback() -> generations.Pointer:
    """Re-activates the generation that served before the last switch."""
    pointer = generations.read_pointer()
    if pointer.previous is None:
        raise ReindexError("No previous generation is recorded; nothing to roll back to.")
    if pointer.previous not in generations.list_generations():
        raise ReindexError(
            f"The previous generation {pointer.previous!r} has been garbage-collected."
        )
    return activate(pointer.previous)


def describe_generations() -> list[GenerationInfo]:
    pointer = generations.read_pointer()
    infos = []
    for generation in generations.list_generations():
        directory = generations.generation_dir(generation)
        metadata = load_index_metadata(directory)
        try:
            manifest = load_manifest(directory)
        except Exception:
            manifest = {}
        info = GenerationInfo(
            generation=generation,
            active=generation == pointer.generation,
            previous=generation == pointer.previous,
            embedding_model=metadata.embedding_model if metadata else None,
            sources=len(manifest),
            chunks=sum(len(e.get("chunk_ids", [])) for e in manifest.values()),
            updated_at=metadata.updated_at if metadata else None,
        )
        configured = get_settings().embedding_model_name
        if info.active and info.embedding_model and info.embedding_model != configured:
            info.notes.append(
                f"configured embedding model is {configured!r}; "
                "`rag-assistant reindex build` migrates to it"
            )
        infos.append(info)
    return infos


def garbage_collect(keep_previous: bool = True, dry_run: bool = False) -> list[str]:
    """Deletes every generation except the active one (and, by default, the previous one,
    which is what `rollback` needs). The legacy generation is never deleted: it lives in the
    index root itself, alongside the pointer and every other generation."""
    pointer = generations.read_pointer()
    protected = {pointer.generation, generations.LEGACY}
    if keep_previous and pointer.previous is not None:
        protected.add(pointer.previous)
    doomed = [g for g in generations.list_generations() if g not in protected]
    if dry_run:
        return doomed
    for generation in doomed:
        _drop(generation)
    return doomed


def _drop(generation: str) -> None:
    directory = generations.generation_dir(generation)
    settings = get_settings()
    evict_store(directory)
    invalidate_bm25_index(directory)
    if settings.vector_backend == "pgvector":
        from rag_assistant.retrieval.pgvector_store import drop_generation

        drop_generation(generation)
    elif settings.chroma_server_host:
        _drop_server_collections(generation)
    if directory.exists():
        shutil.rmtree(directory)
    logger.info("index generation %s deleted", generation)


def _drop_server_collections(generation: str) -> None:
    import chromadb

    from rag_assistant.ingestion.generations import collection_suffix
    from rag_assistant.retrieval.vector_store import COLLECTION_NAME

    settings = get_settings()
    client = chromadb.HttpClient(
        host=settings.chroma_server_host,
        port=settings.chroma_server_port,
        ssl=settings.chroma_server_ssl,
    )
    base = COLLECTION_NAME + collection_suffix(generation)
    for collection in client.list_collections():
        name = getattr(collection, "name", collection)
        if name == base or name.startswith(base + "__t_"):
            client.delete_collection(name)
