from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.config import get_settings
from rag_assistant.ingestion.loaders import load_documents
from rag_assistant.ingestion.manifest import hash_content, load_manifest, save_manifest
from rag_assistant.ingestion.splitter import split_documents
from rag_assistant.retrieval.vector_store import get_vector_store


@dataclass
class IndexResult:
    indexed_chunks: int
    changed_files: int
    skipped_files: int
    removed_files: int


def _chunk_ids(source: str, chunks: list[Document]) -> list[str]:
    return [f"{source}::{i}" for i in range(len(chunks))]


def build_index(
    source_dir: Path | None = None,
    persist_dir: Path | None = None,
    embeddings: Embeddings | None = None,
    incremental: bool = True,
) -> IndexResult:
    """Load the corpus, chunk it, embed it, and (re)populate the Chroma collection.

    Incremental by default: a content-hash manifest alongside the Chroma collection tracks
    what was last indexed per source file, so unchanged files are skipped, changed files have
    their old chunks deleted and replaced, and files removed from the corpus have their chunks
    deleted too -- only new/changed content pays for embedding calls. Pass `incremental=False`
    to reset the collection and manifest and rebuild everything from scratch.
    """
    settings = get_settings()
    source_dir = source_dir or settings.corpus_dir
    persist_dir = persist_dir or settings.chroma_persist_dir

    documents = load_documents(source_dir)
    store = get_vector_store(embeddings=embeddings, persist_dir=persist_dir)

    if not incremental:
        store.reset_collection()
        manifest: dict[str, dict] = {}
    else:
        manifest = load_manifest(persist_dir)

    current_sources = {doc.metadata["source"]: doc for doc in documents}

    removed_sources = set(manifest) - set(current_sources)
    for source in removed_sources:
        store.delete(ids=manifest[source]["chunk_ids"])
        del manifest[source]

    indexed_chunks = 0
    changed_files = 0
    skipped_files = 0
    for source, doc in current_sources.items():
        content_hash = hash_content(doc.page_content)
        existing = manifest.get(source)
        if existing and existing["hash"] == content_hash:
            skipped_files += 1
            continue

        if existing:
            store.delete(ids=existing["chunk_ids"])

        chunks = split_documents([doc])
        chunk_ids = _chunk_ids(source, chunks)
        if chunks:
            store.add_documents(chunks, ids=chunk_ids)
        manifest[source] = {"hash": content_hash, "chunk_ids": chunk_ids}
        indexed_chunks += len(chunks)
        changed_files += 1

    save_manifest(persist_dir, manifest)
    return IndexResult(
        indexed_chunks=indexed_chunks,
        changed_files=changed_files,
        skipped_files=skipped_files,
        removed_files=len(removed_sources),
    )
