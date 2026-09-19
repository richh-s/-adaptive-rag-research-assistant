"""The record of what is currently indexed, kept alongside the Chroma collection.

One entry per source file:

    "_t/alice/report.md": {
        "file_hash":        sha256 of the file's raw bytes,
        "chunk_ids":        the Chroma ids produced from it,
        "chunking_version": splitter strategy it was chunked with,
        "loader_version":   loader that parsed it,
        "owner":            tenant it belongs to
    }

`file_hash` covers the raw bytes rather than the parsed text on purpose: deciding whether a
file needs re-indexing must not require parsing it, since parsing is the expensive step
re-indexing was going to perform (a PDF parse runs pymupdf4llm and, with PDF_VISION on, a
vision API call per figure). The two version fields make a splitter or loader change a
self-applying migration -- without them an unchanged file's hash still matches, the file is
skipped, and the collection quietly keeps serving chunks built by code that no longer exists.
"""

import json
from pathlib import Path

from rag_assistant.config import get_settings

MANIFEST_FILENAME = "ingestion_manifest.json"


def _shared_backend() -> bool:
    """True when index state lives in Postgres rather than beside the persist directory.

    Tied to VECTOR_BACKEND rather than given a switch of its own: the manifest describes the
    chunk ids in the vector store, so the two have to move together or a replica decides what
    to re-index from a record of someone else's collection.
    """
    return get_settings().vector_backend == "pgvector"


def manifest_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / MANIFEST_FILENAME


def load_manifest(persist_dir: Path) -> dict[str, dict]:
    """Loads the manifest, or an empty one when nothing has been indexed yet. Entries written
    by an older version simply lack the newer fields, which makes them compare unequal and
    re-index -- the correct outcome, and the reason no explicit migration is needed here."""
    if _shared_backend():
        from rag_assistant.retrieval.pgvector_store import load_manifest_rows

        return load_manifest_rows()
    path = manifest_path(persist_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_manifest(persist_dir: Path, manifest: dict[str, dict]) -> None:
    """Writes the manifest whole. Callers rely on that: a source absent from `manifest` is a
    source no longer indexed, which is how removals are recorded.

    The file implementation writes via a temporary file and an atomic rename rather than
    truncating in place. `write_text` truncates first, so a crash or a full disk mid-write
    leaves a zero-length or half-written JSON file -- and a manifest that fails to parse is
    read as "nothing is indexed", which re-parses and re-embeds the entire corpus.
    """
    if _shared_backend():
        from rag_assistant.retrieval.pgvector_store import save_manifest_rows

        save_manifest_rows(manifest)
        return
    path = manifest_path(persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(path)
