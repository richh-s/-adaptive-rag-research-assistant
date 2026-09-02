"""Backup and restore for everything that isn't in git.

Two directories hold all the state: the Chroma persist directory (vector index, ingestion
manifest, index metadata, and the conversations database) and the corpus directory (the source
documents, including tenant uploads). Losing either is unrecoverable — the corpus because
uploads exist nowhere else, the index because rebuilding it costs a full re-embed of every
document.

The subtlety is that most of that state is SQLite, and copying a live SQLite file is not a
backup. In WAL mode the committed data lives across `.db`, `.db-wal` and `.db-shm`, and a
plain `cp` of the three catches them at different instants — producing an archive that
restores, opens without complaint, and is missing or corrupting recent writes. So every SQLite
file goes through the online backup API, which produces a transactionally consistent snapshot
of a database that is being written to, and the `-wal`/`-shm` sidecars are deliberately not
copied because the snapshot has already folded them in.

Vector segment files (HNSW binaries) are copied as plain files, so the backup holds the
ingestion lock to stop this process from mutating them mid-copy. That covers ingestion, which
is the only thing that writes them.
"""

import json
import logging
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_assistant.config import get_settings
from rag_assistant.ingestion.index_metadata import load_index_metadata
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.ingestion.splitter import CHUNKING_VERSION

logger = logging.getLogger(__name__)

BACKUP_FORMAT_VERSION = 1
METADATA_FILENAME = "backup_metadata.json"
CHROMA_DIR = "chroma"
CORPUS_DIR = "corpus"

_SQLITE_SUFFIXES = {".sqlite3", ".sqlite", ".db"}
# Folded into the snapshot by the online backup API; copying them over a consistent snapshot
# would reintroduce exactly the torn state the snapshot exists to avoid.
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


@dataclass
class BackupMetadata:
    """Written into the archive so a restore can tell what it is holding before unpacking it
    over live data."""

    format_version: int
    created_at: str
    embedding_model: str | None
    embedding_dimension: int | None
    chunking_version: int
    indexed_sources: int
    corpus_files: int
    conversations: int


@dataclass
class RestoreResult:
    restored_chroma: bool
    restored_corpus: bool
    previous_kept_at: Path | None
    metadata: BackupMetadata
    embedding_model_changed: bool


def _is_sqlite(path: Path) -> bool:
    return path.suffix.lower() in _SQLITE_SUFFIXES


def _is_sqlite_sidecar(path: Path) -> bool:
    return path.name.endswith(_SQLITE_SIDECAR_SUFFIXES)


def _snapshot_sqlite(source: Path, destination: Path) -> None:
    """A transactionally consistent copy of a database that may be being written to.

    `Connection.backup()` is SQLite's online backup API: it copies pages under the database's
    own locking and restarts if a writer changes something mid-copy, so the result is a
    single point-in-time image rather than a smear across the copy's duration.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dest_conn = sqlite3.connect(destination)
    try:
        source_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        source_conn.close()


def _copy_tree_consistently(source_dir: Path, destination_dir: Path) -> None:
    """Copies a directory, snapshotting SQLite databases and skipping their sidecars."""
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if _is_sqlite_sidecar(path):
            continue
        target = destination_dir / path.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if _is_sqlite(path):
            _snapshot_sqlite(path, target)
        else:
            shutil.copy2(path, target)


def _count_conversations(persist_dir: Path) -> int:
    database = get_settings().conversations_db_path
    if not database.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            return conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        finally:
            conn.close()
    except Exception:
        # A database that predates the conversations table, or is mid-migration. Not worth
        # failing a backup over -- the count is descriptive, the data is what matters.
        return 0


def _gather_metadata(persist_dir: Path, corpus_dir: Path) -> BackupMetadata:
    index_metadata = load_index_metadata(persist_dir)
    return BackupMetadata(
        format_version=BACKUP_FORMAT_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        embedding_model=index_metadata.embedding_model if index_metadata else None,
        embedding_dimension=index_metadata.embedding_dimension if index_metadata else None,
        chunking_version=CHUNKING_VERSION,
        indexed_sources=len(load_manifest(persist_dir)),
        corpus_files=sum(1 for p in corpus_dir.rglob("*") if p.is_file())
        if corpus_dir.exists()
        else 0,
        conversations=_count_conversations(persist_dir),
    )


def create_backup(
    output_dir: Path | None = None,
    persist_dir: Path | None = None,
    corpus_dir: Path | None = None,
) -> Path:
    """Writes a timestamped `.tar.gz` holding the index, the corpus and the metadata
    describing both. Returns the archive path.

    Held under the ingestion lock so no ingest can mutate the vector segments while they are
    being copied. That covers this process; a fully quiesced backup on a busy deployment
    still wants traffic stopped, which the restore procedure in the README says plainly.
    """
    # Imported here rather than at module scope: build_index imports the retrieval stack,
    # which builds a Chroma client, and a backup should not depend on being able to do that.
    from rag_assistant.ingestion.build_index import INGEST_LOCK

    settings = get_settings()
    persist_dir = persist_dir or settings.chroma_persist_dir
    corpus_dir = corpus_dir or settings.corpus_dir
    output_dir = output_dir or Path.cwd() / "backups"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_path = output_dir / f"rag-assistant-backup-{timestamp}.tar.gz"

    with INGEST_LOCK, tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        metadata = _gather_metadata(persist_dir, corpus_dir)
        (staging / METADATA_FILENAME).write_text(
            json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n"
        )
        if persist_dir.exists():
            _copy_tree_consistently(persist_dir, staging / CHROMA_DIR)
        if corpus_dir.exists():
            _copy_tree_consistently(corpus_dir, staging / CORPUS_DIR)

        # Written to a temporary name and moved into place, so an interrupted backup never
        # leaves a truncated archive sitting in the backups directory looking usable.
        partial = archive_path.with_suffix(".partial")
        with tarfile.open(partial, "w:gz") as archive:
            for entry in sorted(staging.iterdir()):
                archive.add(entry, arcname=entry.name)
        partial.replace(archive_path)

    logger.info(
        "backup written",
        extra={
            "route": str(archive_path),
            "node": f"{metadata.indexed_sources} sources, {metadata.conversations} conversations",
        },
    )
    return archive_path


def read_backup_metadata(archive_path: Path) -> BackupMetadata:
    """Reads the archive's metadata without unpacking the rest of it."""
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.extractfile(METADATA_FILENAME)
        except KeyError as exc:
            # tarfile raises KeyError for a missing member; surface the same ValueError as
            # every other "this isn't one of ours" case so callers have one thing to catch.
            raise ValueError(
                f"{archive_path} is not a rag-assistant backup (no metadata)."
            ) from exc
        if member is None:
            raise ValueError(f"{archive_path} is not a rag-assistant backup (no metadata).")
        payload = json.loads(member.read())
    return BackupMetadata(**payload)


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extraction that refuses paths escaping the destination.

    A tar archive can name `../../etc/whatever`, and a restore runs wherever the operator
    happens to be. The archives this tool writes are safe by construction; the ones it is
    handed at restore time are whatever was on the backup disk.
    """
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Refusing to extract {member.name!r}: path escapes the archive root.")
        if member.issym() or member.islnk():
            raise ValueError(f"Refusing to extract link member {member.name!r}.")
    archive.extractall(destination)


def restore_backup(
    archive_path: Path,
    persist_dir: Path | None = None,
    corpus_dir: Path | None = None,
    keep_previous: bool = True,
) -> RestoreResult:
    """Replaces the index and corpus with the archive's contents.

    The existing directories are moved aside rather than deleted, and the swap happens only
    after the whole archive has extracted successfully -- so a corrupt or truncated archive
    fails with the live data still in place, instead of halfway through replacing it.

    Caches are process-local (the Chroma client, the BM25 index, the conversations
    connection), so a running server must be restarted afterwards to see restored data. The
    CLI says so; this function does not try to hot-swap a live process.
    """
    settings = get_settings()
    persist_dir = persist_dir or settings.chroma_persist_dir
    corpus_dir = corpus_dir or settings.corpus_dir

    metadata = read_backup_metadata(archive_path)
    if metadata.format_version > BACKUP_FORMAT_VERSION:
        raise ValueError(
            f"Backup format version {metadata.format_version} is newer than this build "
            f"understands ({BACKUP_FORMAT_VERSION}). Restore with a newer version."
        )

    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, staging)

        staged_chroma = staging / CHROMA_DIR
        staged_corpus = staging / CORPUS_DIR
        if not staged_chroma.exists() and not staged_corpus.exists():
            raise ValueError(f"{archive_path} contains neither an index nor a corpus.")

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        previous_kept_at: Path | None = None

        def swap(staged: Path, live: Path) -> bool:
            nonlocal previous_kept_at
            if not staged.exists():
                return False
            if live.exists():
                if keep_previous:
                    aside = live.with_name(f"{live.name}.pre-restore-{timestamp}")
                    live.rename(aside)
                    previous_kept_at = aside.parent
                else:
                    shutil.rmtree(live)
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), str(live))
            return True

        restored_chroma = swap(staged_chroma, persist_dir)
        restored_corpus = swap(staged_corpus, corpus_dir)

    configured_model = settings.gemini_embedding_model
    embedding_model_changed = bool(
        metadata.embedding_model and metadata.embedding_model != configured_model
    )

    logger.info(
        "restore complete",
        extra={
            "route": str(archive_path),
            "node": f"chroma={restored_chroma} corpus={restored_corpus}",
        },
    )
    return RestoreResult(
        restored_chroma=restored_chroma,
        restored_corpus=restored_corpus,
        previous_kept_at=previous_kept_at,
        metadata=metadata,
        embedding_model_changed=embedding_model_changed,
    )


def prune_backups(output_dir: Path, keep: int) -> list[Path]:
    """Deletes all but the `keep` newest archives, newest-first by filename (which is an
    ISO-ordered timestamp, so lexical order is chronological). Returns what was removed."""
    if keep <= 0:
        return []
    archives = sorted(
        output_dir.glob("rag-assistant-backup-*.tar.gz"), key=lambda p: p.name, reverse=True
    )
    removed = []
    for archive in archives[keep:]:
        archive.unlink()
        removed.append(archive)
    return removed
