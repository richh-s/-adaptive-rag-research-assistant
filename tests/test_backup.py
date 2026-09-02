"""Tests for backup and restore.

The property that matters is that a restore is *safe to attempt*: a corrupt or hostile
archive must fail with the live data still in place, and a successful restore must bring back
the real thing rather than something that merely opens without error. So these round-trip
through a real Chroma index and a real WAL-mode SQLite database, then read the data back.
"""

import sqlite3
import tarfile

import pytest

from rag_assistant.backup import (
    BACKUP_FORMAT_VERSION,
    METADATA_FILENAME,
    create_backup,
    prune_backups,
    read_backup_metadata,
    restore_backup,
)
from rag_assistant.conversations import store
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.index_metadata import load_index_metadata
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.ingestion.ownership import TENANT_DIR


@pytest.fixture
def live_state(tmp_path, monkeypatch, fake_embeddings):
    """A populated deployment: indexed corpus (public + tenant) and saved conversations."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "anthropic.md").write_text("# Anthropic\n\n## Focus\n\nConstitutional AI research.\n")
    tenant = corpus / TENANT_DIR / "alice"
    tenant.mkdir(parents=True)
    (tenant / "private.md").write_text("# Alice\n\n## Plan\n\nAlice confidential roadmap.\n")

    persist = tmp_path / "chroma"
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist))
    monkeypatch.setenv("CONVERSATIONS_DB_PATH", str(persist / "conversations.db"))
    store.reset_store_cache()

    build_index(source_dir=corpus, persist_dir=persist, embeddings=fake_embeddings)

    conversation = store.create_conversation("Who founded Anthropic?", owner="alice")
    store.append_turn(conversation.id, question="Who founded it?", answer="Dario and Daniela.")

    yield {"corpus": corpus, "persist": persist, "conversation_id": conversation.id}
    store.reset_store_cache()


# ---- backup ----


def test_backup_captures_index_corpus_and_conversations(live_state, tmp_path):
    archive = create_backup(output_dir=tmp_path / "backups")

    assert archive.exists()
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert METADATA_FILENAME in names
    assert any(n.startswith("chroma/") for n in names)
    assert any(n.startswith("corpus/") for n in names)
    # Tenant uploads exist nowhere else, so a backup that skipped them would be useless.
    assert any(f"{TENANT_DIR}/alice/private.md" in n for n in names)


def test_backup_metadata_describes_the_contents(live_state, tmp_path):
    archive = create_backup(output_dir=tmp_path / "backups")

    metadata = read_backup_metadata(archive)

    assert metadata.format_version == BACKUP_FORMAT_VERSION
    assert metadata.indexed_sources == 2
    assert metadata.corpus_files == 2
    assert metadata.conversations == 1
    assert metadata.embedding_model


def test_sqlite_sidecars_are_not_copied(live_state, tmp_path):
    """The `-wal`/`-shm` files are folded into the snapshot; copying them over it would
    reintroduce the torn state the snapshot exists to avoid."""
    archive = create_backup(output_dir=tmp_path / "backups")

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()

    assert not any(n.endswith(("-wal", "-shm", "-journal")) for n in names)


def test_a_sqlite_snapshot_is_readable_and_complete(live_state, tmp_path):
    """A plain file copy of a WAL-mode database can restore into something that opens fine
    and is missing recent writes -- so the check is that the data is actually there."""
    archive = create_backup(output_dir=tmp_path / "backups")
    extracted = tmp_path / "extracted"
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extracted)

    snapshot = extracted / "chroma" / "conversations.db"
    conn = sqlite3.connect(snapshot)
    try:
        titles = [row[0] for row in conn.execute("SELECT title FROM conversations")]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()

    assert titles == ["Who founded Anthropic?"]
    assert messages == 2


def test_no_partial_archive_is_left_behind(live_state, tmp_path):
    output = tmp_path / "backups"
    create_backup(output_dir=output)

    assert list(output.glob("*.partial")) == []


# ---- restore ----


def test_restore_round_trips_index_corpus_and_conversations(live_state, tmp_path, fake_embeddings):
    archive = create_backup(output_dir=tmp_path / "backups")
    # Simulate a bad deploy: corpus wiped, conversations gone.
    (live_state["corpus"] / "anthropic.md").unlink()
    store.reset_store_cache()

    result = restore_backup(archive)
    store.reset_store_cache()

    assert result.restored_chroma is True
    assert result.restored_corpus is True
    assert (live_state["corpus"] / "anthropic.md").exists()
    assert (live_state["corpus"] / TENANT_DIR / "alice" / "private.md").exists()
    assert len(load_manifest(live_state["persist"])) == 2
    assert load_index_metadata(live_state["persist"]) is not None
    conversations = store.list_conversations(owner="alice")
    assert [c.title for c in conversations] == ["Who founded Anthropic?"]
    assert store.get_messages(live_state["conversation_id"])[0].content == "Who founded it?"


def test_restore_keeps_the_previous_data_aside_by_default(live_state, tmp_path):
    archive = create_backup(output_dir=tmp_path / "backups")

    result = restore_backup(archive)

    assert result.previous_kept_at is not None
    kept = list(tmp_path.glob("*.pre-restore-*"))
    assert kept, "previous data should be moved aside, not deleted"


def test_restore_can_discard_the_previous_data_on_request(live_state, tmp_path):
    archive = create_backup(output_dir=tmp_path / "backups")

    result = restore_backup(archive, keep_previous=False)

    assert result.previous_kept_at is None
    assert list(tmp_path.glob("*.pre-restore-*")) == []


def test_a_corrupt_archive_fails_without_touching_live_data(live_state, tmp_path):
    """The whole point of extracting to staging first: a bad archive must not leave the
    deployment halfway replaced."""
    corrupt = tmp_path / "corrupt.tar.gz"
    corrupt.write_bytes(b"this is not a tar archive")

    with pytest.raises(Exception):
        restore_backup(corrupt)

    assert (live_state["corpus"] / "anthropic.md").exists()
    assert len(load_manifest(live_state["persist"])) == 2


def test_an_archive_without_a_metadata_file_is_rejected(live_state, tmp_path):
    bogus = tmp_path / "bogus.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("nothing useful")
    with tarfile.open(bogus, "w:gz") as tar:
        tar.add(payload, arcname="payload.txt")

    with pytest.raises(ValueError, match="not a rag-assistant backup"):
        restore_backup(bogus)

    assert (live_state["corpus"] / "anthropic.md").exists()


def test_path_traversal_in_an_archive_is_refused(live_state, tmp_path):
    """Archives handed to a restore are whatever was on the backup disk, and a tar member can
    name `../../etc/whatever`."""
    malicious = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("pwned")
    metadata = tmp_path / METADATA_FILENAME
    metadata.write_text(
        '{"format_version": 1, "created_at": "now", "embedding_model": null, '
        '"embedding_dimension": null, "chunking_version": 2, "indexed_sources": 0, '
        '"corpus_files": 0, "conversations": 0}'
    )
    with tarfile.open(malicious, "w:gz") as tar:
        tar.add(metadata, arcname=METADATA_FILENAME)
        tar.add(payload, arcname="../escaped.txt")

    with pytest.raises(ValueError, match="escapes the archive root"):
        restore_backup(malicious)

    assert not (tmp_path.parent / "escaped.txt").exists()


def test_a_newer_backup_format_is_refused(live_state, tmp_path, monkeypatch):
    archive = create_backup(output_dir=tmp_path / "backups")
    monkeypatch.setattr("rag_assistant.backup.BACKUP_FORMAT_VERSION", 0)

    with pytest.raises(ValueError, match="newer than this build"):
        restore_backup(archive)


def test_restoring_an_index_built_with_a_different_embedding_model_is_flagged(
    live_state, tmp_path, monkeypatch
):
    """Silent corruption otherwise: queries embedded into a space the stored vectors don't
    live in return plausible nonsense rather than an error."""
    archive = create_backup(output_dir=tmp_path / "backups")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "models/some-other-embedding-model")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()

    result = restore_backup(archive)

    assert result.embedding_model_changed is True


# ---- retention ----


def test_prune_keeps_only_the_newest_archives(tmp_path):
    output = tmp_path / "backups"
    output.mkdir()
    for stamp in ("20240101T000000Z", "20240102T000000Z", "20240103T000000Z"):
        (output / f"rag-assistant-backup-{stamp}.tar.gz").write_bytes(b"x")

    removed = prune_backups(output, keep=2)

    remaining = sorted(p.name for p in output.glob("*.tar.gz"))
    assert len(removed) == 1
    assert remaining == [
        "rag-assistant-backup-20240102T000000Z.tar.gz",
        "rag-assistant-backup-20240103T000000Z.tar.gz",
    ]


def test_prune_with_keep_zero_deletes_nothing(tmp_path):
    output = tmp_path / "backups"
    output.mkdir()
    (output / "rag-assistant-backup-20240101T000000Z.tar.gz").write_bytes(b"x")

    assert prune_backups(output, keep=0) == []
    assert len(list(output.glob("*.tar.gz"))) == 1
