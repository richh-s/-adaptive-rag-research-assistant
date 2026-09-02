"""Tests for the conversation store's schema migrations and retention policy.

These cover the two things that only go wrong on a *deployed* database with existing data --
so they're the two the rest of the suite, which starts from an empty tmp file every time,
structurally cannot catch.
"""

import sqlite3
import time

import pytest

from rag_assistant.conversations import store


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "conversations.db"
    monkeypatch.setenv("CONVERSATIONS_DB_PATH", str(path))
    store.reset_store_cache()
    yield path
    store.reset_store_cache()


def user_version(path) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


# ---- migrations ----


def test_fresh_database_is_stamped_at_the_latest_version(db_path):
    store.create_conversation("hello")

    assert user_version(db_path) == len(store._MIGRATIONS)


def test_migrations_are_not_reapplied_on_reconnect(db_path):
    conversation = store.create_conversation("hello")
    store.append_turn(conversation.id, question="q", answer="a")
    store.reset_store_cache()

    # Reopening must be a no-op, not a re-run that could clobber existing rows.
    assert [c.id for c in store.list_conversations()] == [conversation.id]
    assert user_version(db_path) == len(store._MIGRATIONS)


def test_legacy_pre_multi_tenancy_database_is_migrated_in_place(db_path):
    """The shape that shipped before the `owner` column existed: rows are real user data on a
    mounted volume, so the migration has to add the column and adopt those rows into the
    `public` tenant -- which is exactly who could read them before tenancy existed."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            report TEXT,
            summary_json TEXT,
            created_at REAL NOT NULL
        );
        """
    )
    now = time.time()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("legacy1", "An old conversation", now, now),
    )
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
        ("legacy1", "old question", now),
    )
    conn.commit()
    conn.close()
    assert user_version(db_path) == 0

    listed = store.list_conversations(owner="public")

    assert [c.id for c in listed] == ["legacy1"]
    assert listed[0].message_count == 1
    assert user_version(db_path) == len(store._MIGRATIONS)


def test_database_already_carrying_owner_migrates_without_duplicating_the_column(db_path):
    """The third pre-versioning shape: `owner` was already added by the old ad-hoc ALTER that
    ran on every connect. Re-running that ALTER would raise `duplicate column name`."""
    store.create_conversation("created by current code")
    # Rewind the stamp to simulate a database written before versioning existed.
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()
    store.reset_store_cache()

    listed = store.list_conversations()

    assert len(listed) == 1
    assert user_version(db_path) == len(store._MIGRATIONS)


def test_a_newly_appended_migration_runs_on_an_existing_database(db_path, monkeypatch):
    """Proves the mechanism actually works going forward, rather than just stamping 1 once."""
    store.create_conversation("hello")
    store.reset_store_cache()
    applied: list[str] = []

    def _migration_appended_example(conn: sqlite3.Connection) -> None:
        applied.append("appended")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_example ON conversations(created_at)")

    expected_version = len(store._MIGRATIONS) + 1
    monkeypatch.setattr(store, "_MIGRATIONS", [*store._MIGRATIONS, _migration_appended_example])

    assert len(store.list_conversations()) == 1
    assert applied == ["appended"]
    assert user_version(db_path) == expected_version

    # And it is not applied a second time on the next connect.
    store.reset_store_cache()
    assert len(store.list_conversations()) == 1
    assert applied == ["appended"]


# ---- retention ----


def _age_conversation(conversation_id: str, days: int) -> None:
    """Backdates a conversation past the retention window without waiting for real time."""
    with store._LOCK:
        conn = store._get_conn()
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (time.time() - days * 86400, conversation_id),
        )
        conn.commit()


def test_retention_deletes_conversations_past_the_age_cutoff(db_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "30")
    monkeypatch.setenv("CONVERSATION_MAX_PER_OWNER", "0")

    old = store.create_conversation("ancient")
    recent = store.create_conversation("recent")
    _age_conversation(old.id, days=45)

    removed = store.prune_conversations()

    assert removed == 1
    assert [c.id for c in store.list_conversations()] == [recent.id]


def test_retention_cascades_to_messages(db_path, monkeypatch):
    """Orphaned message rows would be the whole point missed -- the transcript is the bulk of
    the data. This works only because `_get_conn` enables `PRAGMA foreign_keys`."""
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "30")

    conversation = store.create_conversation("ancient")
    store.append_turn(conversation.id, question="q", answer="a")
    assert len(store.get_messages(conversation.id)) == 2
    _age_conversation(conversation.id, days=45)

    store.prune_conversations()

    assert store.get_messages(conversation.id) == []


def test_retention_enforces_the_per_owner_count_cap_keeping_the_newest(db_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "0")
    monkeypatch.setenv("CONVERSATION_MAX_PER_OWNER", "3")

    created = [store.create_conversation(f"conversation {i}") for i in range(5)]
    # list_conversations orders by updated_at DESC; creation order is ascending in time, so
    # the last three created are the ones that must survive.
    for index, conversation in enumerate(created):
        _age_conversation(conversation.id, days=-index)

    removed = store.prune_conversations()

    assert removed == 2
    assert {c.id for c in store.list_conversations()} == {c.id for c in created[2:]}


def test_retention_is_scoped_to_one_owner(db_path, monkeypatch):
    """Pruning tenant A must never touch tenant B's rows."""
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "30")

    mine = store.create_conversation("mine", owner="tenant-a")
    theirs = store.create_conversation("theirs", owner="tenant-b")
    _age_conversation(mine.id, days=45)
    _age_conversation(theirs.id, days=45)

    store.prune_conversations(owner="tenant-a")

    assert store.list_conversations(owner="tenant-a") == []
    assert [c.id for c in store.list_conversations(owner="tenant-b")] == [theirs.id]


def test_retention_disabled_by_zero_keeps_everything(db_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "0")
    monkeypatch.setenv("CONVERSATION_MAX_PER_OWNER", "0")

    conversation = store.create_conversation("ancient")
    _age_conversation(conversation.id, days=10_000)

    assert store.prune_conversations() == 0
    assert len(store.list_conversations()) == 1


def test_append_turn_prunes_inline(db_path, monkeypatch):
    """Retention runs on write, so a long-lived deployment stays bounded without a cron."""
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "0")
    monkeypatch.setenv("CONVERSATION_MAX_PER_OWNER", "2")

    created = [store.create_conversation(f"conversation {i}") for i in range(3)]
    for index, conversation in enumerate(created):
        _age_conversation(conversation.id, days=-index)

    store.append_turn(created[-1].id, question="q", answer="a")

    assert {c.id for c in store.list_conversations()} == {c.id for c in created[1:]}


def test_append_turn_survives_a_failing_retention_pass(db_path, monkeypatch):
    """The turn is committed before pruning, so a retention bug must not lose the user's
    answer -- it should be logged and swallowed."""
    conversation = store.create_conversation("hello")

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(store, "_prune_locked", _boom)

    store.append_turn(conversation.id, question="q", answer="a")

    assert [m.content for m in store.get_messages(conversation.id)] == ["q", "a"]
