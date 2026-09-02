"""Tests for the Postgres conversations backend.

Skipped unless a Postgres is reachable at `RAG_TEST_DATABASE_URL`, so a default checkout and
CI without a database service stay green. Run one locally with:

    initdb -D /tmp/ragpg/data -U postgres --auth=trust
    pg_ctl -D /tmp/ragpg/data -o "-p 55432 -k /tmp/ragpg" -l /tmp/ragpg/log start
    RAG_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/postgres uv run pytest tests/test_postgres_store.py

These assert the *same* behaviours the SQLite tests do -- ownership scoping, atomic turns,
retention -- because the two backends are interchangeable only if they actually behave the
same. A backend that merely stores rows without matching the tenancy semantics would be a
data-isolation bug that swapping a config value silently introduces.
"""

import os

import pytest

from rag_assistant.config import get_settings

DATABASE_URL = os.environ.get("RAG_TEST_DATABASE_URL", "")


def _postgres_reachable() -> bool:
    if not DATABASE_URL:
        return False
    try:
        import psycopg

        with psycopg.connect(DATABASE_URL, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(), reason="no Postgres at RAG_TEST_DATABASE_URL"
)


@pytest.fixture
def pg_store(monkeypatch):
    """A clean schema per test, so ordering can't leak rows between them."""
    import psycopg

    schema = "rag_test"
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.execute(f"CREATE SCHEMA {schema}")

    url = f"{DATABASE_URL}?options=-csearch_path%3D{schema}"
    monkeypatch.setenv("CONVERSATIONS_BACKEND", "postgres")
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()

    from rag_assistant.conversations import postgres_store, store

    postgres_store.reset_store_cache()
    yield store
    postgres_store.reset_store_cache()


# ---- migrations ----


def test_migrations_apply_and_record_themselves(pg_store):
    pg_store.create_conversation("hello")

    from rag_assistant.conversations import postgres_store

    with postgres_store._get_pool().connection() as conn:
        applied = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]

    assert applied == len(postgres_store._MIGRATIONS)


def test_migrations_are_not_reapplied(pg_store):
    pg_store.create_conversation("hello")

    from rag_assistant.conversations import postgres_store

    postgres_store.reset_store_cache()
    pg_store.create_conversation("second")

    with postgres_store._get_pool().connection() as conn:
        applied = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert applied == len(postgres_store._MIGRATIONS)
    assert len(pg_store.list_conversations()) == 2


# ---- the same semantics the SQLite backend has ----


def test_conversation_round_trip(pg_store):
    conversation = pg_store.create_conversation("Who founded Anthropic?")
    pg_store.append_turn(conversation.id, question="Who?", answer="Dario and Daniela.")

    listed = pg_store.list_conversations()
    messages = pg_store.get_messages(conversation.id)

    assert [c.id for c in listed] == [conversation.id]
    assert listed[0].message_count == 2
    assert [m.role for m in messages] == ["user", "assistant"]


def test_history_is_in_the_shape_the_graph_expects(pg_store):
    conversation = pg_store.create_conversation("t")
    pg_store.append_turn(conversation.id, question="q", answer="a")

    assert pg_store.get_history(conversation.id) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_a_summary_survives_the_json_round_trip(pg_store):
    conversation = pg_store.create_conversation("t")
    pg_store.append_turn(
        conversation.id, question="q", answer="a", report="# R", summary={"route": "vector"}
    )

    assert pg_store.get_messages(conversation.id)[1].summary == {"route": "vector"}


def test_conversations_are_scoped_per_tenant(pg_store):
    mine = pg_store.create_conversation("mine", owner="alice")
    pg_store.create_conversation("theirs", owner="bob")

    assert [c.id for c in pg_store.list_conversations(owner="alice")] == [mine.id]
    assert pg_store.get_conversation(mine.id, owner="bob") is None


def test_deleting_is_owner_scoped(pg_store):
    conversation = pg_store.create_conversation("mine", owner="alice")

    assert pg_store.delete_conversation(conversation.id, owner="bob") is False
    assert pg_store.delete_conversation(conversation.id, owner="alice") is True


def test_deleting_a_conversation_cascades_to_messages(pg_store):
    conversation = pg_store.create_conversation("t", owner="alice")
    pg_store.append_turn(conversation.id, question="q", answer="a")

    pg_store.delete_conversation(conversation.id, owner="alice")

    assert pg_store.get_messages(conversation.id) == []


# ---- retention ----


def test_retention_enforces_the_per_owner_cap(pg_store, monkeypatch):
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "0")
    monkeypatch.setenv("CONVERSATION_MAX_PER_OWNER", "2")
    get_settings.cache_clear()

    created = [pg_store.create_conversation(f"c{i}", owner="alice") for i in range(4)]
    for index, conversation in enumerate(created):
        _age(pg_store, conversation.id, offset=index)

    removed = pg_store.prune_conversations(owner="alice")

    assert removed == 2
    assert len(pg_store.list_conversations(owner="alice")) == 2


def test_retention_is_scoped_to_one_owner(pg_store, monkeypatch):
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "1")
    monkeypatch.setenv("CONVERSATION_MAX_PER_OWNER", "0")
    get_settings.cache_clear()
    mine = pg_store.create_conversation("mine", owner="alice")
    theirs = pg_store.create_conversation("theirs", owner="bob")
    _age(pg_store, mine.id, offset=-100_000)
    _age(pg_store, theirs.id, offset=-100_000)

    pg_store.prune_conversations(owner="alice")

    assert pg_store.list_conversations(owner="alice") == []
    assert [c.id for c in pg_store.list_conversations(owner="bob")] == [theirs.id]


def _age(store, conversation_id: str, offset: float) -> None:
    import time

    from rag_assistant.conversations import postgres_store

    with postgres_store._get_pool().connection() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = %s WHERE id = %s",
            (time.time() + offset, conversation_id),
        )


# ---- feedback ----


def test_feedback_round_trips_and_summarises(pg_store):
    pg_store.record_feedback("bad one", "down", owner="alice", note="wrong source")
    pg_store.record_feedback("good one", "up", owner="alice")

    summary = pg_store.feedback_summary(owner="alice")

    assert (summary["up"], summary["down"]) == (1, 1)
    assert summary["recent_downvoted_questions"] == ["bad one"]
    assert pg_store.list_feedback(owner="alice")[0].note in {"wrong source", None}


def test_feedback_is_scoped_per_tenant(pg_store):
    pg_store.record_feedback("mine", "up", owner="alice")
    pg_store.record_feedback("theirs", "up", owner="bob")

    assert [f.question for f in pg_store.list_feedback(owner="alice")] == ["mine"]


# ---- the point of the whole backend ----


def test_two_independent_connections_see_each_others_writes(pg_store):
    """The reason this backend exists. SQLite's single-writer lock is what prevents a second
    replica; here a conversation written through one pool is visible through another."""
    from rag_assistant.conversations import postgres_store

    conversation = pg_store.create_conversation("written by replica one", owner="alice")

    postgres_store.reset_store_cache()  # a second replica: fresh pool, same database

    assert pg_store.get_conversation(conversation.id, owner="alice") is not None
