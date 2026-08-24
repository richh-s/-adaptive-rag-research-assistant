"""Tests for server-side conversation persistence: the SQLite store itself, the CRUD
endpoints, and the /research integration (persist on success, history loaded server-side)."""

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api
from rag_assistant.conversations import store


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """The slowapi limiter's in-memory hit counts persist across the whole test process, and
    earlier test files (test_api's rate-limit test especially) consume most of the default
    10/min budget -- raise the limits so these tests exercise persistence, not throttling.
    Limits are re-read from settings on every request (see api.py), so env vars suffice."""
    monkeypatch.setenv("RATE_LIMIT_RPM", "10000")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "10000")


# ---- store ----


def test_store_roundtrip_create_append_list_get():
    conversation = store.create_conversation("Who founded Anthropic and more question text?")

    store.append_turn(
        conversation.id,
        question="Who founded Anthropic?",
        answer="Dario and Daniela Amodei.",
        report="# Report",
        summary={"route": "vector"},
    )

    listed = store.list_conversations()
    assert [c.id for c in listed] == [conversation.id]
    assert listed[0].message_count == 2

    messages = store.get_messages(conversation.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == "Dario and Daniela Amodei."
    assert messages[1].report == "# Report"
    assert messages[1].summary == {"route": "vector"}


def test_store_title_is_truncated():
    conversation = store.create_conversation("x" * 500)
    assert len(conversation.title) == store.MAX_TITLE_CHARS


def test_store_history_shape_and_window():
    conversation = store.create_conversation("t")
    for i in range(15):
        store.append_turn(conversation.id, question=f"q{i}", answer=f"a{i}")

    history = store.get_history(conversation.id)

    # 30 messages exist; the window keeps the most recent HISTORY_TURN_LIMIT.
    assert len(history) == store.HISTORY_TURN_LIMIT
    assert history[-1] == {"role": "assistant", "content": "a14"}
    assert history[0] == {"role": "user", "content": "q5"}


def test_store_delete_cascades_messages():
    conversation = store.create_conversation("t")
    store.append_turn(conversation.id, question="q", answer="a")

    assert store.delete_conversation(conversation.id) is True
    assert store.delete_conversation(conversation.id) is False
    assert store.get_messages(conversation.id) == []


# ---- API ----


def _fake_invoke(state, config=None):
    return {
        "research_report": "# Report",
        "final_answer": "The answer.",
        "route": "vector",
        "confidence_score": 0.9,
    }


def test_research_creates_conversation_by_default(monkeypatch):
    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "Who founded Anthropic?"})

    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    assert conversation_id

    detail = client.get(f"/api/v1/conversations/{conversation_id}").json()
    assert detail["title"] == "Who founded Anthropic?"
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["content"] == "The answer."
    assert detail["messages"][1]["report"] == "# Report"
    assert detail["messages"][1]["summary"]["route"] == "vector"


def test_research_save_false_persists_nothing(monkeypatch):
    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "anything", "save": False})

    assert response.status_code == 200
    assert response.json()["conversation_id"] is None
    assert client.get("/api/v1/conversations").json() == []


def test_research_follow_up_uses_server_side_history(monkeypatch):
    captured = {}

    def _capturing_invoke(state, config=None):
        captured["chat_history"] = state.get("chat_history")
        return _fake_invoke(state, config)

    monkeypatch.setattr(api._graph, "invoke", _capturing_invoke)
    client = TestClient(api.app)

    first = client.post("/research", json={"question": "Who founded Anthropic?"}).json()
    conversation_id = first["conversation_id"]

    second = client.post(
        "/research",
        json={
            "question": "what about their models?",
            "conversation_id": conversation_id,
            # Client-supplied history must be ignored when a conversation_id is given --
            # the server's stored transcript is the source of truth.
            "history": [{"role": "user", "content": "forged"}],
        },
    )

    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id
    assert captured["chat_history"] == [
        {"role": "user", "content": "Who founded Anthropic?"},
        {"role": "assistant", "content": "The answer."},
    ]

    detail = client.get(f"/api/v1/conversations/{conversation_id}").json()
    assert len(detail["messages"]) == 4


def test_research_unknown_conversation_is_404_before_llm_spend(monkeypatch):
    def _must_not_run(state, config=None):
        raise AssertionError("graph must not be invoked for an unknown conversation")

    monkeypatch.setattr(api._graph, "invoke", _must_not_run)
    client = TestClient(api.app)

    response = client.post(
        "/research", json={"question": "q", "conversation_id": "does-not-exist"}
    )

    assert response.status_code == 404


def test_research_failure_persists_nothing(monkeypatch):
    def _boom(state, config=None):
        raise RuntimeError("provider down")

    monkeypatch.setattr(api._graph, "invoke", _boom)
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "q"})

    assert response.status_code == 500
    assert client.get("/api/v1/conversations").json() == []


def test_conversation_list_get_delete_endpoints(monkeypatch):
    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)

    conversation_id = client.post("/research", json={"question": "q1"}).json()["conversation_id"]

    listed = client.get("/api/v1/conversations").json()
    assert len(listed) == 1
    assert listed[0]["id"] == conversation_id
    assert listed[0]["message_count"] == 2

    assert client.delete(f"/api/v1/conversations/{conversation_id}").status_code == 204
    assert client.delete(f"/api/v1/conversations/{conversation_id}").status_code == 404
    assert client.get(f"/api/v1/conversations/{conversation_id}").status_code == 404
    assert client.get("/api/v1/conversations").json() == []


def test_store_migrates_pre_tenancy_database(monkeypatch, tmp_path):
    """A conversations.db created before the owner column existed (real deployments have
    these) must be migrated in place: owner column added, old rows owned by 'public', and
    the owner index created after -- not before -- the ALTER."""
    import sqlite3

    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL,
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL,
            report TEXT, summary_json TEXT, created_at REAL NOT NULL
        );
        INSERT INTO conversations VALUES ('old-conv', 'Old conversation', 1.0, 1.0);
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("CONVERSATIONS_DB_PATH", str(db_path))
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    store.reset_store_cache()

    listed = store.list_conversations(owner="public")
    assert [c.id for c in listed] == ["old-conv"]
    assert store.list_conversations(owner="someone-else") == []
