"""Tests for the user feedback signal.

Every other metric in this system measures whether the service is working. This one measures
whether it is any good, which is the gap that a golden eval dataset silently develops: the
dataset stops resembling what people actually ask, and nothing says so.
"""

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from rag_assistant import api
from rag_assistant.conversations import store


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def sample(name: str, labels: dict[str, str]) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


# ---- store ----


def test_feedback_round_trips():
    store.record_feedback("Who founded Anthropic?", "up", route="vector", confidence_score=0.9)

    rows = store.list_feedback()

    assert len(rows) == 1
    assert rows[0].rating == "up"
    assert rows[0].route == "vector"
    assert rows[0].confidence_score == 0.9


def test_feedback_is_scoped_per_tenant():
    store.record_feedback("mine", "up", owner="alice")
    store.record_feedback("theirs", "down", owner="bob")

    assert [r.question for r in store.list_feedback(owner="alice")] == ["mine"]
    assert [r.question for r in store.list_feedback(owner="bob")] == ["theirs"]


def test_summary_counts_and_satisfaction():
    for rating in ("up", "up", "up", "down"):
        store.record_feedback("q", rating)

    summary = store.feedback_summary()

    assert (summary["up"], summary["down"], summary["total"]) == (3, 1, 4)
    assert summary["satisfaction"] == pytest.approx(0.75)


def test_summary_with_no_feedback_reports_no_satisfaction():
    """None rather than 0.0: no ratings is not the same as everyone hating it, and a
    dashboard showing 0% for a new deployment would be actively misleading."""
    summary = store.feedback_summary()

    assert summary["total"] == 0
    assert summary["satisfaction"] is None


def test_summary_surfaces_the_downvoted_questions():
    """The actually useful output -- the questions real users asked that were answered badly
    are exactly the rows a stale eval dataset is missing."""
    store.record_feedback("a bad answer question", "down")
    store.record_feedback("a good one", "up")

    summary = store.feedback_summary()

    assert summary["recent_downvoted_questions"] == ["a bad answer question"]


def test_feedback_survives_its_conversation_being_pruned(monkeypatch):
    """`conversation_id` is deliberately not a foreign key: retention deletes old
    conversations, and losing the quality signal because the transcript aged out would defeat
    the point of collecting it."""
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "0")
    monkeypatch.setenv("CONVERSATION_MAX_PER_OWNER", "1")
    conversation = store.create_conversation("first")
    store.record_feedback("q", "down", conversation_id=conversation.id)
    newer = store.create_conversation("second")
    store.append_turn(newer.id, question="q", answer="a")

    assert store.get_conversation(conversation.id) is None
    assert len(store.list_feedback()) == 1


# ---- API ----


def test_posting_feedback_records_it(client: TestClient):
    response = client.post(
        "/api/v1/feedback",
        json={"question": "Who founded Anthropic?", "rating": "down", "note": "wrong source"},
    )

    assert response.status_code == 201
    assert response.json()["recorded"] is True
    rows = store.list_feedback()
    assert rows[0].note == "wrong source"


def test_an_invalid_rating_is_rejected(client: TestClient):
    response = client.post("/api/v1/feedback", json={"question": "q", "rating": "sideways"})

    assert response.status_code == 422


def test_the_summary_endpoint_reports_aggregates(client: TestClient):
    client.post("/api/v1/feedback", json={"question": "q1", "rating": "up"})
    client.post("/api/v1/feedback", json={"question": "q2", "rating": "down"})

    summary = client.get("/api/v1/feedback/summary").json()

    assert summary["up"] == 1
    assert summary["down"] == 1
    assert summary["recent_downvoted_questions"] == ["q2"]


def test_feedback_is_not_rate_limited(client: TestClient, monkeypatch):
    """Throttling the one channel that reports the system is answering badly is the wrong
    thing to throttle."""
    monkeypatch.setenv("RATE_LIMIT_RPM", "1")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "1")
    api.limiter.reset()
    api.global_limiter.reset()

    statuses = [
        client.post("/api/v1/feedback", json={"question": f"q{i}", "rating": "down"}).status_code
        for i in range(5)
    ]

    assert statuses == [201] * 5


def test_feedback_requires_a_key_when_auth_is_enabled(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:secret-a")
    from rag_assistant import auth
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    auth.reset_api_key_cache()
    client = TestClient(api.app)

    assert (
        client.post("/api/v1/feedback", json={"question": "q", "rating": "up"}).status_code == 401
    )


def test_feedback_is_recorded_against_the_authenticated_tenant(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:secret-a")
    from rag_assistant import auth
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    auth.reset_api_key_cache()
    client = TestClient(api.app)

    client.post(
        "/api/v1/feedback",
        headers={"X-API-Key": "secret-a"},
        json={"question": "q", "rating": "up"},
    )

    assert len(store.list_feedback(owner="alice")) == 1
    assert store.list_feedback(owner="public") == []


# ---- metrics ----


def test_feedback_increments_its_metric(client: TestClient):
    before = sample("rag_feedback_total", {"rating": "down"})

    client.post("/api/v1/feedback", json={"question": "q", "rating": "down"})

    assert sample("rag_feedback_total", {"rating": "down"}) == before + 1
