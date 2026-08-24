"""Tests for API-key auth and tenant scoping. API_KEYS is blank by default in tests, so
every other test file exercises open/demo mode implicitly; these tests turn auth on."""

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api
from rag_assistant.auth import parse_api_keys, resolve_owner


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    # Same rationale as test_conversations.py: the limiter's in-memory hit counts persist
    # across the whole test process.
    monkeypatch.setenv("RATE_LIMIT_RPM", "10000")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "10000")


def _fake_invoke(state, config=None):
    return {
        "research_report": "# Report",
        "final_answer": "The answer.",
        "route": "vector",
        "confidence_score": 0.9,
    }


def test_parse_api_keys_labels_and_bare_keys():
    keys = parse_api_keys("alice:secret-a, secret-b ,, bob: ")
    assert keys["secret-a"] == "alice"
    assert keys["secret-b"].startswith("key-")
    assert len(keys) == 2  # "bob:" has no key and is dropped


def test_resolve_owner_open_mode_and_enabled_mode(monkeypatch):
    monkeypatch.setenv("API_KEYS", "")
    assert resolve_owner(None) == "public"
    assert resolve_owner("anything") == "public"

    monkeypatch.setenv("API_KEYS", "alice:secret-a")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    assert resolve_owner("secret-a") == "alice"
    assert resolve_owner("wrong") is None
    assert resolve_owner(None) is None


def test_protected_endpoints_reject_without_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:secret-a")
    client = TestClient(api.app)

    assert client.post("/research", json={"question": "q"}).status_code == 401
    assert client.get("/api/v1/conversations").status_code == 401
    assert client.get("/api/v1/auth/check").status_code == 401
    # Liveness and the frontend stay open.
    assert client.get("/health").status_code == 200


def test_valid_key_via_header_and_bearer(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:secret-a")
    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)

    checked = client.get("/api/v1/auth/check", headers={"X-API-Key": "secret-a"})
    assert checked.status_code == 200
    assert checked.json() == {"ok": True, "auth_required": True, "owner": "alice"}

    bearer = client.get("/api/v1/auth/check", headers={"Authorization": "Bearer secret-a"})
    assert bearer.status_code == 200


def test_auth_check_reports_open_mode():
    client = TestClient(api.app)
    body = client.get("/api/v1/auth/check").json()
    assert body == {"ok": True, "auth_required": False, "owner": "public"}


def test_conversations_are_scoped_per_tenant(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:secret-a,bob:secret-b")
    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)
    alice = {"X-API-Key": "secret-a"}
    bob = {"X-API-Key": "secret-b"}

    conversation_id = client.post(
        "/research", json={"question": "alice's question"}, headers=alice
    ).json()["conversation_id"]

    # Alice sees her conversation; Bob sees nothing and can't read, continue, or delete hers.
    assert [c["id"] for c in client.get("/api/v1/conversations", headers=alice).json()] == [
        conversation_id
    ]
    assert client.get("/api/v1/conversations", headers=bob).json() == []
    assert client.get(f"/api/v1/conversations/{conversation_id}", headers=bob).status_code == 404
    assert (
        client.post(
            "/research",
            json={"question": "follow-up", "conversation_id": conversation_id},
            headers=bob,
        ).status_code
        == 404
    )
    assert (
        client.delete(f"/api/v1/conversations/{conversation_id}", headers=bob).status_code == 404
    )
    # Alice still can.
    assert (
        client.delete(f"/api/v1/conversations/{conversation_id}", headers=alice).status_code
        == 204
    )
