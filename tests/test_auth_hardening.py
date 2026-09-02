"""Tests for API key scopes, expiry, per-key rate limits, and the audit trail."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api, auth
from rag_assistant.auth import (
    ALL_SCOPES,
    READ,
    WRITE,
    ApiKey,
    load_api_keys,
    rate_limit_for_identity,
    required_scope,
    resolve_key,
)
from rag_assistant.config import get_settings


def write_key_file(tmp_path, keys: list[dict]):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": keys}))
    return path


@pytest.fixture
def keyed(monkeypatch, tmp_path):
    """Installs a key file and returns a client. Rate limits are raised out of the way so
    these tests exercise auth rather than throttling."""

    def _install(keys: list[dict]):
        path = write_key_file(tmp_path, keys)
        monkeypatch.setenv("API_KEYS_FILE", str(path))
        monkeypatch.setenv("RATE_LIMIT_RPM", "10000")
        monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "10000")
        get_settings.cache_clear()
        auth.reset_api_key_cache()
        return TestClient(api.app), path

    return _install


# ---- loading ----


def test_the_simple_env_format_still_works(monkeypatch):
    """The demo path must not regress: bare `label:key` gets full scopes and no expiry."""
    monkeypatch.setenv("API_KEYS", "alice:secret-a")
    get_settings.cache_clear()
    auth.reset_api_key_cache()

    record = resolve_key("secret-a")

    assert record.owner == "alice"
    assert record.scopes == ALL_SCOPES
    assert record.expires_at is None


def test_keys_load_from_a_file(keyed):
    keyed([{"key": "sk-ro", "owner": "reporting", "scopes": ["read"]}])

    record = resolve_key("sk-ro")

    assert record.owner == "reporting"
    assert record.scopes == frozenset({READ})


def test_both_sources_combine(monkeypatch, tmp_path):
    path = write_key_file(tmp_path, [{"key": "sk-file", "owner": "from-file"}])
    monkeypatch.setenv("API_KEYS", "env-owner:sk-env")
    monkeypatch.setenv("API_KEYS_FILE", str(path))
    get_settings.cache_clear()
    auth.reset_api_key_cache()

    assert resolve_key("sk-env").owner == "env-owner"
    assert resolve_key("sk-file").owner == "from-file"


def test_an_unreadable_key_file_disables_its_keys_without_crashing(monkeypatch, tmp_path):
    """A malformed file must cost the credentials it holds, not the whole service."""
    path = tmp_path / "keys.json"
    path.write_text("{ this is not json")
    monkeypatch.setenv("API_KEYS", "alice:secret-a")
    monkeypatch.setenv("API_KEYS_FILE", str(path))
    get_settings.cache_clear()
    auth.reset_api_key_cache()

    assert resolve_key("secret-a").owner == "alice"
    assert load_api_keys()


def test_editing_the_key_file_takes_effect_without_a_restart(keyed):
    """Revocation has to be an operation, not an outage. The cache keys on the file's mtime,
    so a rewritten file is picked up on the next request."""
    _, path = keyed([{"key": "sk-live", "owner": "alice"}])
    assert resolve_key("sk-live") is not None

    import os
    import time

    path.write_text(json.dumps({"keys": []}))
    os.utime(path, (time.time() + 1, time.time() + 1))

    assert resolve_key("sk-live") is None


# ---- expiry ----


def test_an_expired_key_is_rejected(keyed):
    yesterday = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    keyed([{"key": "sk-old", "owner": "alice", "expires_at": yesterday}])

    assert resolve_key("sk-old") is None


def test_an_unexpired_key_is_accepted(keyed):
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    keyed([{"key": "sk-new", "owner": "alice", "expires_at": tomorrow}])

    assert resolve_key("sk-new").owner == "alice"


def test_a_naive_expiry_in_a_key_file_is_read_as_utc(keyed):
    """A key file moves between machines; "expires at 6pm" meaning something different per
    host is a bug waiting for a timezone change."""
    keyed([{"key": "sk-naive", "owner": "alice", "expires_at": "2020-01-01T00:00:00"}])

    assert load_api_keys()[0].expires_at.tzinfo is not None
    assert resolve_key("sk-naive") is None


def test_a_naive_expiry_set_in_code_does_not_raise():
    """Comparing a naive datetime to an aware one raises TypeError, which would turn an
    expiry check into a 500 on every authenticated request."""
    record = ApiKey(key="k", owner="o", expires_at=datetime(2020, 1, 1))

    assert record.is_expired() is True


def test_expiry_is_checked_per_request_not_at_load(keyed):
    """A key that expires while the process runs must stop working without a restart."""
    soon = datetime.now(UTC) + timedelta(seconds=1)
    keyed([{"key": "sk-soon", "owner": "alice", "expires_at": soon.isoformat()}])
    assert resolve_key("sk-soon") is not None

    record = ApiKey(key="sk-soon", owner="alice", expires_at=soon)
    assert record.is_expired(now=soon + timedelta(seconds=1)) is True


# ---- scopes ----


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/api/v1/ingest", WRITE),
        ("POST", "/api/v1/ingest/url", WRITE),
        ("DELETE", "/api/v1/conversations/abc", WRITE),
        ("POST", "/api/v1/research", READ),
        ("GET", "/api/v1/conversations", READ),
        ("GET", "/api/v1/ingest/task-id", READ),
    ],
)
def test_required_scope_per_route(method, path, expected):
    assert required_scope(method, path) == expected


def test_an_unknown_route_defaults_to_read():
    """A new endpoint should be readable by valid keys rather than silently unreachable; a
    new *write* endpoint has to be added to the rules deliberately."""
    assert required_scope("GET", "/api/v1/something-new") == READ


def test_a_read_only_key_can_research(keyed):
    client, _ = keyed([{"key": "sk-ro", "owner": "alice", "scopes": ["read"]}])

    response = client.get("/api/v1/conversations", headers={"X-API-Key": "sk-ro"})

    assert response.status_code == 200


def test_a_read_only_key_cannot_ingest(keyed):
    client, _ = keyed([{"key": "sk-ro", "owner": "alice", "scopes": ["read"]}])

    response = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": "sk-ro"},
        files={"file": ("a.md", b"content", "text/markdown")},
    )

    assert response.status_code == 403
    assert "write" in response.json()["detail"]


def test_a_write_key_can_ingest(keyed, monkeypatch, tmp_path):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setattr(api, "_run_ingest_in_background", lambda *args, **kwargs: None)
    client, _ = keyed([{"key": "sk-rw", "owner": "alice", "scopes": ["read", "write"]}])

    response = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": "sk-rw"},
        files={"file": ("a.md", b"content here", "text/markdown")},
    )

    assert response.status_code == 202


def test_insufficient_scope_is_403_not_401(keyed):
    """401 would tell a read-only client to re-authenticate, which presenting the same valid
    key again cannot fix."""
    client, _ = keyed([{"key": "sk-ro", "owner": "alice", "scopes": ["read"]}])

    unauthenticated = client.post("/api/v1/research", json={"question": "a question here"})
    forbidden = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": "sk-ro"},
        files={"file": ("a.md", b"x", "text/markdown")},
    )

    assert unauthenticated.status_code == 401
    assert forbidden.status_code == 403


# ---- per-key rate limits ----


def test_a_key_can_carry_its_own_rate_limit(keyed):
    keyed([{"key": "sk-fast", "owner": "alice", "rate_limit_rpm": 500}])
    record = resolve_key("sk-fast")

    assert rate_limit_for_identity(f"key:{record.fingerprint}") == 500


def test_a_key_without_an_override_falls_back_to_the_global_limit(keyed):
    keyed([{"key": "sk-plain", "owner": "alice"}])
    record = resolve_key("sk-plain")

    assert rate_limit_for_identity(f"key:{record.fingerprint}") is None


def test_an_anonymous_identity_has_no_override():
    assert rate_limit_for_identity("127.0.0.1") is None


def test_the_limit_lookup_never_sees_a_raw_key(keyed):
    """The bucket identity is a fingerprint; a limiter holding raw secrets would be a second
    place secrets live."""
    keyed([{"key": "sk-secret-value", "owner": "alice", "rate_limit_rpm": 42}])

    assert rate_limit_for_identity("key:sk-secret-value") is None


def test_a_per_key_limit_is_actually_enforced(monkeypatch, tmp_path):
    path = write_key_file(tmp_path, [{"key": "sk-slow", "owner": "alice", "rate_limit_rpm": 1}])
    monkeypatch.setenv("API_KEYS_FILE", str(path))
    monkeypatch.setenv("RATE_LIMIT_RPM", "10000")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "10000")
    get_settings.cache_clear()
    auth.reset_api_key_cache()
    api.limiter.reset()
    api.global_limiter.reset()
    monkeypatch.setattr(
        api._graph, "invoke", lambda *a, **k: {"research_report": "ok", "route": "none"}
    )
    client = TestClient(api.app)

    headers = {"X-API-Key": "sk-slow"}
    first = client.post("/api/v1/research", json={"question": "a question here"}, headers=headers)
    second = client.post("/api/v1/research", json={"question": "a question here"}, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429


# ---- audit ----


def test_the_fingerprint_is_not_the_key():
    record = ApiKey(key="sk-super-secret", owner="alice")

    assert record.fingerprint != "sk-super-secret"
    assert len(record.fingerprint) == 16


def test_an_accepted_request_is_audited(keyed, caplog):
    client, _ = keyed([{"key": "sk-ro", "owner": "alice", "scopes": ["read"]}])

    with caplog.at_level("INFO"):
        client.get("/api/v1/conversations", headers={"X-API-Key": "sk-ro"})

    assert any("auth accepted" in r.message for r in caplog.records)


def test_a_rejected_request_is_audited(keyed, caplog):
    client, _ = keyed([{"key": "sk-ro", "owner": "alice"}])

    with caplog.at_level("INFO"):
        client.get("/api/v1/conversations", headers={"X-API-Key": "wrong"})

    assert any("auth rejected" in r.message for r in caplog.records)


def test_the_audit_trail_never_records_the_key(keyed, caplog):
    client, _ = keyed([{"key": "sk-super-secret", "owner": "alice"}])

    with caplog.at_level("INFO"):
        client.get("/api/v1/conversations", headers={"X-API-Key": "sk-super-secret"})

    assert not any("sk-super-secret" in str(r.__dict__) for r in caplog.records)


# ---- open mode ----


def test_open_mode_is_unchanged_when_no_keys_are_configured():
    client = TestClient(api.app)

    assert client.get("/api/v1/conversations").status_code == 200
    assert client.get("/api/v1/auth/check").json()["auth_required"] is False
