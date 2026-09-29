import httpx
import pytest

from rag_assistant import readiness


def test_check_chroma_ok(monkeypatch):
    class _FakeCollection:
        def count(self):
            return 3

    class _FakeStore:
        _collection = _FakeCollection()

    monkeypatch.setattr(readiness, "get_vector_store", lambda: _FakeStore())

    ok, err = readiness.check_chroma()

    assert ok is True
    assert err is None


def test_check_chroma_failure_returns_error(monkeypatch):
    def _raise():
        raise RuntimeError("no such collection")

    class _FakeStore:
        class _collection:
            @staticmethod
            def count():
                _raise()

    monkeypatch.setattr(readiness, "get_vector_store", lambda: _FakeStore())

    ok, err = readiness.check_chroma()

    assert ok is False
    assert "no such collection" in err


def test_check_web_search_ok(monkeypatch):
    class _FakeResponse:
        status_code = 200

    monkeypatch.setattr(readiness.httpx, "head", lambda url, timeout=None: _FakeResponse())

    ok, err = readiness.check_web_search()

    assert ok is True
    assert err is None


def test_check_web_search_unreachable_returns_error(monkeypatch):
    def _raise(url, timeout=None):
        raise readiness.httpx.ConnectError("connection refused")

    monkeypatch.setattr(readiness.httpx, "head", _raise)

    ok, err = readiness.check_web_search()

    assert ok is False
    assert "connection refused" in err


def test_check_local_llm_is_ok_when_not_configured(monkeypatch):
    """No local box is a valid deployment, not a degraded one."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "")

    ok, err = readiness.check_local_llm()

    assert ok
    assert err == "not configured"


def test_check_local_llm_unreachable_reports_failure(monkeypatch):
    """The graph still answers via the Anthropic fallback, so this is reported rather than
    swallowed: silently paying for Claude because a tailnet route dropped is the failure
    mode worth seeing."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://gpu-box.example.ts.net:11434/v1")

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(readiness.httpx, "get", _boom)

    ok, err = readiness.check_local_llm()

    assert not ok
    assert "unreachable" in err


def test_check_local_llm_probes_the_models_endpoint(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://gpu-box.example.ts.net:11434/v1/")
    called = {}

    def _fake_get(url, **kwargs):
        called["url"] = url
        return None

    monkeypatch.setattr(readiness.httpx, "get", _fake_get)

    ok, err = readiness.check_local_llm()

    assert ok and err is None
    assert called["url"] == "http://gpu-box.example.ts.net:11434/v1/models"


def test_a_local_embedding_server_that_is_unreachable_is_not_ready(monkeypatch, tmp_path):
    """Unlike the local *chat* tier, embeddings have no fallback: only the model that built
    the index can query it. An unreachable server means every question fails, so the replica
    must leave the load balancer rather than merely report the problem."""
    monkeypatch.setattr(readiness, "index_embedding_model", lambda persist_dir: None)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_BASE_URL", "http://gpu-box.example.ts.net:11434/v1")

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(readiness.httpx, "get", _boom)

    ok, err = readiness.check_embeddings()

    assert not ok
    assert "unreachable" in err


def test_a_reachable_local_embedding_server_is_ready(monkeypatch):
    monkeypatch.setattr(readiness, "index_embedding_model", lambda persist_dir: None)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_BASE_URL", "http://gpu-box.example.ts.net:11434/v1/")
    called = {}
    monkeypatch.setattr(readiness.httpx, "get", lambda url, **kw: called.update(url=url))

    ok, err = readiness.check_embeddings()

    assert ok and err is None
    assert called["url"] == "http://gpu-box.example.ts.net:11434/v1/models"


def test_a_hosted_embedding_provider_makes_no_network_call(monkeypatch):
    """Gemini/OpenAI embeddings need no probe; a readiness poll must stay ~free."""
    monkeypatch.setattr(readiness, "index_embedding_model", lambda persist_dir: None)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(
        readiness.httpx, "get", lambda *a, **k: pytest.fail("probed a hosted provider")
    )

    assert readiness.check_embeddings() == (True, None)


def test_a_configured_model_that_differs_is_a_pending_migration_not_an_outage(monkeypatch):
    """Every reader embeds with the model the serving generation recorded, so changing the
    configured model cannot make the index return nonsense any more. It is the model the
    next generation will be built with, and readiness says so without pulling the replica."""
    monkeypatch.setattr(
        readiness, "index_embedding_model", lambda persist_dir: "models/older-embedding-model"
    )
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")

    ok, note = readiness.check_embeddings()

    assert ok
    assert "models/older-embedding-model" in note and "reindex" in note


def test_a_recorded_model_this_deployment_cannot_use_is_not_ready(monkeypatch):
    """The index was built with OpenAI embeddings and the key has since been removed: no
    query can be embedded, and there is no fallback model that could read this index."""
    monkeypatch.setattr(
        readiness, "index_embedding_model", lambda persist_dir: "openai/text-embedding-3-small"
    )
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    ok, err = readiness.check_embeddings()

    assert not ok
    assert "OPENAI_API_KEY" in err


def test_the_local_server_probed_is_the_one_the_index_needs(monkeypatch):
    """A generation built on the self-hosted server is probed even when the configured
    provider has moved on to a hosted one -- it is the server every query still depends on."""
    monkeypatch.setattr(readiness, "index_embedding_model", lambda persist_dir: "local/all-minilm")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("LOCAL_EMBEDDING_BASE_URL", "http://gpu-box.example.ts.net:11434/v1")

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(readiness.httpx, "get", _boom)

    ok, err = readiness.check_embeddings()

    assert not ok and "unreachable" in err
