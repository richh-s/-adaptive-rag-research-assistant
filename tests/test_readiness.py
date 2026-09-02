import httpx

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
