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
