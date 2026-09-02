import json

from rag_assistant import cache


class _FakeRedisClient:
    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value


class _FailingRedisClient:
    def get(self, key: str) -> str | None:
        raise ConnectionError("simulated Redis outage")

    def setex(self, key: str, ttl: int, value: str) -> None:
        raise ConnectionError("simulated Redis outage")


def test_cache_key_is_stable_and_namespaced():
    key = cache.cache_key("router", "What is X?")

    assert key.startswith("v1:router:")
    assert cache.cache_key("router", "What is X?") == key


def test_cache_key_differs_by_namespace_and_parts():
    assert cache.cache_key("router", "a") != cache.cache_key("web_search", "a")
    assert cache.cache_key("router", "a") != cache.cache_key("router", "b")


def test_cache_get_returns_none_when_caching_disabled(monkeypatch):
    monkeypatch.setattr(cache, "_get_client", lambda: None)

    assert cache.cache_get("v1:router:whatever") is None


def test_cache_set_is_noop_when_caching_disabled(monkeypatch):
    monkeypatch.setattr(cache, "_get_client", lambda: None)

    cache.cache_set("v1:router:whatever", {"route": "vector"}, 300)


def test_cache_set_then_get_round_trips_through_fake_client(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(cache, "_get_client", lambda: fake_client)

    cache.cache_set("v1:router:key", {"route": "vector"}, 300)

    assert cache.cache_get("v1:router:key") == {"route": "vector"}
    assert json.loads(fake_client.store["v1:router:key"]) == {"route": "vector"}


def test_cache_get_returns_none_on_miss(monkeypatch):
    monkeypatch.setattr(cache, "_get_client", lambda: _FakeRedisClient())

    assert cache.cache_get("v1:router:missing") is None


def test_cache_get_degrades_to_none_on_redis_error(monkeypatch):
    monkeypatch.setattr(cache, "_get_client", lambda: _FailingRedisClient())

    assert cache.cache_get("v1:router:key") is None


def test_cache_set_degrades_silently_on_redis_error(monkeypatch):
    monkeypatch.setattr(cache, "_get_client", lambda: _FailingRedisClient())

    cache.cache_set("v1:router:key", {"route": "vector"}, 300)


def test_get_client_is_none_when_use_cache_false(monkeypatch):
    monkeypatch.setenv("USE_CACHE", "false")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    cache.reset_client_cache()

    assert cache._get_client() is None

    get_settings.cache_clear()
    cache.reset_client_cache()


# ---- failure reporting stays quiet after the first occurrence ----


def test_a_down_redis_is_reported_once_not_once_per_call(monkeypatch, caplog):
    """158 stack traces in one eval run is not "degrading silently", which is what this
    module's docstring promises. The first occurrence keeps its traceback so the cause stays
    diagnosable; the rest drop to debug."""
    from rag_assistant import cache

    monkeypatch.setenv("USE_CACHE", "true")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    cache.reset_client_cache()

    class Dead:
        def get(self, key):
            raise ConnectionError("connection refused")

        def setex(self, *args):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(cache, "_get_client", lambda: Dead())

    with caplog.at_level("WARNING"):
        for _ in range(10):
            cache.cache_get(cache.cache_key("router", "q"))

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1


def test_distinct_failure_kinds_are_each_reported_once(monkeypatch, caplog):
    from rag_assistant import cache

    monkeypatch.setenv("USE_CACHE", "true")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    cache.reset_client_cache()

    class Dead:
        def get(self, key):
            raise ConnectionError("nope")

        def setex(self, *args):
            raise ConnectionError("nope")

    monkeypatch.setattr(cache, "_get_client", lambda: Dead())

    with caplog.at_level("WARNING"):
        cache.cache_get(cache.cache_key("router", "q"))
        cache.cache_set(cache.cache_key("router", "q"), {"a": 1}, 60)

    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 2


def test_a_down_redis_still_returns_a_miss_rather_than_raising(monkeypatch):
    """The behaviour the quieting must not change."""
    from rag_assistant import cache

    monkeypatch.setenv("USE_CACHE", "true")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    cache.reset_client_cache()

    class Dead:
        def get(self, key):
            raise ConnectionError("nope")

    monkeypatch.setattr(cache, "_get_client", lambda: Dead())

    assert cache.cache_get(cache.cache_key("router", "q")) is None
