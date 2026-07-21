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
    assert cache.cache_key("router", "a") != cache.cache_key("tavily", "a")
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
