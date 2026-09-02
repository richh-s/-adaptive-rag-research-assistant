"""Tests for the opt-in multi-replica backends.

Defaults are unchanged and must stay that way: the whole design is that a single-container
deployment needs no infrastructure, and the scaling backends are configuration rather than a
rewrite. So these check both that the switches work and that they are off unless asked for.

Redis is faked rather than run. What's worth testing here is the store's own logic -- state
crossing process boundaries, the degradation path when Redis is down -- not that redis-py can
talk to Redis.
"""

import json

import pytest

from rag_assistant.config import get_settings
from rag_assistant.ingestion import tasks
from rag_assistant.retrieval.vector_store import get_vector_store


class FakeRedis:
    """Enough of the redis client for the task registry: setex/get/ping."""

    def __init__(self, *, alive: bool = True):
        self.store: dict[str, str] = {}
        self.alive = alive
        self.ttls: dict[str, int] = {}

    def ping(self):
        if not self.alive:
            raise ConnectionError("redis down")
        return True

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value
        self.ttls[key] = ttl

    def get(self, key: str) -> str | None:
        return self.store.get(key)


@pytest.fixture
def redis_tasks(monkeypatch):
    """Switches the task registry to a fake Redis."""

    def _install(client=None):
        client = client if client is not None else FakeRedis()
        monkeypatch.setenv("TASK_BACKEND", "redis")
        get_settings.cache_clear()
        monkeypatch.setattr(tasks, "get_redis_client", lambda: client)
        tasks.reset_tasks()
        return client

    return _install


# ---- defaults ----


def test_the_task_backend_defaults_to_memory():
    assert get_settings().task_backend == "memory"


def test_chroma_defaults_to_embedded_mode():
    assert get_settings().chroma_server_host == ""


def test_in_memory_tasks_still_work():
    task = tasks.create_task("a.md", "a.md")

    tasks.update_task(task.task_id, stage="indexing", message="working")

    assert tasks.get_task(task.task_id).stage == "indexing"


# ---- redis-backed tasks ----


def test_a_task_is_written_to_redis(redis_tasks):
    client = redis_tasks()

    task = tasks.create_task("a.md", "original.md")

    assert any(task.task_id in key for key in client.store)
    stored = json.loads(next(iter(client.store.values())))
    assert stored["original_filename"] == "original.md"


def test_a_task_written_by_one_replica_is_visible_to_another(redis_tasks):
    """The entire point: with in-memory tasks, a client polling a load-balanced deployment
    gets 404s from every worker that didn't accept the upload."""
    redis_tasks()
    task = tasks.create_task("a.md", "a.md")
    tasks.update_task(task.task_id, stage="indexed", message="done", indexed_chunks=7)

    # Simulate a second replica: no in-memory state at all, same Redis.
    tasks.reset_tasks()

    fetched = tasks.get_task(task.task_id)
    assert fetched is not None
    assert fetched.stage == "indexed"
    assert fetched.indexed_chunks == 7


def test_redis_tasks_carry_a_ttl(redis_tasks):
    """An ingest task is only interesting while someone is polling it; a TTL bounds the keys
    without anything having to run a sweep."""
    client = redis_tasks()

    task = tasks.create_task("a.md", "a.md")

    assert client.ttls[f"v1:ingest-task:{task.task_id}"] > 0


def test_updating_an_unknown_redis_task_is_a_no_op(redis_tasks):
    redis_tasks()

    tasks.update_task("does-not-exist", stage="indexed")

    assert tasks.get_task("does-not-exist") is None


def test_an_unreachable_redis_falls_back_to_memory_rather_than_failing_the_upload(monkeypatch):
    """Degrading to per-process task state costs visibility on one job; refusing to ingest
    costs the upload."""
    monkeypatch.setenv("TASK_BACKEND", "redis")
    get_settings.cache_clear()
    monkeypatch.setattr(tasks, "get_redis_client", lambda: None)
    tasks.reset_tasks()

    task = tasks.create_task("a.md", "a.md")

    assert tasks.get_task(task.task_id) is not None


def test_a_dead_redis_client_is_reported_as_unavailable():
    """`from_url` is lazy, so a client object exists happily against a dead server -- the
    PING is what turns that into a usable signal."""
    from rag_assistant import cache

    cache.reset_client_cache()
    dead = FakeRedis(alive=False)
    original = cache._get_connection
    try:
        cache._get_connection = lambda: dead
        assert cache.get_redis_client() is None
    finally:
        cache._get_connection = original
        cache.reset_client_cache()


def test_the_redis_client_is_not_gated_on_use_cache(monkeypatch):
    """USE_CACHE governs caching. The task registry stores state, and must not be switched
    off by a flag about caching."""
    monkeypatch.setenv("USE_CACHE", "false")
    get_settings.cache_clear()
    from rag_assistant import cache

    cache.reset_client_cache()

    # The caching client is disabled...
    assert cache._get_client() is None
    # ...while the connection accessor still constructs one (it may be unreachable here,
    # which is a different question from being switched off).
    assert cache._get_connection() is not None


# ---- chroma server mode ----


def test_server_mode_builds_an_http_client(monkeypatch, fake_embeddings):
    monkeypatch.setenv("CHROMA_SERVER_HOST", "chroma.internal")
    monkeypatch.setenv("CHROMA_SERVER_PORT", "9000")
    get_settings.cache_clear()

    captured = {}

    class FakeHttpClient:
        def __init__(self, host, port, ssl):
            captured.update(host=host, port=port, ssl=ssl)
            raise RuntimeError("stop here -- construction args are what matter")

    import chromadb

    monkeypatch.setattr(chromadb, "HttpClient", FakeHttpClient)

    with pytest.raises(RuntimeError):
        get_vector_store(embeddings=fake_embeddings)

    assert captured == {"host": "chroma.internal", "port": 9000, "ssl": False}


def test_an_explicit_persist_dir_still_uses_embedded_mode(monkeypatch, tmp_path, fake_embeddings):
    """Tests and tooling pass explicit persist directories; configuring a server must not
    silently redirect them."""
    monkeypatch.setenv("CHROMA_SERVER_HOST", "chroma.internal")
    get_settings.cache_clear()

    store = get_vector_store(embeddings=fake_embeddings, persist_dir=tmp_path / "chroma")

    assert store is not None
