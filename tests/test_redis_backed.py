"""The Redis-backed paths against a real Redis, not a fake.

Three things move into Redis when a deployment grows past one process -- the answer cache, the
per-tenant token budget and the ingest task registry -- and all three were covered only by a
hand-written fake. A fake agrees with whatever the code expects of it: it cannot disagree about
TTL semantics, about `setex` on an existing key, or about what a second client sees. This file
is what turns "shared across replicas" from a design statement into a tested one, by using
separate clients for the write and the read.

Skips without RAG_TEST_REDIS_URL, which is how a developer machine and a fork stay green; CI
sets it against a service container (see .github/workflows/ci.yml).
"""

import os
import time
import uuid

import pytest

from rag_assistant import budget, cache
from rag_assistant.config import get_settings
from rag_assistant.ingestion import tasks as ingest_tasks

REDIS_URL = os.environ.get("RAG_TEST_REDIS_URL")

pytestmark = pytest.mark.skipif(not REDIS_URL, reason="RAG_TEST_REDIS_URL is not set")


@pytest.fixture
def redis_env(monkeypatch):
    """Point the app at the real Redis, with caches reset so the URL actually takes effect."""
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    monkeypatch.setenv("USE_CACHE", "true")
    monkeypatch.setenv("TASK_BACKEND", "redis")
    get_settings.cache_clear()
    cache.reset_client_cache()
    budget.reset_budget_state()
    yield
    get_settings.cache_clear()
    cache.reset_client_cache()
    budget.reset_budget_state()


def test_the_answer_cache_round_trips_through_real_redis(redis_env):
    key = f"test:{uuid.uuid4()}"

    cache.cache_set(key, {"final_answer": "42", "citations": []}, ttl_seconds=60)

    assert cache.cache_get(key) == {"final_answer": "42", "citations": []}


def test_a_cached_entry_actually_expires(redis_env):
    """TTL is what keeps a stale answer from outliving a re-index; a fake can only claim it."""
    key = f"test:{uuid.uuid4()}"

    cache.cache_set(key, {"final_answer": "42"}, ttl_seconds=1)
    time.sleep(1.5)

    assert cache.cache_get(key) is None


def test_the_budget_counter_is_shared_between_clients(redis_env):
    """The point of Redis here: two replicas charging the same tenant must not each get their
    own allowance. Read back through a *separate* client, which is the part a fake cannot
    test."""
    owner = f"tenant-{uuid.uuid4()}"

    budget.charge(owner, 1000)
    cache.reset_client_cache()  # forces a new connection, standing in for another replica
    budget.reset_budget_state()

    assert budget.used_tokens(owner) == 1000


def test_an_ingest_task_is_visible_to_another_replica(redis_env):
    """TASK_BACKEND=redis exists so the replica that did not accept an upload can still answer
    a status poll about it -- the failure mode being a 404 for a task that is running fine."""
    name = f"{uuid.uuid4()}.pdf"
    task = ingest_tasks.create_task(filename=name, original_filename=name, owner="public")

    ingest_tasks.update_task(task.task_id, stage="indexing", message="Embedding...")
    # A different replica: fresh client, and no in-process registry state to fall back on.
    cache.reset_client_cache()
    ingest_tasks.reset_tasks()

    fetched = ingest_tasks.get_task(task.task_id)

    assert fetched is not None, "task was not visible outside the process that created it"
    assert fetched.stage == "indexing"
    assert fetched.message == "Embedding..."


def test_rate_limit_counters_are_shared_between_replicas(redis_env, monkeypatch):
    """The multi-replica bug, against a real Redis.

    Two `Limiter` objects built separately stand in for two replicas. What has to be true is
    that the second one sees the first one's hits -- with in-process storage it does not, and
    a cap documented as global is enforced N times over. A fake cannot check this: it would
    be the same object twice.
    """
    monkeypatch.setenv("RATE_LIMIT_STORAGE_URI", REDIS_URL)
    get_settings.cache_clear()

    from limits import parse

    from rag_assistant import api

    bucket = f"tenant-{uuid.uuid4()}"
    limit = parse("2/minute")

    replica_a = api._build_limiter(lambda request: bucket, "rag:test")
    replica_b = api._build_limiter(lambda request: bucket, "rag:test")

    assert replica_a.limiter.hit(limit, "rag:test", bucket) is True
    assert replica_b.limiter.hit(limit, "rag:test", bucket) is True
    # The third hit crosses the cap. It only fails on the second replica if the first
    # replica's two hits were visible to it.
    assert replica_b.limiter.hit(limit, "rag:test", bucket) is False, (
        "the second replica did not see the first's hits -- limiter counters are not shared"
    )


def test_the_per_caller_and_global_buckets_do_not_collide_in_shared_redis(redis_env, monkeypatch):
    """Both limiters write to one Redis under the multi-replica profile. If their keyspaces
    overlapped, one caller's requests would consume the global allowance and vice versa."""
    monkeypatch.setenv("RATE_LIMIT_STORAGE_URI", REDIS_URL)
    get_settings.cache_clear()

    from limits import parse

    from rag_assistant import api

    identity = f"caller-{uuid.uuid4()}"
    limit = parse("1/minute")

    caller = api._build_limiter(lambda request: identity, api.limiter._key_prefix)
    everyone = api._build_limiter(lambda request: identity, api.global_limiter._key_prefix)

    assert caller.limiter.hit(limit, api.limiter._key_prefix, identity) is True
    # Same identity string, different limiter: it must still have its own allowance.
    assert everyone.limiter.hit(limit, api.global_limiter._key_prefix, identity) is True


def test_readiness_sees_a_reachable_rate_limit_store(redis_env, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_STORAGE_URI", REDIS_URL)
    get_settings.cache_clear()

    from rag_assistant.readiness import check_rate_limit_storage

    assert check_rate_limit_storage() == (True, None)
