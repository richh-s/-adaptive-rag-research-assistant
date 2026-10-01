"""Where the rate limiter keeps its counters.

The limiter used to be constructed with slowapi's default in-process storage and nothing
else. On one container that is correct -- one process, so its buckets are the whole
deployment's. On the multi-replica profile it silently broke the documented contract:
RATE_LIMIT_RPM_GLOBAL is described as a cap on aggregate load regardless of client, and with
N replicas each holding private buckets the real cap was N times that.

These tests pin the three things that matter: the default stays in-process, the profile that
shares every other piece of state shares this one too, and an unreachable store degrades to
per-process limiting instead of taking the API down with it.
"""

import socket

import pytest
from slowapi import Limiter

from rag_assistant import api
from rag_assistant.config import get_settings


@pytest.fixture
def unreachable_redis_url() -> str:
    """A Redis URL guaranteed to refuse connections.

    Allocated by binding an ephemeral port and closing it, rather than hardcoding one that is
    "probably free" -- this test originally picked 6399, and passed until a Redis happened to
    be listening there, at which point it quietly asserted the opposite of what it meant.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"redis://127.0.0.1:{port}/0"


@pytest.fixture
def settings_env(monkeypatch):
    """Sets env vars and clears the settings cache so they take effect."""

    def _apply(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield _apply
    get_settings.cache_clear()


def test_the_default_keeps_counters_in_process(settings_env):
    """The single-container default must not acquire an infrastructure dependency."""
    settings_env(RATE_LIMIT_STORAGE_URI="")

    assert api.rate_limit_storage_uri() == "memory://"


def test_an_explicit_memory_uri_is_the_same_as_blank(settings_env):
    settings_env(RATE_LIMIT_STORAGE_URI="memory://")

    assert api.rate_limit_storage_uri() == "memory://"


def test_the_multi_replica_profile_shares_the_limiter_counters(settings_env):
    """The bug this file exists for: the profile shared the index, the conversations and the
    task registry, and left every replica enforcing its own private copy of the global cap."""
    settings_env(
        DEPLOYMENT_PROFILE="multi-replica",
        DATABASE_URL="postgresql://localhost/rag",
        REDIS_URL="redis://redis.internal:6379/3",
    )

    assert api.rate_limit_storage_uri() == "redis://redis.internal:6379/3"


def test_an_explicit_storage_uri_survives_the_profile(settings_env):
    """Same rule as the other backends: a profile that overrode explicit configuration would
    make the individual switches lie."""
    settings_env(
        DEPLOYMENT_PROFILE="multi-replica",
        DATABASE_URL="postgresql://localhost/rag",
        REDIS_URL="redis://redis.internal:6379/3",
        RATE_LIMIT_STORAGE_URI="redis://limits.internal:6379/9",
    )

    assert api.rate_limit_storage_uri() == "redis://limits.internal:6379/9"


def test_the_two_limiters_do_not_share_a_bucket_namespace():
    """Per-caller and global counters land in one Redis under the profile; distinct prefixes
    are what keep 'this caller' and 'everyone' from being the same key."""
    assert api.limiter._key_prefix != api.global_limiter._key_prefix
    assert api.limiter._key_prefix and api.global_limiter._key_prefix


def test_a_limiter_over_shared_storage_falls_back_instead_of_failing_requests():
    """A limiter that 500s the API because its bookkeeping store is unreachable has inverted
    its own purpose. Enforcement degrades to per-process; it does not disappear, and it does
    not take the service with it."""
    built = api._build_limiter(lambda request: "k", "rag:test")

    assert built._in_memory_fallback_enabled is True
    assert built._fallback_limiter is not None


def test_the_limiter_is_built_over_the_configured_storage(settings_env, unreachable_redis_url):
    """Constructed, not merely configured -- a URI that `limits` cannot parse should surface
    here rather than on the first request after a deploy."""
    settings_env(RATE_LIMIT_STORAGE_URI=unreachable_redis_url)

    built = api._build_limiter(lambda request: "k", "rag:test")

    assert isinstance(built, Limiter)
    assert type(built._storage).__name__ == "RedisStorage"


def test_an_unreachable_store_is_reported_by_readiness_without_failing_it(
    settings_env, unreachable_redis_url
):
    """Reported, not fatal: the replica still answers and still limits, just alone."""
    from rag_assistant.readiness import check_rate_limit_storage

    settings_env(RATE_LIMIT_STORAGE_URI=unreachable_redis_url)

    ok, detail = check_rate_limit_storage()

    assert ok is False
    assert "per-replica" in detail


def test_in_process_storage_reads_as_ready(settings_env):
    from rag_assistant.readiness import check_rate_limit_storage

    settings_env(RATE_LIMIT_STORAGE_URI="")

    assert check_rate_limit_storage() == (True, "in-process")


def test_the_readiness_probe_reuses_one_client_per_uri(settings_env, unreachable_redis_url):
    """An orchestrator polls /ready every few seconds. Building a fresh Redis client per poll
    would leak a connection pool each time -- a readiness probe that degrades the thing it is
    probing."""
    from rag_assistant.readiness import _rate_limit_storage, check_rate_limit_storage

    settings_env(RATE_LIMIT_STORAGE_URI=unreachable_redis_url)

    for _ in range(5):
        check_rate_limit_storage()

    assert _rate_limit_storage(unreachable_redis_url) is _rate_limit_storage(unreachable_redis_url)
    assert _rate_limit_storage.cache_info().hits >= 4
