"""Redis-backed cache for router decisions, web search results, and synthesized answers. Every
key is namespaced under a `v1:` prefix so an incompatible value-shape change can be rolled out
by bumping the prefix instead of migrating or invalidating existing entries. Values are JSON.

Caching is best-effort: `USE_CACHE=false` (set for tests -- see conftest.py) or any Redis
error at read/write time both degrade silently to "no cache" rather than crashing the graph --
the graph already worked with zero caching before this existed, so a cache outage should never
be worse than that.
"""

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

import redis

from rag_assistant.config import get_settings
from rag_assistant.metrics import record_cache

logger = logging.getLogger(__name__)

_KEY_PREFIX = "v1"

# Redis being down is one condition, not one condition per operation. Reporting it on every
# call turned a graceful degradation into 158 stack traces in a single eval run, which buries
# the output the user actually asked for -- and contradicts this module's own promise to
# degrade quietly. The first occurrence carries the traceback so the cause is diagnosable;
# after that the same failure logs once at debug level.
_reported_failures: set[str] = set()


def _report_once(kind: str, message: str, *args) -> None:
    if kind in _reported_failures:
        logger.debug(message, *args)
        return
    _reported_failures.add(kind)
    logger.warning(
        "%s (further occurrences at debug level)",
        message % args if args else message,
        exc_info=True,
    )


def reset_failure_reporting() -> None:
    """Lets tests assert on the first-occurrence behaviour independently."""
    _reported_failures.clear()


@lru_cache
def _get_client() -> "redis.Redis | None":
    settings = get_settings()
    if not settings.use_cache:
        return None
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def _get_connection() -> "redis.Redis | None":
    """A Redis client independent of USE_CACHE.

    USE_CACHE governs whether *caching* is on. Other Redis users -- the ingest task registry
    under TASK_BACKEND=redis -- are storing state, not caching it, and must not be switched
    off by a flag about caching. Sharing one connection keeps it to a single pool.
    """
    try:
        return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
    except Exception:
        _report_once("construct", "Could not construct a Redis client")
        return None


def get_redis_client() -> "redis.Redis | None":
    """A live Redis connection, or None when it cannot be reached.

    Connectivity is checked with a PING rather than assumed: `from_url` is lazy, so a client
    object exists happily against a dead server and every caller would discover that
    separately, mid-operation.
    """
    client = _get_connection()
    if client is None:
        return None
    try:
        client.ping()
    except Exception:
        _report_once("unreachable", "Redis is configured but unreachable")
        return None
    return client


def reset_client_cache() -> None:
    """Forces the next cache_get/cache_set to re-read settings and reconnect. Needed because
    `_get_client` is `lru_cache`d across the process, so tests that flip USE_CACHE/REDIS_URL
    via monkeypatch need a way to invalidate the stale cached client (see conftest.py)."""
    _get_client.cache_clear()
    _get_connection.cache_clear()
    _reported_failures.clear()


def cache_key(namespace: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return f"{_KEY_PREFIX}:{namespace}:{digest}"


def _namespace_of(key: str) -> str:
    """The namespace segment of a `v1:<namespace>:<digest>` key, for metric labelling. The
    digest is deliberately excluded -- one series per cached question would be unbounded."""
    parts = key.split(":")
    return parts[1] if len(parts) > 2 else "unknown"


def cache_get(key: str) -> Any | None:
    client = _get_client()
    if client is None:
        record_cache(_namespace_of(key), "disabled")
        return None
    try:
        raw = client.get(key)
    except Exception:
        _report_once("get", "Redis GET failed; treating as cache miss")
        record_cache(_namespace_of(key), "error")
        return None
    record_cache(_namespace_of(key), "hit" if raw is not None else "miss")
    return json.loads(raw) if raw is not None else None


def cache_set(key: str, value: Any, ttl_seconds: int) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, json.dumps(value))
    except Exception:
        _report_once("set", "Redis SET failed; continuing without caching it")
