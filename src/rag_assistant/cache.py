"""Redis-backed cache for router decisions, Tavily results, and synthesized answers. Every
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

logger = logging.getLogger(__name__)

_KEY_PREFIX = "v1"


@lru_cache
def _get_client() -> "redis.Redis | None":
    settings = get_settings()
    if not settings.use_cache:
        return None
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def reset_client_cache() -> None:
    """Forces the next cache_get/cache_set to re-read settings and reconnect. Needed because
    `_get_client` is `lru_cache`d across the process, so tests that flip USE_CACHE/REDIS_URL
    via monkeypatch need a way to invalidate the stale cached client (see conftest.py)."""
    _get_client.cache_clear()


def cache_key(namespace: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return f"{_KEY_PREFIX}:{namespace}:{digest}"


def cache_get(key: str) -> Any | None:
    client = _get_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
    except Exception:
        logger.warning("Redis GET failed for key=%s; treating as cache miss", key, exc_info=True)
        return None
    return json.loads(raw) if raw is not None else None


def cache_set(key: str, value: Any, ttl_seconds: int) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, json.dumps(value))
    except Exception:
        logger.warning("Redis SET failed for key=%s; continuing without caching it", key, exc_info=True)
