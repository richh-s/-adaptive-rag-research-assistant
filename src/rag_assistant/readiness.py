"""Liveness checks for the external dependencies the graph can't function without.
Kept lightweight and side-effect-free: no embedding calls, no web-search requests spent —
these run on every `/ready` poll from a load balancer/orchestrator, so cost has to stay ~0."""

from functools import lru_cache

import httpx

from rag_assistant.config import get_settings
from rag_assistant.ingestion.generations import active_index_dir, read_pointer
from rag_assistant.ingestion.index_metadata import index_embedding_model
from rag_assistant.llm import (
    EmbeddingModelUnavailable,
    get_embeddings_model,
    parse_embedding_model_name,
)
from rag_assistant.retrieval.vector_store import get_vector_store


def check_chroma() -> tuple[bool, str | None]:
    try:
        get_vector_store()._collection.count()
    except Exception as exc:
        return False, str(exc)
    return True, None


def check_embeddings() -> tuple[bool, str | None]:
    """Whether the serving index generation can embed a query.

    Every reader embeds with the model the serving generation *recorded* (see
    vector_store.index_embeddings), so a configured model that differs is not a failure --
    it is the model the next generation will be built with, and is reported as a pending
    migration. What does fail readiness is the recorded model being unusable: its provider's
    credentials are missing, or its self-hosted server has gone away. There is no fallback for
    either, because only the model that built the index can query it, so the replica should
    leave the load balancer.

    Still cheap: a metadata read, constructing a client (no request), plus -- for a local
    server -- one HEAD-weight GET of /models. No embedding call is ever spent on a readiness
    poll.
    """
    settings = get_settings()
    recorded = index_embedding_model(active_index_dir())
    serving = recorded or settings.embedding_model_name
    try:
        get_embeddings_model(recorded)
    except EmbeddingModelUnavailable as exc:
        return False, str(exc)
    provider, _ = parse_embedding_model_name(serving)
    if provider == "local":
        url = f"{settings.local_embedding_base_url.rstrip('/')}/models"
        try:
            httpx.get(
                url,
                timeout=httpx.Timeout(
                    3.0, connect=settings.local_embedding_connect_timeout_seconds
                ),
            )
        except httpx.HTTPError as exc:
            return False, f"embedding server {settings.local_embedding_base_url} unreachable: {exc}"
    if recorded and recorded != settings.embedding_model_name:
        return True, (
            f"serving index generation embeds with {recorded!r}; the configured "
            f"{settings.embedding_model_name!r} applies to the next generation -- "
            "`rag-assistant reindex build` then `reindex activate` migrates without downtime"
        )
    return True, None


def check_index_generation() -> tuple[bool, str | None]:
    """Which index generation serves, for visibility. Not part of the verdict: every
    generation that can be pointed at is complete by construction."""
    try:
        pointer = read_pointer()
    except Exception as exc:
        return False, f"index pointer unreadable: {exc}"
    return True, pointer.generation or "legacy"


def check_row_security() -> tuple[bool, str | None]:
    """Whether Postgres is actually enforcing row-level tenant isolation (pgvector only).
    Reported, not part of the verdict: the query predicate still separates tenants, but a
    deployment that believes it has two layers of isolation and has one should be told."""
    if get_settings().vector_backend != "pgvector":
        return True, "not applicable"
    try:
        from rag_assistant.retrieval.pgvector_store import row_security_status

        return row_security_status()
    except Exception as exc:
        return False, f"could not determine row security status: {exc}"


def check_web_search() -> tuple[bool, str | None]:
    try:
        response = httpx.head("https://duckduckgo.com", timeout=3.0)
        # DuckDuckGo's base domain doesn't necessarily return 2xx for a bare HEAD --
        # reachability (a response at all, not a connection error/timeout) is the actual
        # signal here.
        del response
    except httpx.HTTPError as exc:
        return False, str(exc)
    return True, None


def check_local_llm() -> tuple[bool, str | None]:
    """Reachability of the self-hosted OpenAI-compatible endpoint, when one is configured.

    Returns (True, "not configured") when LOCAL_LLM_BASE_URL is blank -- an absent local box
    is a valid deployment, not a degraded one. When it IS configured but unreachable the
    graph still answers (Anthropic/Gemini pick it up), so this is reported as a real failure
    for visibility rather than being swallowed: silently paying for Claude on every call
    because a tailnet route dropped is exactly the kind of thing you want surfaced.
    """
    settings = get_settings()
    if not settings.local_llm_base_url:
        return True, "not configured"
    try:
        httpx.get(
            f"{settings.local_llm_base_url.rstrip('/')}/models",
            timeout=httpx.Timeout(3.0, connect=settings.local_llm_connect_timeout_seconds),
        )
    except httpx.HTTPError as exc:
        return False, f"{settings.local_llm_base_url} unreachable: {exc}"
    return True, None


# Bounded rather than unbounded: the URI is configuration and changes at most once per
# process in production, while tests exercise several. A small cap keeps a test run from
# accumulating clients without making the production path miss.
@lru_cache(maxsize=8)
def _rate_limit_storage(uri: str):
    from limits.storage import storage_from_string

    return storage_from_string(uri)


def check_rate_limit_storage() -> tuple[bool, str | None]:
    """Reachability of the shared rate-limit store, when one is configured.

    Returns (True, "in-process") when the limiter keeps its counters in memory -- the
    single-container default, where there is nothing to be unreachable.

    Reported but deliberately NOT part of the ready/unavailable verdict, for the same reason
    as `check_local_llm`: the limiter falls back to per-process counters when its store is
    unreachable (see api.py's `_build_limiter`), so the service keeps answering and keeps
    limiting -- just not in concert with its peers. That is a degradation worth seeing, not
    an outage worth pulling a healthy replica out of the load balancer for. The failure it
    makes visible is the quiet one: N replicas each enforcing a private copy of a cap
    documented as global.
    """
    from rag_assistant.api import rate_limit_storage_uri

    uri = rate_limit_storage_uri()
    if uri.startswith("memory://"):
        return True, "in-process"
    try:
        # `check()` is a PING, not a write, and the storage object is cached: an orchestrator
        # polls /ready every few seconds, and building a fresh client per poll would leak a
        # connection pool each time -- a readiness probe that degrades the thing it probes.
        if _rate_limit_storage(uri).check():
            return True, None
        return False, f"rate-limit storage {uri} unreachable; limits are per-replica"
    except Exception as exc:
        return False, f"rate-limit storage {uri} unusable: {exc}"
