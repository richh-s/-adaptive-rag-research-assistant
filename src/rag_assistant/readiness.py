"""Liveness checks for the two external dependencies the graph can't function without.
Kept lightweight and side-effect-free: no embedding calls, no web-search requests spent —
these run on every `/ready` poll from a load balancer/orchestrator, so cost has to stay ~0."""

import httpx

from rag_assistant.config import get_settings
from rag_assistant.retrieval.vector_store import get_vector_store


def check_chroma() -> tuple[bool, str | None]:
    try:
        get_vector_store()._collection.count()
    except Exception as exc:
        return False, str(exc)
    return True, None


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
