"""Liveness checks for the two external dependencies the graph can't function without.
Kept lightweight and side-effect-free: no embedding calls, no Tavily search credits spent —
these run on every `/ready` poll from a load balancer/orchestrator, so cost has to stay ~0."""

import httpx

from rag_assistant.retrieval.vector_store import get_vector_store


def check_chroma() -> tuple[bool, str | None]:
    try:
        get_vector_store()._collection.count()
    except Exception as exc:
        return False, str(exc)
    return True, None


def check_tavily() -> tuple[bool, str | None]:
    try:
        response = httpx.head("https://api.tavily.com", timeout=3.0)
        # Tavily's base domain doesn't necessarily return 2xx for a bare HEAD -- reachability
        # (a response at all, not a connection error/timeout) is the actual signal here.
        del response
    except httpx.HTTPError as exc:
        return False, str(exc)
    return True, None
