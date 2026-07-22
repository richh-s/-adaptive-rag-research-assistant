import logging
from typing import Any, Protocol

from ddgs import DDGS

from rag_assistant.cache import cache_get, cache_key, cache_set
from rag_assistant.config import get_settings
from rag_assistant.schemas.models import RetrievedDoc

logger = logging.getLogger(__name__)


class DdgsLike(Protocol):
    def search(self, query: str, max_results: int, timeout: float = 60) -> list[dict[str, Any]]: ...


class _DdgsClient:
    """Thin wrapper over the `ddgs` package -- an unofficial, key-less DuckDuckGo client.
    No signup, no API key, no billing account required (unlike Google Custom Search, which
    gates even its free tier behind a linked billing account). Trade-off: it scrapes
    DuckDuckGo's HTML rather than calling a documented API, so it can break if DDG changes
    their markup -- WebSearchTool.search() already degrades to [] on any failure, so an
    outage here behaves the same as any other web-search provider outage."""

    def search(self, query: str, max_results: int, timeout: float = 60) -> list[dict[str, Any]]:
        return DDGS(timeout=timeout).text(query, max_results=max_results)


class WebSearchTool:
    def __init__(self, client: DdgsLike | None = None):
        self._client = client or _DdgsClient()

    def search(self, query: str, max_results: int = 5) -> list[RetrievedDoc]:
        key = cache_key("web_search", query, str(max_results))
        cached = cache_get(key)
        if cached is not None:
            return [RetrievedDoc(**doc) for doc in cached]

        try:
            results = self._client.search(
                query=query,
                max_results=max_results,
                timeout=get_settings().web_search_timeout_seconds,
            )
        except Exception:
            # An outage/rate-limit/network blip shouldn't crash the graph node -- treat it
            # the same as "the web had nothing," and let routing/grading/corrective logic
            # downstream handle a thin result set the way it already does.
            logger.warning("Web search failed for query=%r; returning no results", query, exc_info=True)
            return []
        docs = [
            RetrievedDoc(
                content=result.get("body", ""),
                metadata={"title": result.get("title", ""), "url": result.get("href", "")},
                source_id=result.get("href", ""),
                # DDG's text search doesn't return a numeric relevance score -- left as None,
                # which RRF fusion never reads (rank position only, see rrf.py).
                score=None,
            )
            for result in results
        ]
        cache_set(key, [doc.model_dump() for doc in docs], get_settings().cache_ttl_web_search)
        return docs
