import logging
from typing import Any, Protocol

from tavily import TavilyClient

from rag_assistant.config import get_settings
from rag_assistant.schemas.models import RetrievedDoc

logger = logging.getLogger(__name__)


class TavilyLike(Protocol):
    def search(self, query: str, max_results: int) -> dict[str, Any]: ...


class WebSearchTool:
    def __init__(self, client: TavilyLike | None = None):
        self._client = client or TavilyClient(api_key=get_settings().tavily_api_key)

    def search(self, query: str, max_results: int = 5) -> list[RetrievedDoc]:
        try:
            response = self._client.search(query=query, max_results=max_results)
        except Exception:
            # A Tavily outage/rate-limit/network blip shouldn't crash the graph node --
            # treat it the same as "the web had nothing," and let routing/grading/corrective
            # logic downstream handle a thin result set the way it already does.
            logger.warning("Tavily search failed for query=%r; returning no results", query, exc_info=True)
            return []
        return [
            RetrievedDoc(
                content=result.get("content", ""),
                metadata={"title": result.get("title", ""), "url": result.get("url", "")},
                source_id=result.get("url", ""),
                score=result.get("score"),
            )
            for result in response.get("results", [])
        ]
