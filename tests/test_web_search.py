import pytest

from rag_assistant.retrieval.web_search import WebSearchTool
from rag_assistant.schemas.models import RetrievedDoc


class _FakeTavilyClient:
    def __init__(self, response: dict):
        self._response = response

    def search(self, query: str, max_results: int) -> dict:
        return self._response


def test_search_normalizes_tavily_response():
    fake_response = {
        "results": [
            {
                "title": "Anthropic",
                "url": "https://anthropic.com",
                "content": "Anthropic builds Claude.",
                "score": 0.9,
            },
            {
                "title": "OpenAI",
                "url": "https://openai.com",
                "content": "OpenAI builds GPT.",
                "score": 0.7,
            },
        ]
    }
    tool = WebSearchTool(client=_FakeTavilyClient(fake_response))

    results = tool.search("AI labs", max_results=2)

    assert results == [
        RetrievedDoc(
            content="Anthropic builds Claude.",
            metadata={"title": "Anthropic", "url": "https://anthropic.com"},
            source_id="https://anthropic.com",
            score=0.9,
        ),
        RetrievedDoc(
            content="OpenAI builds GPT.",
            metadata={"title": "OpenAI", "url": "https://openai.com"},
            source_id="https://openai.com",
            score=0.7,
        ),
    ]


def test_search_handles_missing_fields_gracefully():
    tool = WebSearchTool(client=_FakeTavilyClient({"results": [{"content": "partial data"}]}))

    results = tool.search("query")

    assert len(results) == 1
    assert results[0].content == "partial data"
    assert results[0].source_id == ""
    assert results[0].score is None


def test_search_handles_empty_results():
    tool = WebSearchTool(client=_FakeTavilyClient({"results": []}))

    assert tool.search("query") == []


@pytest.mark.live
def test_live_tavily_search_returns_results():
    results = WebSearchTool().search("what is LangGraph", max_results=2)

    assert len(results) > 0
    assert results[0].content
