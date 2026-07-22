import pytest

from rag_assistant.retrieval.web_search import WebSearchTool
from rag_assistant.schemas.models import RetrievedDoc


class _FakeDdgsClient:
    def __init__(self, response: list[dict]):
        self._response = response

    def search(self, query: str, max_results: int, timeout: float = 60) -> list[dict]:
        return self._response


def test_search_normalizes_ddgs_response():
    fake_response = [
        {
            "title": "Anthropic",
            "href": "https://anthropic.com",
            "body": "Anthropic builds Claude.",
        },
        {
            "title": "OpenAI",
            "href": "https://openai.com",
            "body": "OpenAI builds GPT.",
        },
    ]
    tool = WebSearchTool(client=_FakeDdgsClient(fake_response))

    results = tool.search("AI labs", max_results=2)

    assert results == [
        RetrievedDoc(
            content="Anthropic builds Claude.",
            metadata={"title": "Anthropic", "url": "https://anthropic.com"},
            source_id="https://anthropic.com",
            score=None,
        ),
        RetrievedDoc(
            content="OpenAI builds GPT.",
            metadata={"title": "OpenAI", "url": "https://openai.com"},
            source_id="https://openai.com",
            score=None,
        ),
    ]


def test_search_handles_missing_fields_gracefully():
    tool = WebSearchTool(client=_FakeDdgsClient([{"body": "partial data"}]))

    results = tool.search("query")

    assert len(results) == 1
    assert results[0].content == "partial data"
    assert results[0].source_id == ""
    assert results[0].score is None


def test_search_handles_empty_results():
    tool = WebSearchTool(client=_FakeDdgsClient([]))

    assert tool.search("query") == []


class _FailingDdgsClient:
    def search(self, query: str, max_results: int, timeout: float = 60) -> list[dict]:
        raise RuntimeError("simulated DuckDuckGo outage")


def test_web_search_failure_returns_empty():
    tool = WebSearchTool(client=_FailingDdgsClient())

    assert tool.search("query") == []


def test_search_passes_configured_timeout_to_client():
    captured = {}

    class _CapturingDdgsClient:
        def search(self, query: str, max_results: int, timeout: float = 60) -> list[dict]:
            captured["timeout"] = timeout
            return []

    tool = WebSearchTool(client=_CapturingDdgsClient())

    tool.search("query")

    assert captured["timeout"] == 10.0


def test_search_returns_cached_results_without_calling_client(monkeypatch):
    class _UnusedClient:
        def search(self, query: str, max_results: int, timeout: float = 60) -> list[dict]:
            raise AssertionError("client should not be called on a cache hit")

    cached_doc = RetrievedDoc(content="cached", metadata={}, source_id="https://cached.com", score=None)
    monkeypatch.setattr(
        "rag_assistant.retrieval.web_search.cache_get", lambda key: [cached_doc.model_dump()]
    )
    tool = WebSearchTool(client=_UnusedClient())

    results = tool.search("query")

    assert results == [cached_doc]


def test_search_caches_results_after_client_call(monkeypatch):
    fake_response = [{"body": "fresh", "href": "https://fresh.com", "title": "Fresh"}]
    monkeypatch.setattr("rag_assistant.retrieval.web_search.cache_get", lambda key: None)
    captured = {}
    monkeypatch.setattr(
        "rag_assistant.retrieval.web_search.cache_set",
        lambda key, value, ttl: captured.update(key=key, value=value, ttl=ttl),
    )
    tool = WebSearchTool(client=_FakeDdgsClient(fake_response))

    results = tool.search("query", max_results=1)

    assert captured["value"] == [doc.model_dump() for doc in results]
    assert captured["ttl"] == 600


@pytest.mark.live
def test_live_web_search_returns_results():
    results = WebSearchTool().search("what is LangGraph", max_results=2)

    assert len(results) > 0
    assert results[0].content
