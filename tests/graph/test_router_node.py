from unittest.mock import MagicMock

from rag_assistant.graph.nodes.router import after_route, route_query
from rag_assistant.schemas.models import RouteDecision


def test_route_query_returns_route_and_reasoning(monkeypatch):
    fake_decision = RouteDecision(route="vector", reasoning="Local KB covers this.")
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = fake_decision

    monkeypatch.setattr(
        "rag_assistant.graph.nodes.router.get_structured_llm", lambda schema: fake_structured_llm
    )

    result = route_query({"question": "Who founded Anthropic?"})

    assert result == {"route": "vector", "route_reasoning": "Local KB covers this."}


def test_after_route_skips_decomposition_when_no_retrieval_needed():
    assert after_route({"route": "none"}) == "synthesize_answer"


def test_after_route_goes_to_decomposition_otherwise():
    assert after_route({"route": "vector"}) == "decompose_query"
    assert after_route({"route": "web"}) == "decompose_query"
    assert after_route({"route": "both"}) == "decompose_query"


def test_route_query_returns_cached_route_without_calling_llm(monkeypatch):
    fake_structured_llm = MagicMock()
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.router.get_structured_llm", lambda schema: fake_structured_llm
    )
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.router.cache_get",
        lambda key: {"route": "web", "route_reasoning": "cached"},
    )

    result = route_query({"question": "Who founded Anthropic?"})

    assert result == {"route": "web", "route_reasoning": "cached"}
    fake_structured_llm.invoke.assert_not_called()


def test_route_query_caches_result_after_llm_call(monkeypatch):
    fake_decision = RouteDecision(route="vector", reasoning="Local KB covers this.")
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = fake_decision
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.router.get_structured_llm", lambda schema: fake_structured_llm
    )
    monkeypatch.setattr("rag_assistant.graph.nodes.router.cache_get", lambda key: None)
    captured = {}
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.router.cache_set",
        lambda key, value, ttl: captured.update(key=key, value=value, ttl=ttl),
    )

    route_query({"question": "Who founded Anthropic?"})

    assert captured["value"] == {"route": "vector", "route_reasoning": "Local KB covers this."}
    assert captured["ttl"] == 300
