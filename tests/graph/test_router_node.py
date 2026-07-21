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
