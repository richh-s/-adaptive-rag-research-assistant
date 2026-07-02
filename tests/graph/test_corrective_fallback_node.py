from unittest.mock import MagicMock

from rag_assistant.graph.nodes.corrective_fallback import corrective_web_search
from rag_assistant.schemas.models import RetrievedDoc


def test_corrective_web_search_runs_one_pass_per_subquery_and_sets_guard(monkeypatch):
    fake_tool = MagicMock()
    fake_tool.search.return_value = [RetrievedDoc(content="web doc", source_id="https://x")]
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.corrective_fallback.WebSearchTool", lambda: fake_tool
    )

    result = corrective_web_search({"sub_queries": ["q1", "q2"]})

    assert result["correction_attempted"] is True
    assert len(result["web_results"]) == 2
    assert {r.sub_query for r in result["web_results"]} == {"q1", "q2"}
    assert fake_tool.search.call_count == 2
