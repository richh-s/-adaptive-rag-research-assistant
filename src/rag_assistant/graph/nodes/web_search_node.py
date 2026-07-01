from rag_assistant.retrieval.web_search import WebSearchTool
from rag_assistant.schemas.models import SubQueryResult


def web_search(state: dict) -> dict:
    """Live web search for one sub-query. Invoked once per sub-query via `Send`, so
    `state` here is just `{"sub_query": str}`, not the full graph state."""
    sub_query = state["sub_query"]
    results = WebSearchTool().search(sub_query, max_results=4)
    return {"web_results": [SubQueryResult(sub_query=sub_query, docs=results)]}
