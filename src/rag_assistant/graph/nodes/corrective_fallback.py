from rag_assistant.graph.state import ResearchState
from rag_assistant.retrieval.web_search import WebSearchTool
from rag_assistant.schemas.models import SubQueryResult


def corrective_web_search(state: ResearchState) -> dict:
    """Corrective-RAG fallback: local retrieval graded low-confidence, so run one extra web
    search pass across the same sub-queries. Loops back to `fuse_results` to re-fuse with
    the added results. `correction_attempted` is the guard that stops this from looping more
    than once -- `grade_and_score` checks it before setting `needs_correction` again."""
    results = [
        SubQueryResult(sub_query=sub_query, docs=WebSearchTool().search(sub_query, max_results=4))
        for sub_query in state["sub_queries"]
    ]
    return {"web_results": results, "correction_attempted": True}
