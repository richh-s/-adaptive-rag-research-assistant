from rag_assistant.fusion.rrf import reciprocal_rank_fusion
from rag_assistant.graph.state import ResearchState


def fuse_results(state: ResearchState) -> dict:
    """Join point: LangGraph waits for every `retrieve_vector`/`web_search` Send from the
    fan-out to finish and merges their writes via the `operator.add` reducer before this
    node runs, so it always sees the complete set of per-sub-query ranked lists."""
    ranked_lists = [
        result.docs for result in state.get("vector_results", []) + state.get("web_results", [])
    ]
    fused = reciprocal_rank_fusion(ranked_lists)
    return {"fused_documents": fused}
