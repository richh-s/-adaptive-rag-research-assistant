from rag_assistant.fusion.rrf import reciprocal_rank_fusion
from rag_assistant.graph.state import ResearchState
from rag_assistant.retrieval.reranker import rerank_documents


def fuse_results(state: ResearchState) -> dict:
    """Join point: LangGraph waits for every `retrieve_vector`/`retrieve_bm25`/`web_search`
    Send from the fan-out to finish and merges their writes via the `operator.add` reducer
    before this node runs, so it always sees the complete set of per-sub-query ranked lists."""
    ranked_lists = [
        result.docs
        for result in (
            state.get("vector_results", [])
            + state.get("bm25_results", [])
            + state.get("web_results", [])
        )
    ]
    fused = reciprocal_rank_fusion(ranked_lists)
    # Reranking sits between fusion and grading on purpose: grading is an LLM call over the
    # top-N, so putting a better ordering in front of it means those calls are spent on the
    # documents most likely to matter. A no-op when RERANKER=none (the default).
    # `question` is always present in a real graph run; guarded because a cross-encoder
    # scored against an empty query returns meaningless relevance, and silently reordering
    # by noise is worse than not reranking at all.
    question = state.get("question") or ""
    if question:
        fused = rerank_documents(question, fused)
    return {"fused_documents": fused}
