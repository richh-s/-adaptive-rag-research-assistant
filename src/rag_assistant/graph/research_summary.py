from rag_assistant.schemas.api import NodeLatency, ResearchSummary, RetrievalCounts


def _doc_count(results: list) -> int:
    return sum(len(r.docs) for r in results)


def build_research_summary(state: dict) -> ResearchSummary:
    """Pure function turning final graph state into the explainability panel's data --
    every field here is already computed by some node; this just reshapes it for the API."""
    node_timings = state.get("node_timings", [])
    return ResearchSummary(
        route=state.get("route"),
        sub_queries=state.get("sub_queries", []),
        retrieval_counts=RetrievalCounts(
            vector=_doc_count(state.get("vector_results", [])),
            bm25=_doc_count(state.get("bm25_results", [])),
            web=_doc_count(state.get("web_results", [])),
        ),
        fused_document_count=len(state.get("fused_documents", [])),
        confidence_score=state.get("confidence_score"),
        correction_attempted=bool(state.get("correction_attempted")),
        node_latencies_ms=[NodeLatency(**t) for t in node_timings],
        total_latency_ms=round(sum(t["latency_ms"] for t in node_timings), 1),
    )
