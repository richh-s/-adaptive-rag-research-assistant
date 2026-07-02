from rag_assistant.graph.state import ResearchState


def format_report(state: ResearchState) -> dict:
    """Assembles the final markdown report: the answer, its sources, and a "how this was
    researched" transparency section covering the routing decision, any decomposition, and
    the confidence/correction outcome. A pure function so both the CLI and the FastAPI
    endpoint (Phase 8) can render the exact same report without duplicating this logic."""
    lines = [state["final_answer"], ""]

    citations = state.get("citations", [])
    if citations:
        lines.append("**Sources:**")
        lines.extend(f"- {c.marker} {c.source_id}" for c in citations)
        lines.append("")

    lines.append("**How this was researched:**")
    route = state.get("route") or "none"
    lines.append(f"- Route: `{route}` -- {state.get('route_reasoning') or 'no retrieval needed'}")

    sub_queries = state.get("sub_queries", [])
    if len(sub_queries) > 1:
        lines.append(f"- Decomposed into {len(sub_queries)} sub-queries:")
        lines.extend(f"  - {sub_query}" for sub_query in sub_queries)

    if "confidence_score" in state:
        lines.append(f"- Retrieval confidence: {state['confidence_score']:.2f}")

    if state.get("correction_attempted"):
        lines.append("- Low confidence triggered a corrective web search fallback.")

    return {"research_report": "\n".join(lines)}
