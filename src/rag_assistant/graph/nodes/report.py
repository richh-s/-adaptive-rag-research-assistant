import re

from rag_assistant.graph.state import ResearchState

_MARKER_RE = re.compile(r"\[\d+\]")


def used_citations(final_answer: str, citations: list) -> list:
    """The citations the answer actually references.

    `state["citations"]` carries one entry per document that reached the synthesis prompt,
    whether or not the model referenced it -- markers are assigned positionally before the
    model has said anything. So its length measures *context size*, not how well grounded the
    answer is, and the two diverge most precisely where it matters: an honest "no relevant
    sources were found" still has a full context behind it.

    Shared with the eval harness for exactly that reason. Counting the unfiltered list there
    made every abstention look like a heavily-cited answer.
    """
    used_markers = set(_MARKER_RE.findall(final_answer or ""))
    return [c for c in citations if c.marker in used_markers]


def format_report(state: ResearchState) -> dict:
    """Assembles the final markdown report: the answer plus a source list. Routing/retrieval/
    confidence detail lives only in the structured research summary (see build_research_summary
    in api.py) -- keeping it out of this prose avoids showing non-technical readers internal
    jargon ("route: both", "confidence: 0.05") next to their answer.

    The source list is filtered to citation markers the model actually used in `final_answer`
    (fused documents the model never referenced would otherwise show up as unexplained
    "sources") and deduped by source_id, since several fused chunks often come from the same
    file -- listing that file three times reads as a bug to a non-technical reader."""
    lines = [state["final_answer"], ""]

    cited = used_citations(state["final_answer"], state.get("citations", []))

    if cited:
        markers_by_source: dict[str, list[str]] = {}
        order: list[str] = []
        for c in cited:
            if c.source_id not in markers_by_source:
                markers_by_source[c.source_id] = []
                order.append(c.source_id)
            markers_by_source[c.source_id].append(c.marker)

        lines.append("**Sources:**")
        lines.extend(
            f"- {''.join(markers_by_source[source_id])} {source_id}" for source_id in order
        )
        lines.append("")

    return {"research_report": "\n".join(lines)}
