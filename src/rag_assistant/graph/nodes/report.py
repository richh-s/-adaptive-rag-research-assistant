import re

from rag_assistant.config import get_settings
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

    caveat = _groundedness_caveat(state)
    if caveat:
        lines.extend([caveat, ""])

    return {"research_report": "\n".join(lines)}


def _groundedness_caveat(state: ResearchState) -> str | None:
    """A visible note when the answer asserted more than its sources support.

    On the report, never on `final_answer`. The answer is what gets streamed to the browser,
    stored in the transcript and replayed as conversation history, and editing it on the word
    of a check that is itself a model call would rewrite history on a maybe. The report is
    the surface that already carries the pipeline's own commentary about how the answer was
    produced, so the caveat belongs beside the source list rather than inside the prose.

    Silent unless a check actually ran and actually found something: an unverified answer
    gets no note, because "not checked" is not "checked and clean" and a reassuring absence
    would conflate them.
    """
    if not state.get("groundedness_checked"):
        return None
    unsupported = state.get("unsupported_claims") or []
    score = state.get("groundedness_score")
    if not unsupported or score is None:
        return None
    if score >= get_settings().groundedness_threshold:
        return None
    count = len(unsupported)
    noun = "statement" if count == 1 else "statements"
    return (
        f"> **Check:** {count} {noun} in this answer could not be traced back to the sources "
        f"above. Treat {'it' if count == 1 else 'them'} as unverified."
    )
