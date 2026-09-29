from rag_assistant.config import get_settings
from rag_assistant.grading.groundedness import verify_answer
from rag_assistant.graph.state import ResearchState


async def verify_groundedness(state: ResearchState) -> dict:
    """Checks the written answer against the documents synthesis was actually given.

    Runs against `context_documents` rather than `fused_documents` on purpose: those are the
    documents that survived parent expansion and the context budget, in the order the
    citation markers were assigned from. Verifying against the full fused list would ask the
    model about text the answer's author never saw, and the markers it reports would point at
    different documents than the ones the reader is shown.
    """
    if not get_settings().groundedness_check:
        return {}
    result = await verify_answer(
        state["question"],
        state.get("final_answer") or "",
        state.get("context_documents") or [],
    )
    return {
        "groundedness_checked": result.checked,
        "groundedness_score": result.score,
        "unsupported_claims": result.unsupported_claims,
    }
