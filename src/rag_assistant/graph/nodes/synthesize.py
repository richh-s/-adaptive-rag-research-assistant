from rag_assistant.graph.state import ResearchState
from rag_assistant.llm import get_chat_model
from rag_assistant.prompts.synthesis_prompt import (
    EMPTY_RETRIEVAL_PROMPT,
    NO_CONTEXT_PROMPT,
    SYNTHESIS_PROMPT,
)
from rag_assistant.schemas.models import Citation, FusedDocument


def synthesize_answer(state: ResearchState) -> dict:
    """Builds the final cited answer from the fused, deduplicated, rank-ordered documents,
    or answers directly from the model's own knowledge when the router decided no retrieval
    was needed. Citation markers follow fused rank order, so the highest-consensus documents
    get the lowest (most prominent) marker numbers."""
    docs: list[FusedDocument] = state.get("fused_documents", [])

    if not docs:
        # An empty `fused_documents` means two very different things: the router decided
        # retrieval wasn't needed at all ("none" -- safe to answer from general knowledge),
        # or retrieval was attempted on the "vector"/"web"/"both" route and came back empty
        # (risky -- answering confidently here looks indistinguishable from a grounded answer).
        if state.get("route") == "none":
            prompt = NO_CONTEXT_PROMPT.format(question=state["question"])
        else:
            prompt = EMPTY_RETRIEVAL_PROMPT.format(question=state["question"])
        answer = get_chat_model().invoke(prompt)
        return {"final_answer": answer.content, "citations": []}

    context = "\n\n".join(f"[{i + 1}] (source: {d.source_id})\n{d.content}" for i, d in enumerate(docs))
    prompt = SYNTHESIS_PROMPT.format(question=state["question"], context=context)
    answer = get_chat_model().invoke(prompt)
    citations = [Citation(marker=f"[{i + 1}]", source_id=d.source_id) for i, d in enumerate(docs)]
    return {"final_answer": answer.content, "citations": citations}
