from rag_assistant.graph.state import ResearchState
from rag_assistant.llm import get_chat_model
from rag_assistant.prompts.synthesis_prompt import NO_CONTEXT_PROMPT, SYNTHESIS_PROMPT
from rag_assistant.schemas.models import Citation, FusedDocument


def synthesize_answer(state: ResearchState) -> dict:
    """Builds the final cited answer from the fused, deduplicated, rank-ordered documents,
    or answers directly from the model's own knowledge when the router decided no retrieval
    was needed. Citation markers follow fused rank order, so the highest-consensus documents
    get the lowest (most prominent) marker numbers."""
    docs: list[FusedDocument] = state.get("fused_documents", [])

    if not docs:
        answer = get_chat_model().invoke(NO_CONTEXT_PROMPT.format(question=state["question"]))
        return {"final_answer": answer.content, "citations": []}

    context = "\n\n".join(f"[{i + 1}] (source: {d.source_id})\n{d.content}" for i, d in enumerate(docs))
    prompt = SYNTHESIS_PROMPT.format(question=state["question"], context=context)
    answer = get_chat_model().invoke(prompt)
    citations = [Citation(marker=f"[{i + 1}]", source_id=d.source_id) for i, d in enumerate(docs)]
    return {"final_answer": answer.content, "citations": citations}
