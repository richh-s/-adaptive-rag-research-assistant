from rag_assistant.cache import cache_get, cache_key, cache_set
from rag_assistant.config import get_settings
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
    question = state["question"]
    key = cache_key("synthesis", question, state.get("route", ""), *(d.source_id for d in docs))
    cached = cache_get(key)
    if cached is not None:
        return {
            "final_answer": cached["final_answer"],
            "citations": [Citation(**c) for c in cached["citations"]],
        }

    if not docs:
        # An empty `fused_documents` means two very different things: the router decided
        # retrieval wasn't needed at all ("none" -- safe to answer from general knowledge),
        # or retrieval was attempted on the "vector"/"web"/"both" route and came back empty
        # (risky -- answering confidently here looks indistinguishable from a grounded answer).
        if state.get("route") == "none":
            prompt = NO_CONTEXT_PROMPT.format(question=question)
        else:
            prompt = EMPTY_RETRIEVAL_PROMPT.format(question=question)
        answer = get_chat_model().invoke(prompt)
        result = {"final_answer": answer.content, "citations": []}
    else:
        context = "\n\n".join(f"[{i + 1}] (source: {d.source_id})\n{d.content}" for i, d in enumerate(docs))
        prompt = SYNTHESIS_PROMPT.format(question=question, context=context)
        answer = get_chat_model().invoke(prompt)
        citations = [Citation(marker=f"[{i + 1}]", source_id=d.source_id) for i, d in enumerate(docs)]
        result = {"final_answer": answer.content, "citations": citations}

    cache_set(
        key,
        {"final_answer": result["final_answer"], "citations": [c.model_dump() for c in result["citations"]]},
        get_settings().cache_ttl_synthesis,
    )
    return result
