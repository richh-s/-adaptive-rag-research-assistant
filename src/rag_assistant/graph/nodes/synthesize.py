import hashlib
import json

from rag_assistant.cache import cache_get, cache_key, cache_set
from rag_assistant.config import get_settings
from rag_assistant.graph.context_budget import select_context_documents
from rag_assistant.graph.state import ResearchState
from rag_assistant.llm import get_chat_model
from rag_assistant.prompts.synthesis_prompt import (
    EMPTY_RETRIEVAL_PROMPT,
    NO_CONTEXT_PROMPT,
    SYNTHESIS_PROMPT,
)
from rag_assistant.ingestion.ownership import display_source
from rag_assistant.schemas.models import Citation, FusedDocument

# Mirrors condense.py's window: enough turns for continuity of tone/topic, without pasting
# the whole session (including full prior reports) into every synthesis prompt.
_MAX_HISTORY_TURNS = 6
_MAX_TURN_CHARS = 800


def _history_block(history: list[dict]) -> str:
    """Renders recent turns into a prompt section, or "" on the first turn -- the prompts
    interpolate this directly, so an empty first turn leaves them byte-identical to the
    pre-conversational versions."""
    if not history:
        return ""
    lines = []
    for turn in history[-_MAX_HISTORY_TURNS:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if len(content) > _MAX_TURN_CHARS:
            content = content[:_MAX_TURN_CHARS] + " ..."
        lines.append(f"{role}: {content}")
    joined = "\n".join(lines)
    return (
        "\nConversation so far (for continuity only -- answer just the final question, and "
        f"don't repeat what was already covered unless asked):\n{joined}\n"
    )


def synthesize_answer(state: ResearchState) -> dict:
    """Builds the final cited answer from the fused, deduplicated, rank-ordered documents,
    or answers directly from the model's own knowledge when the router decided no retrieval
    was needed. Citation markers follow fused rank order, so the highest-consensus documents
    get the lowest (most prominent) marker numbers."""
    settings = get_settings()
    all_docs: list[FusedDocument] = state.get("fused_documents", [])
    # Bound the prompt before anything is built from it -- the cache key, the numbered
    # context and the citation markers all have to describe the same set of documents, so
    # the budget is applied once here and everything downstream reads `docs`.
    budgeted = select_context_documents(
        all_docs,
        budget_tokens=settings.synthesis_context_budget_tokens,
        chars_per_token=settings.synthesis_chars_per_token,
    )
    docs = budgeted.documents
    question = state["question"]
    history = state.get("chat_history") or []
    # History changes the rendered prompt, so it must be part of the cache identity -- the
    # same standalone question asked in two different conversations may legitimately get
    # differently-phrased answers.
    history_digest = hashlib.sha256(
        json.dumps(history, sort_keys=True).encode()
    ).hexdigest() if history else ""
    key = cache_key(
        "synthesis", question, state.get("route", ""), history_digest, *(d.source_id for d in docs)
    )
    cached = cache_get(key)
    if cached is not None:
        return {
            "final_answer": cached["final_answer"],
            "citations": [Citation(**c) for c in cached["citations"]],
            "context_documents_dropped": budgeted.dropped_documents,
        }

    history_block = _history_block(history)

    if not docs:
        # An empty `fused_documents` means two very different things: the router decided
        # retrieval wasn't needed at all ("none" -- safe to answer from general knowledge),
        # or retrieval was attempted on the "vector"/"web"/"both" route and came back empty
        # (risky -- answering confidently here looks indistinguishable from a grounded answer).
        if state.get("route") == "none":
            prompt = NO_CONTEXT_PROMPT.format(question=question, history_block=history_block)
        else:
            prompt = EMPTY_RETRIEVAL_PROMPT.format(question=question, history_block=history_block)
        answer = get_chat_model().invoke(prompt)
        result = {"final_answer": answer.text, "citations": []}
    else:
        context = "\n\n".join(
            f"[{i + 1}] (source: {display_source(d.source_id)})\n{d.content}"
            for i, d in enumerate(docs)
        )
        prompt = SYNTHESIS_PROMPT.format(
            question=question, context=context, history_block=history_block
        )
        answer = get_chat_model().invoke(prompt)
        citations = [
            Citation(marker=f"[{i + 1}]", source_id=display_source(d.source_id))
            for i, d in enumerate(docs)
        ]
        result = {"final_answer": answer.text, "citations": citations}

    cache_set(
        key,
        {"final_answer": result["final_answer"], "citations": [c.model_dump() for c in result["citations"]]},
        settings.cache_ttl_synthesis,
    )
    return {**result, "context_documents_dropped": budgeted.dropped_documents}
