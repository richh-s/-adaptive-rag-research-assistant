import logging

from rag_assistant.graph.state import ResearchState
from rag_assistant.llm import get_structured_llm
from rag_assistant.prompts.condense_prompt import CONDENSE_PROMPT
from rag_assistant.schemas.models import CondensedQuestion

logger = logging.getLogger(__name__)

# The condensation prompt only needs enough context to resolve references in the latest
# message, not the whole session -- old turns beyond this window are almost never what a
# pronoun points at, and full reports pasted back in as history would dwarf the question.
MAX_HISTORY_TURNS = 8
MAX_TURN_CHARS = 600


def _format_history(history: list[dict]) -> str:
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if len(content) > MAX_TURN_CHARS:
            content = content[:MAX_TURN_CHARS] + " ..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def condense_question(state: ResearchState) -> dict:
    """Conversational memory: rewrite a follow-up ("what about their pricing?") into a
    self-contained question using the chat history, so routing/decomposition/retrieval all
    operate on something that stands alone. First-turn questions (no history) pass through
    untouched, which also keeps the CLI/eval paths -- which never send history -- unchanged.

    Failure here must never fail the request: any LLM error, or a rewrite that comes back
    empty, degrades to using the question as-is."""
    question = state["question"]
    history = state.get("chat_history") or []
    if not history:
        return {"original_question": None}

    try:
        llm = get_structured_llm(CondensedQuestion)
        result: CondensedQuestion = llm.invoke(
            CONDENSE_PROMPT.format(history=_format_history(history), question=question)
        )
        standalone = (result.standalone_question or "").strip()
    except Exception:
        logger.warning("condense_question failed; using the question as-is", exc_info=True)
        standalone = ""

    if not standalone or standalone == question:
        return {"original_question": None}
    return {"question": standalone, "original_question": question}
