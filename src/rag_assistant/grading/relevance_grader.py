import logging

from rag_assistant import metrics
from rag_assistant.content_trust import build_untrusted_context, new_nonce
from rag_assistant.llm import get_structured_llm
from rag_assistant.prompts.grading_prompt import GRADING_PROMPT
from rag_assistant.schemas.models import DocGrade, DocGradeBatch, FusedDocument

logger = logging.getLogger(__name__)


def grade_documents(question: str, docs: list[FusedDocument]) -> list[DocGrade]:
    """Grades every document's relevance to the question in a single LLM call (Corrective-RAG
    style) rather than one call per document, so grading cost stays flat regardless of how
    many documents were fused."""
    if not docs:
        return []

    # Same fencing as synthesis. Grading is the earlier of the two surfaces a hostile
    # document reaches, and the more consequential: these grades drive the confidence score
    # and whether corrective web search runs at all, so a document that talks its way to a
    # high grade also suppresses the search that might have found something better.
    nonce = new_nonce()
    numbered, injection_categories = build_untrusted_context(
        [(doc.source_id, doc.content) for doc in docs], nonce=nonce
    )
    if injection_categories:
        metrics.record_injection_signals(injection_categories)
        logger.warning(
            "Documents being graded contain injection-shaped phrasing: %s",
            ", ".join(injection_categories),
            extra={"injection_categories": injection_categories},
        )
    llm = get_structured_llm(DocGradeBatch)
    try:
        result: DocGradeBatch = llm.invoke(
            GRADING_PROMPT.format(question=question, documents=numbered)
        )
    except Exception:
        # Structured-output parsing occasionally fails when a provider returns malformed
        # tool-call arguments (e.g. a JSON-encoded string instead of a parsed list) -- treat
        # as ungraded rather than crashing the whole graph invocation, mirroring
        # WebSearchTool.search's degrade-to-empty behavior for the same class of provider
        # flakiness. Default to trusting the retrieval (relevant/high score) rather than
        # assuming irrelevance: a provider hiccup here isn't evidence the docs are bad, and
        # scoring them 0.0 was wrongly forcing needs_correction's corrective web search on
        # every ungraded batch, flooding good local results with unnecessary web ones.
        logger.warning(
            "Document grading failed for question=%r; trusting retrieval", question, exc_info=True
        )
        return [DocGrade(relevant=True, score=1.0) for _ in docs]

    if len(result.grades) != len(docs):
        # Model didn't return one grade per doc -- can't align grades to the wrong documents,
        # so trust the retrieval rather than misgrading (see above).
        return [DocGrade(relevant=True, score=1.0) for _ in docs]
    return result.grades
