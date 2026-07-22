import logging

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

    numbered = "\n\n".join(f"[{i + 1}] {doc.content}" for i, doc in enumerate(docs))
    llm = get_structured_llm(DocGradeBatch)
    try:
        result: DocGradeBatch = llm.invoke(GRADING_PROMPT.format(question=question, documents=numbered))
    except Exception:
        # Structured-output parsing occasionally fails when a provider returns malformed
        # tool-call arguments (e.g. a JSON-encoded string instead of a parsed list) -- treat
        # as ungraded rather than crashing the whole graph invocation, mirroring
        # WebSearchTool.search's degrade-to-empty behavior for the same class of provider
        # flakiness. Default to trusting the retrieval (relevant/high score) rather than
        # assuming irrelevance: a provider hiccup here isn't evidence the docs are bad, and
        # scoring them 0.0 was wrongly forcing needs_correction's corrective web search on
        # every ungraded batch, flooding good local results with unnecessary web ones.
        logger.warning("Document grading failed for question=%r; trusting retrieval", question, exc_info=True)
        return [DocGrade(relevant=True, score=1.0) for _ in docs]

    if len(result.grades) != len(docs):
        # Model didn't return one grade per doc -- can't align grades to the wrong documents,
        # so trust the retrieval rather than misgrading (see above).
        return [DocGrade(relevant=True, score=1.0) for _ in docs]
    return result.grades
