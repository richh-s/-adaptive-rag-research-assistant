from rag_assistant.llm import get_chat_model
from rag_assistant.prompts.grading_prompt import GRADING_PROMPT
from rag_assistant.schemas.models import DocGrade, DocGradeBatch, FusedDocument


def grade_documents(question: str, docs: list[FusedDocument]) -> list[DocGrade]:
    """Grades every document's relevance to the question in a single LLM call (Corrective-RAG
    style) rather than one call per document, so grading cost stays flat regardless of how
    many documents were fused."""
    if not docs:
        return []

    numbered = "\n\n".join(f"[{i + 1}] {doc.content}" for i, doc in enumerate(docs))
    llm = get_chat_model().with_structured_output(DocGradeBatch)
    result: DocGradeBatch = llm.invoke(GRADING_PROMPT.format(question=question, documents=numbered))

    if len(result.grades) != len(docs):
        # Model didn't return one grade per doc -- treat as ungraded/low-confidence rather
        # than misalign grades to the wrong documents.
        return [DocGrade(relevant=False, score=0.0) for _ in docs]
    return result.grades
