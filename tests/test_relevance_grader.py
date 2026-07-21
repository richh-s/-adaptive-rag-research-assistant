from unittest.mock import MagicMock

from rag_assistant.grading.relevance_grader import grade_documents
from rag_assistant.schemas.models import DocGrade, DocGradeBatch, FusedDocument


def _doc(content: str) -> FusedDocument:
    return FusedDocument(content=content, source_id="x", rrf_score=1.0)


def test_grade_documents_returns_grades_in_order(monkeypatch):
    fake_batch = DocGradeBatch(
        grades=[DocGrade(relevant=True, score=0.9), DocGrade(relevant=False, score=0.1)]
    )
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = fake_batch

    monkeypatch.setattr(
        "rag_assistant.grading.relevance_grader.get_structured_llm", lambda schema: fake_structured_llm
    )

    grades = grade_documents("question", [_doc("a"), _doc("b")])

    assert grades == fake_batch.grades


def test_grade_documents_handles_empty_input():
    assert grade_documents("question", []) == []


def test_grade_documents_falls_back_when_grade_count_mismatches(monkeypatch):
    fake_batch = DocGradeBatch(grades=[DocGrade(relevant=True, score=0.9)])
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = fake_batch

    monkeypatch.setattr(
        "rag_assistant.grading.relevance_grader.get_structured_llm", lambda schema: fake_structured_llm
    )

    grades = grade_documents("question", [_doc("a"), _doc("b")])

    assert len(grades) == 2
    assert all(g.relevant is False and g.score == 0.0 for g in grades)
