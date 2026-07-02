import pytest

from rag_assistant.graph.nodes.grade import after_grade, grade_and_score
from rag_assistant.schemas.models import DocGrade, FusedDocument


def _doc(content: str) -> FusedDocument:
    return FusedDocument(content=content, source_id="x", rrf_score=1.0)


def test_grade_and_score_computes_mean_confidence(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.grade.grade_documents",
        lambda question, docs: [
            DocGrade(relevant=True, score=0.8),
            DocGrade(relevant=True, score=0.4),
        ],
    )

    result = grade_and_score(
        {"question": "q", "route": "vector", "fused_documents": [_doc("a"), _doc("b")]}
    )

    assert result["confidence_score"] == pytest.approx(0.6)
    assert result["needs_correction"] is False  # 0.6 is not below the 0.6 threshold


def test_grade_and_score_triggers_correction_when_low_confidence_and_vector_only(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.grade.grade_documents",
        lambda question, docs: [DocGrade(relevant=False, score=0.1)],
    )

    result = grade_and_score({"question": "q", "route": "vector", "fused_documents": [_doc("a")]})

    assert result["needs_correction"] is True


def test_grade_and_score_never_corrects_when_already_attempted(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.grade.grade_documents",
        lambda question, docs: [DocGrade(relevant=False, score=0.1)],
    )

    result = grade_and_score(
        {
            "question": "q",
            "route": "vector",
            "fused_documents": [_doc("a")],
            "correction_attempted": True,
        }
    )

    assert result["needs_correction"] is False


def test_grade_and_score_never_corrects_when_route_is_web_or_both(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.grade.grade_documents",
        lambda question, docs: [DocGrade(relevant=False, score=0.1)],
    )

    for route in ("web", "both"):
        state = {"question": "q", "route": route, "fused_documents": [_doc("a")]}
        assert grade_and_score(state)["needs_correction"] is False


def test_after_grade_routes_correctly():
    assert after_grade({"needs_correction": True}) == "corrective_web_search"
    assert after_grade({"needs_correction": False}) == "synthesize_answer"
