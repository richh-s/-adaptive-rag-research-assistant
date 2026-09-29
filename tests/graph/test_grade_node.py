import pytest

from rag_assistant.graph.nodes.grade import after_grade, grade_and_score
from rag_assistant.schemas.models import DocGrade, FusedDocument


def _grades(*grades):
    """An async stand-in for `grade_documents`, which the node awaits."""

    async def _stub(question, docs):
        return list(grades)

    return _stub


def _doc(content: str) -> FusedDocument:
    return FusedDocument(content=content, source_id="x", rrf_score=1.0)


async def test_grade_and_score_computes_mean_confidence(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.grade.grade_documents",
        _grades(DocGrade(relevant=True, score=0.8), DocGrade(relevant=True, score=0.4)),
    )

    result = await grade_and_score(
        {"question": "q", "route": "vector", "fused_documents": [_doc("a"), _doc("b")]}
    )

    assert result["confidence_score"] == pytest.approx(0.6)
    assert result["needs_correction"] is False  # 0.6 is not below the 0.6 threshold


def _low_confidence(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.grade.grade_documents",
        _grades(DocGrade(relevant=False, score=0.1)),
    )


async def test_the_corpus_is_re_asked_before_the_pipeline_leaves_it(monkeypatch):
    """A low grade says retrieval failed, not *why* it failed.

    Escalating straight to web search assumed the corpus could not contain the answer, when
    the ordinary cause is a question phrased the way a person asks rather than the way a
    document writes -- the one case where leaving the corpus cannot help.
    """
    _low_confidence(monkeypatch)

    result = await grade_and_score(
        {"question": "q", "route": "vector", "fused_documents": [_doc("a")]}
    )

    assert result["needs_correction"] is True
    assert result["correction_action"] == "refine"


async def test_web_search_is_the_second_escalation_not_the_first(monkeypatch):
    """Once the corpus has been re-asked and still graded badly, leaving it is the move."""
    _low_confidence(monkeypatch)

    result = await grade_and_score(
        {
            "question": "q",
            "route": "vector",
            "fused_documents": [_doc("a")],
            "refinement_attempted": True,
        }
    )

    assert result["correction_action"] == "web"


async def test_each_escalation_is_tried_at_most_once(monkeypatch):
    """Both guards set: nothing left to try, so the answer goes out with whatever was found
    and the synthesis prompt's abstention instructions carry it."""
    _low_confidence(monkeypatch)

    result = await grade_and_score(
        {
            "question": "q",
            "route": "vector",
            "fused_documents": [_doc("a")],
            "refinement_attempted": True,
            "correction_attempted": True,
        }
    )

    assert result["needs_correction"] is False
    assert result["correction_action"] is None


async def test_the_both_route_can_still_re_ask_the_corpus(monkeypatch):
    """`both` used to get no correction at all, on the reasoning that a run which already
    searched the web has nothing to escalate to. True of the web, false of the corpus."""
    _low_confidence(monkeypatch)

    result = await grade_and_score(
        {"question": "q", "route": "both", "fused_documents": [_doc("a")]}
    )

    assert result["correction_action"] == "refine"


async def test_a_web_only_route_has_no_local_half_to_re_ask(monkeypatch):
    """The one route that genuinely has nothing left: no local retrieval happened, so there
    is no query to rewrite against the corpus and the web has already been searched."""
    _low_confidence(monkeypatch)

    result = await grade_and_score(
        {"question": "q", "route": "web", "fused_documents": [_doc("a")]}
    )

    assert result["needs_correction"] is False
    assert result["correction_action"] is None


async def test_a_refined_both_route_does_not_fall_through_to_web(monkeypatch):
    """`both` already searched the web on its first pass, so a second web search would
    re-run the identical query for the identical results."""
    _low_confidence(monkeypatch)

    result = await grade_and_score(
        {
            "question": "q",
            "route": "both",
            "fused_documents": [_doc("a")],
            "refinement_attempted": True,
        }
    )

    assert result["correction_action"] is None


async def test_grade_reranks_relevant_docs_and_drops_irrelevant_ones(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.grade.grade_documents",
        _grades(
            DocGrade(relevant=True, score=0.7),
            DocGrade(relevant=False, score=0.1),
            DocGrade(relevant=True, score=0.9),
        ),
    )

    docs = [_doc("a"), _doc("b"), _doc("c"), _doc("ungraded-tail")]
    result = await grade_and_score({"question": "q", "route": "web", "fused_documents": docs})

    # Relevant docs reordered by grade score (c=0.9 before a=0.7), irrelevant b dropped,
    # ungraded tail kept behind the graded docs in its original RRF position.
    assert [d.content for d in result["fused_documents"]] == ["c", "a", "ungraded-tail"]


async def test_grade_keeps_documents_untouched_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.grade.grade_documents",
        _grades(DocGrade(relevant=False, score=0.1)),
    )

    docs = [_doc("a")]
    result = await grade_and_score({"question": "q", "route": "web", "fused_documents": docs})

    # Pruning everything would make synthesis look like retrieval returned nothing.
    assert "fused_documents" not in result


async def test_grade_skips_rerank_when_correction_will_rerun_fusion(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.grade.grade_documents",
        _grades(DocGrade(relevant=True, score=0.2)),
    )

    result = await grade_and_score(
        {"question": "q", "route": "vector", "fused_documents": [_doc("a")]}
    )

    assert result["needs_correction"] is True
    assert "fused_documents" not in result


def test_after_grade_routes_correctly():
    assert after_grade({"correction_action": "refine"}) == "refine_retrieval"
    assert after_grade({"correction_action": "web"}) == "corrective_web_search"
    assert after_grade({"correction_action": None}) == "synthesize_answer"


def test_after_grade_falls_back_to_the_boolean_when_no_action_is_present():
    """A state assembled without the newer key still has to route somewhere sensible rather
    than silently skipping a correction it asked for."""
    assert after_grade({"needs_correction": True}) == "corrective_web_search"
    assert after_grade({"needs_correction": False}) == "synthesize_answer"
