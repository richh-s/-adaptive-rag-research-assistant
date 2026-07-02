import pytest

from rag_assistant.fusion.rrf import reciprocal_rank_fusion
from rag_assistant.schemas.models import RetrievedDoc


def _doc(content: str, source_id: str = "") -> RetrievedDoc:
    return RetrievedDoc(content=content, source_id=source_id)


def test_document_ranked_in_multiple_lists_outranks_single_list_top_result():
    shared = _doc("shared content", "shared")
    solo = _doc("solo content", "solo")

    fused = reciprocal_rank_fusion([[shared, solo], [shared]])

    assert fused[0].content == "shared content"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_deduplicates_identical_content_across_lists():
    doc = _doc("same text", "a")

    fused = reciprocal_rank_fusion([[doc], [doc], [doc]])

    assert len(fused) == 1
    assert fused[0].rrf_score == pytest.approx(3 * (1 / 61))


def test_empty_input_produces_no_fused_documents():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_score_formula_matches_reciprocal_rank_sum():
    fused = reciprocal_rank_fusion([[_doc("a"), _doc("b")]], k=60)

    scores = {d.content: d.rrf_score for d in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)


def test_results_sorted_descending_by_score():
    fused = reciprocal_rank_fusion([[_doc("first"), _doc("second"), _doc("third")]])

    assert [d.rrf_score for d in fused] == sorted(
        [d.rrf_score for d in fused], reverse=True
    )


def test_document_ranked_in_three_lists_outranks_document_in_one():
    consensus = _doc("consensus content", "consensus")
    solo = _doc("solo content", "solo")

    fused = reciprocal_rank_fusion([[consensus], [consensus, solo], [consensus]])

    assert fused[0].content == "consensus content"
    assert fused[0].rrf_score == pytest.approx(3 * (1 / 61))
    assert fused[1].content == "solo content"
    assert fused[0].rrf_score > fused[1].rrf_score
