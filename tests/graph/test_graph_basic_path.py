import pytest

from rag_assistant.graph.build_graph import build_graph

_CONFIG = {"recursion_limit": 50}


@pytest.mark.live
def test_vector_routed_question_returns_cited_answer():
    """Assumes `rag-assistant ingest` has already been run against data/corpus."""
    result = build_graph().invoke(
        {"question": "Who founded Anthropic and what is their safety research called?"}, _CONFIG
    )

    assert result["route"] in ("vector", "both")
    assert result["final_answer"]
    assert result["vector_results"]
    assert result["bm25_results"]
    assert result["fused_documents"]
    assert result["doc_grades"]
    assert 0.0 <= result["confidence_score"] <= 1.0


@pytest.mark.live
def test_web_routed_question_returns_answer():
    result = build_graph().invoke(
        {"question": "What is the most recent Claude model release?"}, _CONFIG
    )

    assert result["route"] in ("web", "both")
    assert result["final_answer"]
    assert result["web_results"]
    assert result["fused_documents"]


@pytest.mark.live
def test_compound_question_decomposes_into_multiple_subqueries():
    result = build_graph().invoke(
        {"question": "Compare Anthropic and Mistral AI's founding stories and safety focus."},
        _CONFIG,
    )

    assert len(result["sub_queries"]) >= 2
    assert result["final_answer"]
    assert result["vector_results"]
    assert result["bm25_results"]
    assert result["fused_documents"]


@pytest.mark.live
def test_out_of_corpus_vector_question_can_trigger_corrective_web_search():
    """A question the router should classify as vector-only but the local corpus can't
    answer well -- best-effort trigger for the corrective loop; not guaranteed since
    routing is LLM-driven, but the confidence/grading fields must always be well-formed."""
    result = build_graph().invoke(
        {"question": "What safety research did Anthropic publish this week?"}, _CONFIG
    )

    assert 0.0 <= result["confidence_score"] <= 1.0
    if result["route"] == "vector" and result["confidence_score"] < 0.6:
        assert result["correction_attempted"] is True
        assert result["web_results"]
