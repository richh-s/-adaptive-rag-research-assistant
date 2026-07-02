from rag_assistant.graph.nodes.report import format_report
from rag_assistant.schemas.models import Citation


def test_format_report_includes_answer_and_sources():
    state = {
        "final_answer": "The answer is 42.",
        "citations": [Citation(marker="[1]", source_id="doc_a.md")],
        "route": "vector",
        "route_reasoning": "local knowledge base has this.",
        "sub_queries": ["question"],
        "confidence_score": 0.85,
        "correction_attempted": False,
    }

    result = format_report(state)

    report = result["research_report"]
    assert report.startswith("The answer is 42.")
    assert "**Sources:**" in report
    assert "- [1] doc_a.md" in report
    assert "Route: `vector` -- local knowledge base has this." in report
    assert "Retrieval confidence: 0.85" in report


def test_format_report_omits_sources_section_when_no_citations():
    state = {
        "final_answer": "General knowledge answer.",
        "citations": [],
        "route": "none",
        "route_reasoning": None,
        "sub_queries": [],
    }

    result = format_report(state)

    report = result["research_report"]
    assert "**Sources:**" not in report
    assert "Route: `none` -- no retrieval needed" in report


def test_format_report_lists_decomposed_sub_queries():
    state = {
        "final_answer": "Answer.",
        "citations": [],
        "route": "web",
        "route_reasoning": "needs current info.",
        "sub_queries": ["sub question one", "sub question two"],
    }

    result = format_report(state)

    report = result["research_report"]
    assert "Decomposed into 2 sub-queries:" in report
    assert "- sub question one" in report
    assert "- sub question two" in report


def test_format_report_notes_corrective_fallback_when_attempted():
    state = {
        "final_answer": "Answer.",
        "citations": [],
        "route": "vector",
        "route_reasoning": "reason.",
        "sub_queries": [],
        "confidence_score": 0.2,
        "correction_attempted": True,
    }

    result = format_report(state)

    assert "Low confidence triggered a corrective web search fallback." in result["research_report"]
