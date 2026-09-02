from rag_assistant.graph.nodes.report import format_report
from rag_assistant.schemas.models import Citation


def test_format_report_includes_answer_and_sources():
    state = {
        "final_answer": "The answer is 42. [1]",
        "citations": [Citation(marker="[1]", source_id="doc_a.md")],
    }

    result = format_report(state)

    report = result["research_report"]
    assert report.startswith("The answer is 42. [1]")
    assert "**Sources:**" in report
    assert "- [1] doc_a.md" in report


def test_format_report_omits_sources_section_when_no_citations():
    state = {"final_answer": "General knowledge answer.", "citations": []}

    result = format_report(state)

    assert "**Sources:**" not in result["research_report"]


def test_format_report_omits_citations_the_answer_never_referenced():
    state = {
        "final_answer": "Only the first fact matters. [1]",
        "citations": [
            Citation(marker="[1]", source_id="doc_a.md"),
            Citation(marker="[2]", source_id="doc_b.md"),
        ],
    }

    report = format_report(state)["research_report"]

    assert "doc_a.md" in report
    assert "doc_b.md" not in report


def test_format_report_dedupes_repeated_source_across_markers():
    state = {
        "final_answer": "Two chunks from the same file. [1][3]",
        "citations": [
            Citation(marker="[1]", source_id="doc_a.md"),
            Citation(marker="[2]", source_id="doc_b.md"),
            Citation(marker="[3]", source_id="doc_a.md"),
        ],
    }

    report = format_report(state)["research_report"]

    assert report.count("doc_a.md") == 1
    assert "[1][3] doc_a.md" in report


def test_used_citations_counts_only_referenced_markers():
    """The distinction the eval harness got wrong: `citations` holds one entry per document
    that reached the prompt, so its length is context size, not groundedness."""
    from rag_assistant.graph.nodes.report import used_citations
    from rag_assistant.schemas.models import Citation

    citations = [Citation(marker=f"[{i}]", source_id=f"{i}.md") for i in range(1, 6)]

    used = used_citations("An answer citing [2] and [4].", citations)

    assert [c.marker for c in used] == ["[2]", "[4]"]


def test_an_ungrounded_answer_references_nothing_despite_a_full_context():
    """The case that made every abstention look heavily cited: a refusal still has every
    retrieved document sitting behind it."""
    from rag_assistant.graph.nodes.report import used_citations
    from rag_assistant.schemas.models import Citation

    citations = [Citation(marker=f"[{i}]", source_id=f"{i}.md") for i in range(1, 13)]

    assert used_citations("No relevant sources were found for this question.", citations) == []


def test_used_citations_handles_a_missing_answer():
    from rag_assistant.graph.nodes.report import used_citations
    from rag_assistant.schemas.models import Citation

    assert used_citations("", [Citation(marker="[1]", source_id="a.md")]) == []
