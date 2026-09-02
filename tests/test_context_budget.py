"""Tests for the synthesis context budget."""

import pytest

from rag_assistant.graph.context_budget import estimate_tokens, select_context_documents
from rag_assistant.schemas.models import FusedDocument


def doc(content: str, rrf_score: float = 1.0, source_id: str = "a.md") -> FusedDocument:
    return FusedDocument(content=content, source_id=source_id, rrf_score=rrf_score)


def test_documents_within_budget_all_survive():
    docs = [doc("word " * 10) for _ in range(3)]

    result = select_context_documents(docs, budget_tokens=1000, chars_per_token=4.0)

    assert result.documents == docs
    assert result.dropped_documents == 0
    assert result.truncated_documents == 0


def test_documents_are_kept_in_rank_order_until_the_budget_runs_out():
    """Documents arrive ranked best-first, so spending the budget greedily down the list
    drops exactly what the pipeline already judged least useful."""
    # 40 chars ≈ 10 tokens each at 4 chars/token.
    docs = [doc("x" * 40, source_id=f"{i}.md") for i in range(5)]

    result = select_context_documents(docs, budget_tokens=25, chars_per_token=4.0)

    assert [d.source_id for d in result.documents] == ["0.md", "1.md"]
    assert result.dropped_documents == 3


def test_the_top_document_is_truncated_rather_than_dropped():
    """Returning nothing would make synthesis say no sources were found -- a false statement
    about a retrieval that succeeded and found something too long."""
    docs = [doc("y" * 4000)]

    result = select_context_documents(docs, budget_tokens=100, chars_per_token=4.0)

    assert len(result.documents) == 1
    assert result.truncated_documents == 1
    assert len(result.documents[0].content) == 400
    assert result.dropped_documents == 0


def test_lower_ranked_documents_are_dropped_not_truncated():
    """A fragment of a lower-ranked chunk still gets a citation marker, inviting the model to
    cite a passage cut off before the part that supported the claim."""
    docs = [doc("a" * 200, source_id="first.md"), doc("b" * 4000, source_id="second.md")]

    result = select_context_documents(docs, budget_tokens=100, chars_per_token=4.0)

    assert [d.source_id for d in result.documents] == ["first.md"]
    assert result.truncated_documents == 0
    assert result.dropped_documents == 1


def test_truncation_does_not_mutate_the_original_document():
    original = doc("z" * 4000)

    select_context_documents([original], budget_tokens=10, chars_per_token=4.0)

    assert len(original.content) == 4000


@pytest.mark.parametrize("budget", [0, -1])
def test_a_non_positive_budget_disables_the_cap(budget):
    docs = [doc("q" * 10_000) for _ in range(4)]

    result = select_context_documents(docs, budget_tokens=budget, chars_per_token=4.0)

    assert result.documents == docs
    assert result.dropped_documents == 0


def test_empty_input_is_handled():
    result = select_context_documents([], budget_tokens=100, chars_per_token=4.0)

    assert result.documents == []
    assert result.dropped_documents == 0
    assert result.estimated_tokens == 0


def test_token_estimate_scales_with_the_configured_ratio():
    assert estimate_tokens("x" * 100, chars_per_token=4.0) == 25
    assert estimate_tokens("x" * 100, chars_per_token=2.0) == 50


def test_synthesis_applies_the_budget_and_aligns_citations_with_it(monkeypatch):
    """The citation markers must describe the documents that actually reached the prompt --
    if the budget drops documents but citations still enumerate all of them, every marker
    past the cut points at the wrong source."""
    from unittest.mock import MagicMock

    from rag_assistant.graph.nodes import synthesize as synthesize_module

    monkeypatch.setenv("SYNTHESIS_CONTEXT_BUDGET_TOKENS", "25")
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(text="An answer [1].")
    monkeypatch.setattr(synthesize_module, "get_chat_model", lambda: fake_llm)

    docs = [doc("x" * 40, source_id=f"{i}.md") for i in range(5)]
    result = synthesize_module.synthesize_answer(
        {"question": "q", "fused_documents": docs, "route": "vector"}
    )

    assert [c.source_id for c in result["citations"]] == ["0.md", "1.md"]
    assert result["context_documents_dropped"] == 3
    prompt = fake_llm.invoke.call_args[0][0]
    assert "[3]" not in prompt
