"""Post-synthesis verification: does the answer stay inside the documents it was written from.

The distinction these tests keep pinning is between "verified and clean" and "not verified".
Both leave `unsupported_claims` empty, and a system that reported the second as the first
would turn a provider outage, a disabled setting and an abstention into a clean bill of
health -- a number in the transparency panel that means the opposite of what it says.
"""

from unittest.mock import AsyncMock

import pytest

from rag_assistant.config import get_settings
from rag_assistant.grading.groundedness import verify_answer
from rag_assistant.graph.nodes.report import format_report
from rag_assistant.graph.nodes.verify import verify_groundedness
from rag_assistant.schemas.models import ClaimCheck, FusedDocument, GroundednessReport


def _docs(*contents: str) -> list[FusedDocument]:
    return [
        FusedDocument(content=c, source_id=f"doc_{i}.md", rrf_score=1.0 / (i + 1), metadata={})
        for i, c in enumerate(contents)
    ]


def _llm_returning(report: GroundednessReport) -> AsyncMock:
    fake = AsyncMock()
    fake.ainvoke.return_value = report
    return fake


def _patch_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(
        "rag_assistant.grading.groundedness.get_structured_llm", lambda *a, **k: llm
    )


async def test_a_fully_supported_answer_scores_one(monkeypatch):
    _patch_llm(
        monkeypatch,
        _llm_returning(
            GroundednessReport(
                claims=[
                    ClaimCheck(claim="Revenue was $2.1M.", supported=True, marker="[1]"),
                    ClaimCheck(claim="It grew year on year.", supported=True, marker="[1]"),
                ]
            )
        ),
    )

    result = await verify_answer(
        "How much revenue?", "Revenue was $2.1M [1].", _docs("Revenue: $2.1M")
    )

    assert result.checked is True
    assert result.score == 1.0
    assert result.unsupported_claims == []


async def test_an_invented_claim_is_reported_not_removed(monkeypatch):
    """The failure this exists to catch: retrieval was fine and the write-up added a figure.

    The answer comes back untouched. Editing it would mean rewriting prose whose fluency
    depends on the sentence, on the word of a check that is itself a model call.
    """
    _patch_llm(
        monkeypatch,
        _llm_returning(
            GroundednessReport(
                claims=[
                    ClaimCheck(claim="Revenue was $2.1M.", supported=True, marker="[1]"),
                    ClaimCheck(claim="Headcount doubled to 400.", supported=False, marker=None),
                ]
            )
        ),
    )

    result = await verify_answer("Tell me about the year.", "...", _docs("Revenue: $2.1M"))

    assert result.checked is True
    assert result.score == 0.5
    assert result.unsupported_claims == ["Headcount doubled to 400."]


async def test_a_failed_check_is_unverified_rather_than_ungrounded(monkeypatch):
    """A provider hiccup is not evidence that an answer was fabricated.

    Scoring it 0.0 would put the pipeline's loudest possible warning on an answer nobody
    looked at -- and, worse, would make the metric indistinguishable from a real regression.
    """
    failing = AsyncMock()
    failing.ainvoke.side_effect = RuntimeError("provider exploded")
    _patch_llm(monkeypatch, failing)

    result = await verify_answer("q", "an answer", _docs("some text"))

    assert result.checked is False
    assert result.score is None
    assert result.unsupported_claims == []


async def test_an_abstention_is_checked_but_unscored(monkeypatch):
    """An answer that makes no claims is the system behaving correctly. Scoring it 0.0 would
    flag the most honest thing it does as its least grounded."""
    _patch_llm(monkeypatch, _llm_returning(GroundednessReport(claims=[])))

    result = await verify_answer("q", "I don't have information on that.", _docs("unrelated"))

    assert result.checked is True
    assert result.score is None
    assert result.claims_checked == 0


async def test_verification_is_skipped_when_nothing_was_retrieved(monkeypatch):
    """The "none" route answers from the model's own knowledge. There is no context to be
    grounded in, so there is nothing to check -- and no LLM call to pay for."""
    never = AsyncMock()
    never.ainvoke.side_effect = AssertionError("should not be called")
    _patch_llm(monkeypatch, never)

    assert (await verify_answer("q", "2 + 2 is 4.", [])).checked is False


async def test_the_node_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("GROUNDEDNESS_CHECK", "false")
    get_settings.cache_clear()
    called = AsyncMock()
    monkeypatch.setattr("rag_assistant.graph.nodes.verify.verify_answer", called)

    assert await verify_groundedness({"question": "q", "final_answer": "a"}) == {}
    called.assert_not_called()


async def test_the_node_verifies_the_context_synthesis_saw_not_everything_fused(monkeypatch):
    """`fused_documents` is the larger set retrieval produced; `context_documents` is what
    survived the budget and what the citation markers were numbered from. Verifying against
    the wrong one asks the model about text the answer's author never saw."""
    seen = {}

    async def fake_verify(question, answer, documents):
        seen["sources"] = [d.source_id for d in documents]
        from rag_assistant.grading.groundedness import GroundednessResult

        return GroundednessResult(checked=True, score=1.0)

    monkeypatch.setattr("rag_assistant.graph.nodes.verify.verify_answer", fake_verify)
    budgeted = _docs("kept")
    await verify_groundedness(
        {
            "question": "q",
            "final_answer": "a",
            "context_documents": budgeted,
            "fused_documents": budgeted + _docs("dropped by the budget"),
        }
    )

    assert seen["sources"] == ["doc_0.md"]


@pytest.mark.parametrize(
    "state,expect_caveat",
    [
        (
            {"groundedness_checked": True, "groundedness_score": 0.5, "unsupported_claims": ["x"]},
            True,
        ),
        (
            {"groundedness_checked": True, "groundedness_score": 1.0, "unsupported_claims": []},
            False,
        ),
        # Not checked: no note. "Unverified" must not read as "verified and clean".
        (
            {"groundedness_checked": False, "groundedness_score": None, "unsupported_claims": []},
            False,
        ),
        # Checked, one claim unsupported, but still above threshold -- a single mislabelled
        # claim in a long answer is the check's own error bar, not a finding.
        (
            {"groundedness_checked": True, "groundedness_score": 0.9, "unsupported_claims": ["x"]},
            False,
        ),
    ],
)
def test_the_report_carries_a_caveat_only_when_one_is_earned(state, expect_caveat):
    report = format_report({"final_answer": "An answer [1].", "citations": [], **state})

    assert ("could not be traced back" in report["research_report"]) is expect_caveat


def test_the_caveat_never_edits_the_answer_itself():
    """The answer is streamed, stored and replayed as history. The note belongs beside it."""
    answer = "Revenue was $2.1M and headcount doubled."
    report = format_report(
        {
            "final_answer": answer,
            "citations": [],
            "groundedness_checked": True,
            "groundedness_score": 0.5,
            "unsupported_claims": ["headcount doubled"],
        }
    )

    assert report["research_report"].startswith(answer)
