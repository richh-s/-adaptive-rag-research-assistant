"""Structural checks for the multi-turn golden dataset, and that the harness threads history.

Condensation is the first node in the graph and it rewrites the question every later node
reads -- routing, decomposition, both retrieval paths and synthesis all see its output rather
than what the user typed. No golden row exercised it, so a regression that broke follow-up
resolution, or that dropped the fencing around the conversation (an untrusted surface, since
an assistant turn carries whatever the web path retrieved), moved no measured number at all.

This set is kept separate from `dataset.jsonl` rather than merged into it, following the same
rule the private corpus does: scores are only comparable against a baseline recorded on the
same questions, so a set with its own questions gets its own baseline.
"""

import json
from unittest.mock import AsyncMock

import pytest

from rag_assistant.config import PROJECT_ROOT
from rag_assistant.eval.golden_dataset import load_golden_dataset
from rag_assistant.eval.run_eval import _run_question
from rag_assistant.schemas.models import GoldenQuestion

DATASET = PROJECT_ROOT / "data" / "golden_eval" / "conversational.jsonl"
CORPUS = PROJECT_ROOT / "data" / "corpus"


@pytest.fixture(scope="module")
def questions():
    return load_golden_dataset(DATASET)


def test_every_row_actually_carries_history(questions):
    """The whole point of the set. A row without history is a single-turn row in the wrong
    file, and it would dilute the one thing this set measures."""
    assert questions
    assert all(q.chat_history for q in questions)


def test_history_turns_are_well_formed(questions):
    for q in questions:
        for turn in q.chat_history:
            assert turn["role"] in {"user", "assistant"}
            assert turn["content"].strip()


def test_history_alternates_and_ends_with_the_assistant(questions):
    """A follow-up arrives after an answer. History ending on a user turn would mean the
    previous question was never answered, which is not a state the API can produce."""
    for q in questions:
        roles = [t["role"] for t in q.chat_history]
        assert roles[0] == "user"
        assert roles[-1] == "assistant"
        assert all(a != b for a, b in zip(roles, roles[1:]))


def test_the_question_is_not_answerable_without_the_history(questions):
    """At least the referential rows must genuinely depend on their history -- otherwise the
    set measures ordinary retrieval and reports it as condensation coverage."""
    referential = [q for q in questions if q.category == "follow_up"]
    assert len(referential) >= 4
    dependent = [
        q
        for q in referential
        if any(
            token in q.question.lower()
            for token in (" their ", " it?", " that ", " them ", "what about", " its ")
        )
    ]
    assert len(dependent) >= 3


def test_expected_sources_exist_in_the_corpus(questions):
    """A row quoting a file that isn't there scores as a retrieval failure forever."""
    available = {p.name for p in CORPUS.glob("*.md")}
    for q in questions:
        for source in q.expected_sources:
            assert source in available, f"{source!r} is not in the sample corpus"


def test_reference_contexts_are_real_passages(questions):
    """Non-LLM context precision/recall is string overlap against these, so a paraphrase
    scores zero and looks like a retrieval regression rather than a dataset error."""
    corpus_text = "\n".join(p.read_text() for p in CORPUS.glob("*.md"))
    normalized = " ".join(corpus_text.split())
    for q in questions:
        for context in q.reference_contexts:
            head = " ".join(context.split())[:60]
            assert head in normalized, f"reference context not found in corpus: {head!r}"


def test_rows_expecting_no_sources_are_the_abstention_cases(questions):
    """An empty `expected_sources` has to be deliberate: the metric reads it as 'this row is
    scored on whether the system declined', so a row that merely forgot to list sources would
    silently invert what it measures."""
    for q in questions:
        if not q.expected_sources:
            assert q.category in {"unanswerable", "no_retrieval", "current"}


def test_the_set_covers_more_than_pronoun_resolution(questions):
    """Ellipsis, topic switches and an already-self-contained follow-up are different
    failures from pronoun resolution, and the last one matters most: condensation must not
    damage a question that needed no rewriting."""
    assert len({q.category for q in questions}) >= 4


def test_dataset_is_valid_jsonl():
    with DATASET.open() as f:
        for line in f:
            if line.strip():
                GoldenQuestion.model_validate(json.loads(line))


def test_the_harness_passes_history_into_the_graph():
    """The dataset is inert unless the runner threads it. Before this, `_run_question` built
    the graph input from `question` alone, so every follow-up row would have been asked cold
    and scored as a retrieval failure."""
    graph = AsyncMock()
    graph.ainvoke.return_value = {"final_answer": "", "route": "vector"}
    history = [
        {"role": "user", "content": "Who founded Anthropic?"},
        {"role": "assistant", "content": "Dario and Daniela Amodei [1]."},
    ]

    _run_question(
        graph,
        GoldenQuestion(
            question="What is their flagship product?",
            ground_truth="Claude.",
            reference_contexts=[],
            expected_route="vector",
            expected_sources=["anthropic.md"],
            category="follow_up",
            chat_history=history,
        ),
    )

    assert graph.ainvoke.call_args.args[0]["chat_history"] == history


def test_single_turn_rows_still_send_an_empty_history():
    """`condense_question` returns early on an empty history, so single-turn rows behave
    exactly as they did before this field existed -- which is what keeps the committed
    baseline comparable."""
    graph = AsyncMock()
    graph.ainvoke.return_value = {"final_answer": "", "route": "vector"}

    _run_question(
        graph,
        GoldenQuestion(
            question="Who founded Anthropic?",
            ground_truth="x",
            reference_contexts=[],
            expected_route="vector",
            expected_sources=["anthropic.md"],
        ),
    )

    assert graph.ainvoke.call_args.args[0]["chat_history"] == []
