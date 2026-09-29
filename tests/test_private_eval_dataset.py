"""Structural checks for a local, git-ignored golden dataset over a private corpus.

The sample-corpus dataset is checked in test_eval_gate.py. A private one (see .gitignore:
data/golden_eval/private/) gets the same guarantees whenever it exists -- a malformed row
there scores as a retrieval regression just as silently -- and skips cleanly in CI, which
has neither the dataset nor the corpus it quotes.
"""

import json

import pytest

from rag_assistant.config import PROJECT_ROOT
from rag_assistant.eval.golden_dataset import load_golden_dataset

PRIVATE_DIR = PROJECT_ROOT / "data" / "golden_eval" / "private"
PRIVATE_DATASET = PRIVATE_DIR / "dataset.jsonl"
PRIVATE_CORPUS = PROJECT_ROOT / "data" / "private_corpus"
# Any other suite kept beside the main one -- a multilingual set, a domain-specific set. They
# are too small to carry the main set's coverage rules, but the rules that make a row *valid*
# apply to every one of them.
OTHER_DATASETS = sorted(p for p in PRIVATE_DIR.glob("*.jsonl") if p != PRIVATE_DATASET)

pytestmark = pytest.mark.skipif(
    not PRIVATE_DATASET.exists(), reason="no private golden dataset in this checkout"
)


@pytest.fixture(scope="module")
def questions():
    return load_golden_dataset(PRIVATE_DATASET)


def test_covers_every_category_and_route(questions):
    assert {q.category for q in questions} == {
        "factual",
        "multi_hop",
        "unanswerable",
        "current",
        "no_retrieval",
    }
    assert {q.expected_route for q in questions} == {"vector", "web", "both", "none"}
    assert len(questions) >= 50


def test_every_row_lists_its_expected_route_as_acceptable(questions):
    for q in questions:
        assert q.expected_route in q.acceptable_routes, q.question


def test_routes_are_not_widened_into_meaninglessness(questions):
    assert not any(len(q.acceptable_routes) >= 4 for q in questions)
    assert sum(1 for q in questions if len(q.acceptable_routes) == 1) > len(questions) / 2


def test_rows_that_should_abstain_expect_no_sources(questions):
    for q in questions:
        if q.category in ("unanswerable", "no_retrieval", "current"):
            assert q.expected_sources == [], q.question


def test_answerable_rows_carry_reference_contexts(questions):
    """Context recall is scored against these; an empty list scores 0 forever."""
    for q in questions:
        if q.category in ("factual", "multi_hop"):
            assert q.reference_contexts, q.question


@pytest.mark.skipif(not PRIVATE_CORPUS.exists(), reason="private corpus not present")
def test_every_expected_source_names_a_document_that_exists(questions):
    available = {p.name for p in PRIVATE_CORPUS.rglob("*") if p.is_file()}
    for q in questions:
        for source in q.expected_sources:
            assert source in available, f"{q.question!r} names missing source {source!r}"


@pytest.mark.skipif(not OTHER_DATASETS, reason="no additional private suites")
@pytest.mark.parametrize("path", OTHER_DATASETS or [None], ids=lambda p: p.name if p else "none")
def test_additional_suites_are_structurally_valid(path):
    """Scores from a malformed row look like a regression, whichever suite it is in."""
    questions = load_golden_dataset(path)

    assert questions, f"{path.name} is empty"
    for q in questions:
        assert q.expected_route in q.acceptable_routes, q.question
        if q.category in ("factual", "multi_hop"):
            assert q.reference_contexts, q.question
            assert q.expected_sources, q.question
        else:
            assert q.expected_sources == [], q.question


@pytest.mark.skipif(not OTHER_DATASETS, reason="no additional private suites")
@pytest.mark.parametrize("path", OTHER_DATASETS or [None], ids=lambda p: p.name if p else "none")
def test_each_additional_suite_has_its_own_baseline(path):
    """Scores are comparable only against a baseline recorded on the same questions, so a
    suite sharing another's baseline would gate on numbers that never described it."""
    baseline = path.with_name(f"{path.stem}-baseline.json")

    assert baseline.exists(), f"{path.name} has no {baseline.name}"
    payload = json.loads(baseline.read_text())
    assert payload["question_count"] == len(load_golden_dataset(path))
