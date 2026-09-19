"""Tests for the deterministic eval metrics and the baseline regression gate.

All offline: the metrics are computed from graph output rather than by calling a model, which
is the property that makes them usable as a build gate in the first place. That also means
the gate's own logic can be tested without spending a single API call.
"""

import json

import pytest

from rag_assistant.eval.baseline import (
    BaselineNotFound,
    compare,
    load_baseline,
    save_baseline,
)
from rag_assistant.eval.golden_dataset import load_golden_dataset
from rag_assistant.eval.metrics import aggregate, score_question


def scored(**overrides):
    defaults = dict(
        question="Who founded Anthropic?",
        category="factual",
        expected_route="vector",
        actual_route="vector",
        expected_sources=["anthropic.md"],
        actual_sources=["anthropic.md"],
        citation_count=2,
    )
    return score_question(**{**defaults, **overrides})


# ---- per-question scoring ----


def test_route_match_is_scored():
    assert scored(actual_route="vector").route_match is True
    assert scored(actual_route="web").route_match is False


def test_source_recall_is_the_fraction_of_expected_sources_retrieved():
    metrics = scored(expected_sources=["a.md", "b.md"], actual_sources=["a.md", "z.md"])

    assert metrics.source_recall == 0.5


def test_reciprocal_rank_rewards_finding_the_source_earlier():
    first = scored(actual_sources=["anthropic.md", "z.md"])
    third = scored(actual_sources=["x.md", "y.md", "anthropic.md"])
    missing = scored(actual_sources=["x.md"])

    assert first.reciprocal_rank == 1.0
    assert third.reciprocal_rank == pytest.approx(1 / 3)
    assert missing.reciprocal_rank == 0.0


def test_rows_with_no_expected_sources_score_recall_as_not_applicable():
    """Averaging a not-applicable row as 0.0 would make the aggregate measure dataset
    composition rather than system quality."""
    metrics = scored(category="unanswerable", expected_sources=[], citation_count=0)

    assert metrics.source_recall is None
    assert metrics.reciprocal_rank is None


def test_unanswerable_rows_score_abstention_not_retrieval():
    correct = scored(category="unanswerable", expected_sources=[], citation_count=0)
    confabulated = scored(category="unanswerable", expected_sources=[], citation_count=3)

    assert correct.abstention_correct is True
    assert confabulated.abstention_correct is False


def test_answerable_rows_penalise_staying_silent():
    """The mirror-image failure, and the more dangerous one: declining to answer when the
    corpus does contain the answer."""
    silent = scored(citation_count=0)

    assert silent.abstained is True
    assert silent.abstention_correct is False


# ---- aggregation ----


def test_aggregate_averages_only_applicable_rows():
    per_question = [
        scored(),
        scored(actual_route="web"),
        scored(category="unanswerable", expected_sources=[], citation_count=0),
    ]

    metrics = aggregate(per_question)

    assert metrics.question_count == 3
    assert metrics.route_accuracy == pytest.approx(2 / 3)
    # Only the two rows with expected sources contribute to recall.
    assert metrics.source_recall == 1.0
    assert metrics.abstention_accuracy == 1.0


def test_a_category_with_no_rows_does_not_drag_the_score_down():
    """A dataset with no unanswerable rows should read as 'nothing to fail here', not as a
    zero that permanently fails the gate."""
    metrics = aggregate([scored(citation_count=1)])

    assert metrics.abstention_accuracy == 1.0


def test_gated_scores_are_the_four_aggregates():
    metrics = aggregate([scored()])

    assert set(metrics.gated_scores()) == {
        "route_accuracy",
        "source_recall",
        "mean_reciprocal_rank",
        "abstention_accuracy",
    }


# ---- the gate ----


def test_regression_beyond_tolerance_fails(tmp_path):
    metrics = aggregate([scored(actual_route="web")])  # route_accuracy 0.0
    baseline = {"route_accuracy": 1.0}

    comparison = compare(metrics, baseline, tolerance=0.05)

    assert comparison.passed is False
    assert [c.name for c in comparison.regressions] == ["route_accuracy"]


def test_a_dip_within_tolerance_passes():
    """LLM routing isn't deterministic even at temperature 0; a gate that fails on one
    borderline flip gets ignored, which is worse than no gate."""
    metrics = aggregate([scored()])
    baseline = {"route_accuracy": 1.04, "source_recall": 1.0}

    assert compare(metrics, baseline, tolerance=0.05).passed is True


def test_improvements_never_fail_the_gate():
    metrics = aggregate([scored()])
    baseline = {"route_accuracy": 0.5}

    comparison = compare(metrics, baseline, tolerance=0.05)

    assert comparison.passed is True
    assert comparison.comparisons[0].delta == pytest.approx(0.5)


def test_metrics_absent_from_the_baseline_are_not_compared():
    """A newly added metric must not fail a build against a baseline recorded before it
    existed."""
    metrics = aggregate([scored(actual_route="web")])

    comparison = compare(metrics, {"source_recall": 1.0}, tolerance=0.0)

    assert [c.name for c in comparison.comparisons] == ["source_recall"]
    assert comparison.passed is True


def test_baseline_roundtrips_through_disk(tmp_path):
    metrics = aggregate([scored()])
    path = tmp_path / "baseline.json"

    save_baseline(metrics, path)
    loaded = load_baseline(path)

    assert loaded == metrics.gated_scores()
    assert "note" in json.loads(path.read_text())


def test_a_missing_baseline_raises_rather_than_silently_passing(tmp_path):
    """Treating a missing baseline as a pass makes the gate inert exactly when it is
    misconfigured."""
    with pytest.raises(BaselineNotFound) as exc:
        load_baseline(tmp_path / "nope.json")

    assert "--record-baseline" in str(exc.value)


# ---- dataset ----


def test_golden_dataset_covers_the_adversarial_categories():
    """A dataset of only answerable questions can't catch the failure that matters most --
    confidently answering something the corpus doesn't contain."""
    categories = {q.category for q in load_golden_dataset()}

    assert {"factual", "unanswerable", "multi_hop", "no_retrieval"} <= categories


def test_unanswerable_rows_declare_no_expected_sources():
    for question in load_golden_dataset():
        if question.category == "unanswerable":
            assert question.expected_sources == []


# ---- acceptable_routes ----


def test_a_defensible_alternative_route_counts_as_a_match():
    """Routing is genuinely ambiguous for a real share of questions. Asserting one answer
    measured the dataset's labelling as much as the router."""
    metrics = score_question(
        question="What is Anthropic's current market valuation?",
        category="factual",
        expected_route="vector",
        acceptable_routes=["vector", "both"],
        actual_route="both",
        expected_sources=["anthropic.md"],
        actual_sources=["anthropic.md"],
        citation_count=2,
    )

    assert metrics.route_match is True


def test_a_route_outside_the_acceptable_set_is_still_a_miss():
    """The widening must not turn the metric off."""
    metrics = score_question(
        question="What is Anthropic's current market valuation?",
        category="factual",
        expected_route="vector",
        acceptable_routes=["vector", "both"],
        actual_route="none",
        expected_sources=["anthropic.md"],
        actual_sources=[],
        citation_count=0,
    )

    assert metrics.route_match is False


def test_an_unlabelled_row_falls_back_to_the_single_expected_route():
    """Rows written before this field stay valid and stay strict."""
    metrics = score_question(
        question="Who founded Anthropic?",
        category="factual",
        expected_route="vector",
        actual_route="both",
        expected_sources=["anthropic.md"],
        actual_sources=["anthropic.md"],
        citation_count=1,
    )

    assert metrics.route_match is False


def test_every_dataset_row_lists_its_expected_route_as_acceptable():
    """A row whose expected route is not in its own acceptable set would score the labelled
    answer as wrong -- the kind of dataset bug that looks like a system regression."""
    from rag_assistant.eval.golden_dataset import load_golden_dataset

    for question in load_golden_dataset():
        assert question.expected_route in question.acceptable_routes, question.question


def test_the_dataset_is_not_widened_into_meaninglessness():
    """Guards the one way this change could quietly destroy the metric: marking every route
    acceptable everywhere would make route_accuracy permanently 1.0."""
    from rag_assistant.eval.golden_dataset import load_golden_dataset

    questions = load_golden_dataset()
    assert not any(len(q.acceptable_routes) >= 4 for q in questions)
    # A majority still asserts exactly one defensible route.
    single = sum(1 for q in questions if len(q.acceptable_routes) == 1)
    assert single > len(questions) / 2


def test_every_expected_source_names_a_document_that_exists():
    """A row citing a filename the corpus does not contain scores source_recall 0 forever,
    which reads as a retrieval regression rather than the typo it is."""
    from rag_assistant.config import PROJECT_ROOT
    from rag_assistant.eval.golden_dataset import load_golden_dataset

    available = {p.name for p in (PROJECT_ROOT / "data" / "corpus").glob("*.md")}
    for question in load_golden_dataset():
        for source in question.expected_sources:
            assert source in available, f"{question.question!r} names missing source {source!r}"


def test_rows_that_should_abstain_expect_no_sources():
    """`unanswerable` scores abstention, and `expected_sources` being non-empty would
    additionally score retrieval against a question whose whole point is that nothing should
    be found -- two metrics disagreeing about one row."""
    from rag_assistant.eval.golden_dataset import load_golden_dataset

    for question in load_golden_dataset():
        if question.category in ("unanswerable", "no_retrieval"):
            assert question.expected_sources == [], question.question


def test_the_dataset_covers_every_category_and_route():
    """A category with no rows scores 1.0 by construction (see `_mean`), so a dataset that
    quietly loses one reports a perfect score for a thing it stopped testing."""
    from rag_assistant.eval.golden_dataset import load_golden_dataset

    questions = load_golden_dataset()
    assert {q.category for q in questions} == {
        "factual",
        "multi_hop",
        "unanswerable",
        "current",
        "no_retrieval",
    }
    assert {q.expected_route for q in questions} == {"vector", "web", "both", "none"}
    # Small datasets move several points on one flipped decision; this is the floor at which
    # the aggregates are worth quoting at all.
    assert len(questions) >= 50


# ---- stale baselines ----


def test_a_stale_baseline_refuses_to_gate(tmp_path):
    """Comparing against an invalid baseline is worse than not gating: it reports a pass or a
    failure with equal confidence and neither means anything."""
    import json

    from rag_assistant.eval.baseline import BaselineStale, load_baseline

    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "metrics": {"route_accuracy": 1.0},
                "stale": True,
                "stale_reason": "prompts changed",
            }
        )
    )

    with pytest.raises(BaselineStale) as excinfo:
        load_baseline(path)

    assert "prompts changed" in str(excinfo.value)
    assert "--record-baseline" in str(excinfo.value)


def test_recording_a_baseline_clears_the_stale_marker(tmp_path):
    import json

    from rag_assistant.eval.baseline import load_baseline, save_baseline

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"metrics": {"route_accuracy": 0.5}, "stale": True}))

    save_baseline(aggregate([scored()]), path)

    assert "stale" not in json.loads(path.read_text())
    assert load_baseline(path)["route_accuracy"] == 1.0


def test_the_committed_baseline_is_currently_marked_stale():
    """Documents the live state rather than asserting a number nobody has re-measured. The
    committed baseline was recorded against a 28-row dataset and the pre-trust-preamble
    prompts; it must not be compared against until re-recorded with real API keys."""
    import json

    from rag_assistant.eval.baseline import DEFAULT_BASELINE_PATH

    payload = json.loads(DEFAULT_BASELINE_PATH.read_text())

    assert payload.get("stale") is True
    assert "re-record" in payload["stale_reason"].lower()
