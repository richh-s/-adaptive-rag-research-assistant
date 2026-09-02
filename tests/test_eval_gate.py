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
    metrics = scored(
        expected_sources=["a.md", "b.md"], actual_sources=["a.md", "z.md"]
    )

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
