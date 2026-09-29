"""Downvotes becoming eval candidates -- the only signal in this system sourced from a human.

The gate compares against a fixed set of questions, which is what makes it a regression test
and what makes it go stale: it cannot tell you the set stopped resembling what people ask.
These tests pin the two halves of closing that loop -- that downvoted questions get carried
over, and that nothing about the failure gets carried over *as an assertion*.
"""

import json

from rag_assistant.eval.golden_dataset import export_feedback_candidates, load_golden_dataset


def _downvote(question, route="vector", note=None, confidence=0.2):
    return {"question": question, "route": route, "note": note, "confidence_score": confidence}


def test_downvoted_questions_become_candidate_rows(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("")

    path, written, skipped = export_feedback_candidates(
        [_downvote("What did Anthropic raise in 2024?")],
        dataset_path=dataset,
        candidates_path=candidates,
    )

    assert path == candidates
    assert (written, skipped) == (1, 0)
    row = json.loads(candidates.read_text().strip())
    assert row["question"] == "What did Anthropic raise in 2024?"


def test_the_assertions_are_left_blank(tmp_path):
    """`ground_truth`, `reference_contexts` and `expected_sources` are what the metrics score
    against. Filling them from the answer the user just rejected would encode the failure as
    the expected behaviour, and the gate would then defend the bug."""
    candidates = tmp_path / "candidates.jsonl"
    export_feedback_candidates(
        [_downvote("bad answer question")],
        dataset_path=tmp_path / "none.jsonl",
        candidates_path=candidates,
    )

    row = json.loads(candidates.read_text().strip())
    assert row["ground_truth"] == ""
    assert row["reference_contexts"] == []
    assert row["expected_sources"] == []
    assert row["_needs_review"] is True


def test_the_observed_route_is_carried_as_a_starting_point(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    export_feedback_candidates(
        [_downvote("q", route="web", note="answered about the wrong company")],
        dataset_path=tmp_path / "none.jsonl",
        candidates_path=candidates,
    )

    row = json.loads(candidates.read_text().strip())
    assert row["expected_route"] == "web"
    assert row["_user_note"] == "answered about the wrong company"


def test_questions_already_in_the_dataset_are_skipped(tmp_path):
    """Otherwise every export re-queues the whole backlog."""
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"question": "Who founded Anthropic?"}) + "\n")
    candidates = tmp_path / "candidates.jsonl"

    _, written, skipped = export_feedback_candidates(
        [_downvote("Who founded Anthropic?"), _downvote("Something new?")],
        dataset_path=dataset,
        candidates_path=candidates,
    )

    assert (written, skipped) == (1, 1)


def test_exporting_twice_converges_rather_than_accumulating(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    rows = [_downvote("repeated question")]

    export_feedback_candidates(rows, dataset_path=dataset, candidates_path=candidates)
    _, written, skipped = export_feedback_candidates(
        rows, dataset_path=dataset, candidates_path=candidates
    )

    assert (written, skipped) == (0, 1)
    assert len(candidates.read_text().strip().splitlines()) == 1


def test_matching_ignores_case_and_surrounding_whitespace(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"question": "Who Founded Anthropic?"}) + "\n")
    candidates = tmp_path / "candidates.jsonl"

    _, written, _ = export_feedback_candidates(
        [_downvote("  who founded anthropic?  ")],
        dataset_path=dataset,
        candidates_path=candidates,
    )

    assert written == 0


def test_a_candidate_row_loads_as_a_golden_question(tmp_path):
    """The file has to be completable in place: a human fills the blanks and moves the row
    into the dataset, rather than retyping it into a different shape."""
    candidates = tmp_path / "candidates.jsonl"
    export_feedback_candidates(
        [_downvote("a question")],
        dataset_path=tmp_path / "none.jsonl",
        candidates_path=candidates,
    )

    loaded = load_golden_dataset(candidates)

    assert len(loaded) == 1
    assert loaded[0].question == "a question"


def test_an_empty_question_is_never_queued(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _, written, skipped = export_feedback_candidates(
        [_downvote("   ")], dataset_path=tmp_path / "none.jsonl", candidates_path=candidates
    )

    assert (written, skipped) == (0, 1)
