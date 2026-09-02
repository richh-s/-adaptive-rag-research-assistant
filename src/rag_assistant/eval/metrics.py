"""Retrieval and behaviour metrics that need no LLM judge.

RAGAS answers "is this answer faithful and relevant", which is the right question and an
expensive one -- it costs extra model calls per row and its scores move a little between runs
even on identical output. That makes it a poor thing to gate a build on.

These metrics are the complement: deterministic, free, and computed from data the graph
already returned. Same inputs give the same numbers every time, which is the property a
regression gate actually needs. RAGAS still runs alongside (`--llm-judge`) for the quality
questions these can't answer.

What they measure is the part of a RAG pipeline most likely to regress silently: a prompt
tweak that changes routing, a chunking change that drops a source out of the top-k, a
threshold change that makes the system stop abstaining on questions it cannot answer. None of
those show up as an exception or a failing unit test -- the system keeps returning confident
prose, just about the wrong documents.
"""

from dataclasses import dataclass, field


@dataclass
class QuestionMetrics:
    """Per-question scores. `None` means "not applicable to this row" rather than zero --
    averaging a not-applicable as 0.0 would silently drag an aggregate down, which is how a
    metric ends up measuring dataset composition instead of system quality."""

    question: str
    category: str
    route_match: bool
    source_recall: float | None
    reciprocal_rank: float | None
    abstained: bool
    abstention_correct: bool | None


@dataclass
class EvalMetrics:
    """Aggregate scores plus the per-question detail behind them."""

    question_count: int
    route_accuracy: float
    source_recall: float
    mean_reciprocal_rank: float
    abstention_accuracy: float
    per_question: list[QuestionMetrics] = field(default_factory=list)

    def gated_scores(self) -> dict[str, float]:
        """The subset a build gates on. Deliberately the four aggregates and not the
        per-question detail: a gate should fail on "retrieval got worse", not on one row
        reordering."""
        return {
            "route_accuracy": self.route_accuracy,
            "source_recall": self.source_recall,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "abstention_accuracy": self.abstention_accuracy,
        }


def _mean(values: list[float]) -> float:
    """Mean of applicable values, or 1.0 when none apply.

    1.0 rather than 0.0 because these feed a regression gate: a dataset with no rows of some
    category should read as "nothing to fail here", not as a perfect score's opposite. A
    dataset that genuinely has such rows and scores them badly still fails.
    """
    return sum(values) / len(values) if values else 1.0


def score_question(
    *,
    question: str,
    category: str,
    expected_route: str,
    actual_route: str | None,
    expected_sources: list[str],
    actual_sources: list[str],
    citation_count: int,
) -> QuestionMetrics:
    """Scores one golden question against what the graph actually did.

    `actual_sources` is expected in retrieval rank order -- reciprocal rank is read off that
    ordering, so passing an unordered set would silently produce a meaningless MRR.
    """
    source_recall: float | None = None
    reciprocal_rank: float | None = None
    if expected_sources:
        expected = set(expected_sources)
        found = expected & set(actual_sources)
        source_recall = len(found) / len(expected)
        reciprocal_rank = 0.0
        for rank, source in enumerate(actual_sources, start=1):
            if source in expected:
                reciprocal_rank = 1.0 / rank
                break

    # "Abstained" means the answer cited nothing -- the observable signal that the system
    # declined to ground a claim, rather than an attempt to parse hedging out of prose.
    abstained = citation_count == 0
    abstention_correct: bool | None = None
    if category == "unanswerable":
        abstention_correct = abstained
    elif expected_sources:
        # The mirror-image failure, and the more dangerous one in practice: staying silent
        # when the corpus does contain the answer.
        abstention_correct = not abstained

    return QuestionMetrics(
        question=question,
        category=category,
        route_match=actual_route == expected_route,
        source_recall=source_recall,
        reciprocal_rank=reciprocal_rank,
        abstained=abstained,
        abstention_correct=abstention_correct,
    )


def aggregate(per_question: list[QuestionMetrics]) -> EvalMetrics:
    return EvalMetrics(
        question_count=len(per_question),
        route_accuracy=_mean([1.0 if m.route_match else 0.0 for m in per_question]),
        source_recall=_mean([m.source_recall for m in per_question if m.source_recall is not None]),
        mean_reciprocal_rank=_mean(
            [m.reciprocal_rank for m in per_question if m.reciprocal_rank is not None]
        ),
        abstention_accuracy=_mean(
            [
                1.0 if m.abstention_correct else 0.0
                for m in per_question
                if m.abstention_correct is not None
            ]
        ),
        per_question=per_question,
    )
