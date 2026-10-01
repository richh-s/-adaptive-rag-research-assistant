import json
from pathlib import Path

from rag_assistant.config import PROJECT_ROOT
from rag_assistant.schemas.models import GoldenQuestion

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "golden_eval" / "dataset.jsonl"


def load_golden_dataset(path: Path | None = None) -> list[GoldenQuestion]:
    dataset_path = path or DEFAULT_DATASET_PATH
    questions = []
    with dataset_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(GoldenQuestion.model_validate(json.loads(line)))
    return questions


CANDIDATES_PATH = DEFAULT_DATASET_PATH.parent / "candidates.jsonl"


def export_feedback_candidates(
    downvoted: list[dict],
    dataset_path: Path | None = None,
    candidates_path: Path | None = None,
) -> tuple[Path, int, int]:
    """Turns downvoted questions into golden-dataset rows awaiting a human.

    The gate compares against a *fixed* set of questions, which is exactly what makes it a
    regression test and exactly what makes it go stale: it cannot tell you the set stopped
    resembling what people actually ask. Downvotes are the only signal here sourced from a
    human rather than from the system's own behaviour, and until now nothing consumed them --
    they were counted, summarised, and left in a table for someone to notice.

    Deliberately *not* appended to the gated dataset. A row's `ground_truth`,
    `reference_contexts` and `expected_sources` are the assertions the metrics are computed
    against; auto-filling them from the answer the user just rejected would encode the
    failure as the expected behaviour, and the gate would then defend the bug. So the
    question is carried over with the route the system actually chose, everything that
    constitutes a claim is left blank, and the file is separate until a person completes it.

    Returns (path, written, skipped) -- skipped counts questions already present in either
    the dataset or a previous export, so running this repeatedly converges instead of
    accumulating duplicates.
    """
    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    candidates_path = candidates_path or CANDIDATES_PATH

    def _questions_in(path: Path) -> set[str]:
        if not path.exists():
            return set()
        seen = set()
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(json.loads(line).get("question", "").strip().lower())
        return seen

    known = _questions_in(dataset_path) | _questions_in(candidates_path)
    written = skipped = 0
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with candidates_path.open("a") as f:
        for row in downvoted:
            question = (row.get("question") or "").strip()
            if not question or question.lower() in known:
                skipped += 1
                continue
            known.add(question.lower())
            f.write(
                json.dumps(
                    {
                        "question": question,
                        # Blank on purpose -- see the docstring. These are the assertions.
                        "ground_truth": "",
                        "reference_contexts": [],
                        "expected_sources": [],
                        # The route the system chose, as a starting point rather than an
                        # expectation: a downvote may well be a routing complaint.
                        "expected_route": row.get("route") or "vector",
                        "category": "factual",
                        "_needs_review": True,
                        "_observed_confidence": row.get("confidence_score"),
                        "_user_note": row.get("note"),
                    }
                )
                + "\n"
            )
            written += 1
    return candidates_path, written, skipped
