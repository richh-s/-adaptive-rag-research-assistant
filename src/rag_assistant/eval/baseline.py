"""The regression gate: compare an eval run against recorded baseline scores.

Absolute thresholds ("route accuracy must exceed 0.9") are the obvious design and the wrong
one. Nobody knows the right absolute number for a given corpus, so it gets set to whatever
today's run produced, and from then on it either blocks unrelated work or is quietly lowered
until it blocks nothing. A baseline plus a tolerance asks the question that actually matters:
*did this change make retrieval worse than it was?*

The tolerance exists because LLM routing is not deterministic even at temperature 0 -- a
provider-side model update can flip one borderline routing decision. On a small dataset one
flipped row is a several-point swing, so a zero-tolerance gate would fail on noise, and a
gate that fails on noise gets ignored, which is worse than not having one.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.config import PROJECT_ROOT
from rag_assistant.eval.metrics import EvalMetrics

DEFAULT_BASELINE_PATH = PROJECT_ROOT / "data" / "golden_eval" / "baseline.json"

# One flipped row on a ~20-question dataset moves an aggregate about 5 points, so this
# absorbs a single borderline routing flip and fails on anything systematic.
DEFAULT_TOLERANCE = 0.05


@dataclass
class MetricComparison:
    name: str
    baseline: float
    current: float
    tolerance: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    @property
    def regressed(self) -> bool:
        return self.current < self.baseline - self.tolerance


@dataclass
class BaselineComparison:
    comparisons: list[MetricComparison]

    @property
    def regressions(self) -> list[MetricComparison]:
        return [c for c in self.comparisons if c.regressed]

    @property
    def passed(self) -> bool:
        return not self.regressions


class BaselineNotFound(RuntimeError):
    """Raised when a gate run finds no baseline to compare against.

    Loud on purpose. The tempting alternative -- treat a missing baseline as a pass -- makes
    the gate silently inert exactly when it is misconfigured, which is when you most need it
    to speak up.
    """


def save_baseline(metrics: EvalMetrics, path: Path | None = None) -> Path:
    target = path or DEFAULT_BASELINE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question_count": metrics.question_count,
        "metrics": metrics.gated_scores(),
        "note": (
            "Recorded by `rag-assistant eval --record-baseline`. Re-record deliberately when "
            "a change is a genuine improvement; never to make a failing gate pass."
        ),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def load_baseline(path: Path | None = None) -> dict[str, float]:
    source = path or DEFAULT_BASELINE_PATH
    if not source.exists():
        raise BaselineNotFound(
            f"No eval baseline at {source}. Record one against a known-good build with:\n"
            f"    uv run rag-assistant eval --limit <n> --record-baseline\n"
            f"This needs real API keys -- the eval harness runs the actual graph."
        )
    return json.loads(source.read_text())["metrics"]


def compare(
    metrics: EvalMetrics,
    baseline: dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> BaselineComparison:
    """Compares current scores against the baseline.

    Only metrics present in *both* are compared. A newly added metric therefore can't fail a
    build against a baseline recorded before it existed -- it starts reporting, and gates
    from the next recorded baseline onward.
    """
    current = metrics.gated_scores()
    return BaselineComparison(
        comparisons=[
            MetricComparison(
                name=name,
                baseline=baseline[name],
                current=current[name],
                tolerance=tolerance,
            )
            for name in sorted(current)
            if name in baseline
        ]
    )
