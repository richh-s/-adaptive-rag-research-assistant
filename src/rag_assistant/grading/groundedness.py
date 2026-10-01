"""Checking the answer against the documents it was written from.

Everything else in this pipeline grades *retrieval*. `grade_and_score` judges whether the
fused documents are relevant to the question, and its output drives the confidence score the
transparency panel reports. That number answers "did retrieval find the right material", and
it is read -- by anyone looking at a UI that prints it next to an answer -- as "is this answer
right". Those are different questions, and the gap between them is precisely the failure mode
RAG exists to prevent: the retrieval can be perfect and the synthesis can still assert
something the documents never said.

Nothing in the pipeline closed that gap at request time. RAGAS faithfulness measures it, but
only offline, only on the golden dataset, and only when `--llm-judge` is passed -- so in
production the claim "this answer is grounded" rested on the synthesis prompt asking the model
nicely.

This is the complement: one structured call, after synthesis, that decomposes the answer into
claims and asks whether the numbered context supports each one. It reports rather than
intervenes. The alternatives were considered and are worse:

* **Regenerating on a low score** doubles cost and latency on exactly the questions the corpus
  is thinnest on, and there is no reason to think the second attempt is better grounded than
  the first -- the model already had the same documents.
* **Stripping unsupported sentences** edits an answer whose fluency depends on them, turning a
  hedged, readable paragraph into a disjointed one, on the word of a check that is itself a
  model call and can be wrong.

So the score is surfaced (research summary, Prometheus, a caveat on the report) and the reader
decides. That is the same choice `content_trust.scan_for_injection` makes and for the same
reason: a signal that is acted on silently is a signal nobody can audit.

The honest boundary: this is a model checking a model, both of which may share a blind spot,
and it verifies *support*, not truth. A claim supported by a document that is itself wrong
scores as grounded, correctly -- the corpus is the ground truth here, and whether the corpus
deserves to be is a different question that no runtime check can answer.
"""

import logging
from dataclasses import dataclass, field

from rag_assistant import metrics
from rag_assistant.config import get_settings
from rag_assistant.content_trust import build_untrusted_context, new_nonce
from rag_assistant.llm import get_structured_llm
from rag_assistant.prompts.groundedness_prompt import GROUNDEDNESS_PROMPT
from rag_assistant.schemas.models import FusedDocument, GroundednessReport

logger = logging.getLogger(__name__)


@dataclass
class GroundednessResult:
    """`checked` is what separates "verified and fine" from "not verified".

    Both produce no unsupported claims, and collapsing them would report a failed or disabled
    check as a clean bill of health -- the same reason `check_embedding_model` distinguishes
    "matches" from "cannot verify".
    """

    checked: bool = False
    score: float | None = None
    claims_checked: int = 0
    unsupported_claims: list[str] = field(default_factory=list)


async def verify_answer(
    question: str, answer: str, documents: list[FusedDocument]
) -> GroundednessResult:
    """Fraction of the answer's factual claims the documents actually support.

    Returns an unchecked result -- never a score of 0.0 -- when the check cannot run. A
    provider hiccup is not evidence that an answer was fabricated, and scoring it as one
    would put a number in the transparency panel that means the opposite of what it says.
    """
    if not documents or not (answer or "").strip():
        return GroundednessResult()

    nonce = new_nonce()
    context, _ = build_untrusted_context(
        [(doc.source_id, doc.content) for doc in documents], nonce=nonce
    )
    llm = get_structured_llm(
        GroundednessReport, timeout=get_settings().groundedness_request_timeout_seconds
    )
    try:
        report: GroundednessReport = await llm.ainvoke(
            GROUNDEDNESS_PROMPT.format(question=question, answer=answer, context=context)
        )
    except Exception:
        # Mirrors the relevance grader's degrade-to-trust: a structured-output failure here
        # must not fail a request whose answer is already written.
        logger.warning(
            "Groundedness check failed for question=%r; answer not verified",
            question,
            exc_info=True,
        )
        metrics.record_groundedness("unavailable")
        return GroundednessResult()

    claims = report.claims
    if not claims:
        # No factual claims to support. An abstention is the common case, and scoring it 0.0
        # would flag the system's most correct behaviour as its least grounded.
        metrics.record_groundedness("no_claims")
        return GroundednessResult(checked=True, score=None, claims_checked=0)

    unsupported = [c.claim for c in claims if not c.supported]
    score = (len(claims) - len(unsupported)) / len(claims)
    metrics.record_groundedness("checked", score=score)
    if unsupported:
        logger.warning(
            "answer contains claims the retrieved context does not support",
            extra={
                "groundedness_score": round(score, 3),
                "unsupported_claim_count": len(unsupported),
            },
        )
    return GroundednessResult(
        checked=True,
        score=score,
        claims_checked=len(claims),
        unsupported_claims=unsupported,
    )
