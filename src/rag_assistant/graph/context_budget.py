"""Bounding how much retrieved context reaches the synthesis prompt.

Fusion returns however many documents the retrieval paths happened to produce -- one ranked
list per (sub-query x source), so a 4-sub-query question on route "both" can fuse dozens of
chunks, and a corrective pass adds more. Synthesis then concatenated all of them. On a
five-file corpus that is invisible; on a real one it is a prompt that grows without limit,
which fails in three ways: cost scales with retrieval breadth rather than with the question,
latency follows it, and past the model's context window the call simply errors -- at the very
end of the pipeline, after every retrieval and grading call has already been paid for.

Documents arrive here already ordered best-first (RRF consensus, then grade-informed rerank),
so the budget is spent greedily down that ranking: it is the cheapest possible policy and it
drops exactly the documents the rest of the pipeline already judged least useful.

Token counts are estimated from character length rather than measured with a real tokenizer.
This project runs against Anthropic, Gemini, and arbitrary self-hosted OpenAI-compatible
servers, which do not share a tokenizer -- an exact count for one is a wrong count for the
others, and pulling in `tiktoken` would buy OpenAI-accurate numbers for providers this app
does not use. The ratio is configurable, and the budget is a safety margin, not an exact fit.
"""

from dataclasses import dataclass

from rag_assistant.schemas.models import FusedDocument


@dataclass
class BudgetedContext:
    """What survived the budget, and what it cost to get there."""

    documents: list[FusedDocument]
    dropped_documents: int
    truncated_documents: int
    estimated_tokens: int


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return int(len(text) / chars_per_token) if chars_per_token > 0 else len(text)


def select_context_documents(
    documents: list[FusedDocument],
    budget_tokens: int,
    chars_per_token: float = 4.0,
) -> BudgetedContext:
    """Takes documents in rank order until the token budget is spent.

    The highest-ranked document is truncated to fit rather than dropped when it alone exceeds
    the budget. Returning nothing would be worse than returning a partial best document:
    synthesis reads an empty context as "retrieval found nothing" and says so, which would be
    a flatly false statement about a retrieval that succeeded and found something too long.

    A non-positive budget disables the cap entirely, which is the escape hatch for anyone
    running a long-context model who would rather not think about this at all.
    """
    if budget_tokens <= 0 or not documents:
        return BudgetedContext(
            documents=list(documents),
            dropped_documents=0,
            truncated_documents=0,
            estimated_tokens=sum(estimate_tokens(d.content, chars_per_token) for d in documents),
        )

    selected: list[FusedDocument] = []
    used = 0
    truncated = 0

    for index, document in enumerate(documents):
        cost = estimate_tokens(document.content, chars_per_token)
        if used + cost <= budget_tokens:
            selected.append(document)
            used += cost
            continue

        # Only the first document is worth truncating. Further down the ranking, a fragment
        # of a lower-ranked chunk is more likely to mislead than to help -- and it would still
        # be assigned a citation marker, inviting the model to cite a passage that was cut off
        # before the part that supported the claim.
        if index == 0:
            remaining_chars = int(budget_tokens * chars_per_token)
            if remaining_chars > 0:
                selected.append(
                    document.model_copy(update={"content": document.content[:remaining_chars]})
                )
                used = budget_tokens
                truncated = 1
        break

    return BudgetedContext(
        documents=selected,
        dropped_documents=len(documents) - len(selected),
        truncated_documents=truncated,
        estimated_tokens=used,
    )
