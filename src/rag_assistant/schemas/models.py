import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _decode_stringified_list(value: Any, field: str) -> Any:
    """Undoes a tool-calling quirk: models occasionally return a list argument as a JSON
    *string* -- sometimes the bare list, sometimes the whole object re-encoded, as in
    `grades='{"grades": [...]}'`. Seen from Claude on real traffic, where it failed decomposition
    outright and silently disabled grading. Anything that doesn't decode to the expected shape
    is passed through untouched so validation still rejects it."""
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    if isinstance(decoded, dict) and field in decoded:
        decoded = decoded[field]
    return decoded if isinstance(decoded, list) else value


class RetrievedDoc(BaseModel):
    """Normalized shape for a single retrieved piece of content, whether it came from the
    local vector store or a web search — lets downstream nodes (fusion, grading, synthesis)
    treat both sources uniformly."""

    content: str
    metadata: dict = {}
    source_id: str
    score: float | None = None


class RouteDecision(BaseModel):
    """Structured output for the router node: which retrieval path(s) the question needs."""

    route: Literal["vector", "web", "both", "none"] = Field(
        description=(
            "'vector' if the local knowledge base likely has this (judge by its listed "
            "contents), 'web' if it needs current/recent information, 'both' if it needs "
            "both, 'none' if it's general knowledge that needs no retrieval at all."
        )
    )
    reasoning: str = Field(description="One sentence explaining the routing choice.")


class CondensedQuestion(BaseModel):
    """Structured output for the follow-up condensation node."""

    standalone_question: str = Field(
        description=(
            "The user's latest message rewritten as one fully self-contained question, with "
            "every pronoun/reference resolved from the conversation. Unchanged if it was "
            "already self-contained."
        )
    )


class SubQueries(BaseModel):
    """Structured output for the decomposition node."""

    sub_queries: list[str] = Field(
        description=(
            "2-5 focused, self-contained sub-questions that together cover the original "
            "question. If the question is already simple/atomic, a single-element list "
            "containing the original question, unchanged."
        )
    )

    @field_validator("sub_queries", mode="before")
    @classmethod
    def _decode(cls, value: Any) -> Any:
        return _decode_stringified_list(value, "sub_queries")


class SubQueryResult(BaseModel):
    """One retrieval path's results for one sub-query -- the unit that Send-based fan-out
    nodes return, later merged across all sub-queries and both paths via `operator.add`."""

    sub_query: str
    docs: list[RetrievedDoc] = []


class FusedDocument(BaseModel):
    """A document after Reciprocal Rank Fusion -- deduplicated across every ranked list it
    appeared in, carrying a single fused score reflecting how consistently high it ranked."""

    content: str
    metadata: dict = {}
    source_id: str
    rrf_score: float


class DocGrade(BaseModel):
    """Corrective-RAG style relevance grade for a single document."""

    relevant: bool = Field(
        description="Whether this document meaningfully helps answer the question."
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 (irrelevant) to 1.0 (highly relevant).",
    )


class DocGradeBatch(BaseModel):
    """Structured output for grading every fused document in a single LLM call."""

    grades: list[DocGrade] = Field(
        description="Exactly one grade per document, in the same order the documents were given."
    )

    @field_validator("grades", mode="before")
    @classmethod
    def _decode(cls, value: Any) -> Any:
        return _decode_stringified_list(value, "grades")


class Citation(BaseModel):
    """One deterministic citation marker, assigned in fused rank order."""

    marker: str
    source_id: str


class GoldenQuestion(BaseModel):
    """One hand-authored row of the eval golden dataset (data/golden_eval/dataset.jsonl)."""

    question: str
    ground_truth: str
    reference_contexts: list[str]
    expected_route: Literal["vector", "web", "both", "none"]
    # Every route a careful reviewer would accept for this question, including
    # `expected_route`. Empty means "only `expected_route`", which keeps rows written before
    # this field valid.
    #
    # Routing is genuinely ambiguous for a real share of questions -- asking for a company's
    # private GPU count can defensibly go to `web` alone or to `both` -- and a dataset that
    # asserts one answer measures its own labelling as much as the router. The gate worked
    # regardless, because routing is deterministic at temperature 0 and a regression moves
    # the number well past tolerance; what it could not do was report an absolute figure
    # anyone should quote. This is what makes `route_accuracy` mean "the router chose
    # defensibly" rather than "the router chose what one reviewer wrote down".
    #
    # Widened only where a second route is genuinely defensible, never to make a failing gate
    # pass -- a dataset edited until the number looks good measures nothing at all.
    acceptable_routes: list[Literal["vector", "web", "both", "none"]] = []
    expected_sources: list[str]
    # What this row is testing. Defaulted so rows written before categories existed stay
    # valid, and so adding a category later never invalidates the dataset.
    #   factual      -- answerable from the local corpus; the happy path
    #   multi_hop    -- needs decomposition; one retrieval pass shouldn't cover it
    #   unanswerable -- deliberately outside the corpus: the system must abstain rather than
    #                   confabulate, which is the failure mode no happy-path row can catch
    #   current      -- needs fresh information the corpus can't have; should route to web
    #   no_retrieval -- general knowledge; retrieving at all is wasted spend
    category: Literal["factual", "multi_hop", "unanswerable", "current", "no_retrieval"] = "factual"
