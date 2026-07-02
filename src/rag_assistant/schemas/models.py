from typing import Literal

from pydantic import BaseModel, Field


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
            "'vector' if the local knowledge base likely has this (AI company facts as of "
            "early 2025), 'web' if it needs current/recent information, 'both' if it needs "
            "both, 'none' if it's general knowledge that needs no retrieval at all."
        )
    )
    reasoning: str = Field(description="One sentence explaining the routing choice.")


class SubQueries(BaseModel):
    """Structured output for the decomposition node."""

    sub_queries: list[str] = Field(
        description=(
            "2-5 focused, self-contained sub-questions that together cover the original "
            "question. If the question is already simple/atomic, a single-element list "
            "containing the original question, unchanged."
        )
    )


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

    relevant: bool = Field(description="Whether this document meaningfully helps answer the question.")
    score: float = Field(
        ge=0.0, le=1.0, description="Relevance score from 0.0 (irrelevant) to 1.0 (highly relevant)."
    )


class DocGradeBatch(BaseModel):
    """Structured output for grading every fused document in a single LLM call."""

    grades: list[DocGrade] = Field(
        description="Exactly one grade per document, in the same order the documents were given."
    )


class Citation(BaseModel):
    """One deterministic citation marker, assigned in fused rank order."""

    marker: str
    source_id: str
