from typing import Literal

from pydantic import BaseModel


class ResearchRequest(BaseModel):
    """POST /research request body."""

    question: str


class RetrievalCounts(BaseModel):
    """Documents retrieved per source, before fusion/dedup."""

    vector: int
    bm25: int
    web: int


class NodeLatency(BaseModel):
    """One node invocation's wall-clock time. Send-fanned nodes (retrieve_vector/
    retrieve_bm25/web_search) appear once per sub-query, not once per node type."""

    node: str
    latency_ms: float


class ResearchSummary(BaseModel):
    """Structured "how this was researched" data for the explainability panel -- the same
    facts already narrated in the markdown report's transparency section, plus latency,
    exposed as typed fields so the frontend can render a dedicated panel instead of parsing
    prose."""

    route: str | None
    sub_queries: list[str]
    retrieval_counts: RetrievalCounts
    fused_document_count: int
    confidence_score: float | None
    correction_attempted: bool
    node_latencies_ms: list[NodeLatency]
    total_latency_ms: float


class ResearchResponse(BaseModel):
    """POST /research response body."""

    question: str
    report: str
    route: str | None
    confidence_score: float | None
    summary: ResearchSummary | None = None


class StreamEvent(BaseModel):
    """One SSE frame from POST /research/stream. `type` discriminates which fields are
    populated: "progress" carries node/message, "done" carries the final report fields,
    "error" carries detail. Kept as one flat model (rather than a Union) since the frontend
    parses raw JSON by hand and checks `type` first regardless."""

    type: Literal["progress", "done", "error"]
    node: str | None = None
    message: str | None = None
    report: str | None = None
    route: str | None = None
    confidence_score: float | None = None
    summary: ResearchSummary | None = None
    detail: str | None = None
