import operator
from typing import Annotated, Literal, TypedDict

from rag_assistant.schemas.models import Citation, DocGrade, FusedDocument, SubQueryResult


class ResearchState(TypedDict):
    """Shared state threaded through every node in the graph. Each node reads what it
    needs and returns a partial dict of updates; LangGraph merges that into this state."""

    question: str

    # conversational memory -- Concept: follow-up condensation. `chat_history` is the prior
    # turns supplied by the client ({"role": "user"|"assistant", "content": str} dicts);
    # when it's non-empty, `condense_question` may rewrite `question` into a self-contained
    # form and preserve what the user literally typed in `original_question` (None when no
    # rewrite happened, so its presence doubles as the "was this condensed?" flag).
    chat_history: list[dict]
    original_question: str | None

    # Set once by the API layer from the request's trace_id (see api.py's TraceIdMiddleware)
    # and read by every node's timing wrapper (build_graph.py's `_timed`) so its per-node log
    # line can be correlated back to the request, independent of contextvar propagation through
    # LangGraph's own task/thread scheduling.
    trace_id: str | None

    # Which tenant is asking. Set by the API layer from the authenticated key (see auth.py);
    # defaults to the public tenant for the CLI, MCP server, and eval harness, which have no
    # request context. Every retrieval path filters on it -- see ingestion/ownership.py.
    owner: str

    # Metadata filters narrowing local retrieval (see schemas/api.RetrievalFilters). Carried
    # as the model itself rather than a dict so the retrieval nodes get validated bounds.
    filters: object | None

    # routing -- Concept: Agentic/Self-RAG
    route: Literal["vector", "web", "both", "none"] | None
    route_reasoning: str | None

    # decomposition -- Concept: query decomposition
    sub_queries: list[str]

    # `operator.add` reducer: each Send-based retrieve_vector/retrieve_bm25/web_search
    # invocation contributes a one-element list for its sub-query, and LangGraph concatenates
    # them all here instead of the default "last write wins" behavior.
    vector_results: Annotated[list[SubQueryResult], operator.add]
    bm25_results: Annotated[list[SubQueryResult], operator.add]
    web_results: Annotated[list[SubQueryResult], operator.add]

    # fusion -- Concept: RAG Fusion. Written once by the `fuse_results` join point, so no
    # reducer needed here.
    fused_documents: list[FusedDocument]

    # confidence / correction -- Concept: Corrective-RAG
    doc_grades: list[DocGrade]
    confidence_score: float
    needs_correction: bool
    correction_attempted: bool

    final_answer: str
    citations: list[Citation]

    # How many fused documents the synthesis context budget dropped (see
    # graph/context_budget.py). Surfaced in the research summary so a truncated answer is
    # visible as a budget decision rather than looking like retrieval simply found less.
    context_documents_dropped: int

    # report formatting -- Concept: transparency reporting
    research_report: str

    errors: list[str]

    # observability -- one entry per node invocation (Send fan-out nodes like retrieve_vector
    # contribute one entry per sub-query), used to build the explainability/latency panel.
    node_timings: Annotated[list[dict], operator.add]
