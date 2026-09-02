import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ALNUM_RE = re.compile(r"\w", re.UNICODE)


class ChatTurn(BaseModel):
    """One prior turn of the conversation, supplied by the client with each request -- the
    server holds no session state, so the client owns the transcript and replays it. For
    assistant turns, clients should send the answer text (not the full markdown report with
    its transparency section) to keep follow-up condensation focused on content."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class RetrievalFilters(BaseModel):
    """Narrows local retrieval before fusion, rather than reranking after it.

    Post-filtering is the tempting shortcut and it is wrong for the same reason it is wrong
    for tenancy: `k` is applied by the store, so filtering the results afterwards silently
    returns fewer documents than asked for -- sometimes none -- and the graph reads that as
    "the corpus has nothing" and falls back to web search. These are pushed into the query.

    Web results are unaffected: a filter on indexed sources has no meaning for a live search.
    """

    sources: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Restrict to these source filenames. Empty means no source restriction.",
    )
    ingested_after: datetime | None = Field(
        None, description="Only documents indexed at or after this instant."
    )
    ingested_before: datetime | None = Field(
        None, description="Only documents indexed at or before this instant."
    )

    def is_empty(self) -> bool:
        return not self.sources and self.ingested_after is None and self.ingested_before is None

    @model_validator(mode="after")
    def _check_range(self) -> "RetrievalFilters":
        if (
            self.ingested_after
            and self.ingested_before
            and self.ingested_after > self.ingested_before
        ):
            raise ValueError("ingested_after must not be later than ingested_before.")
        return self


class ResearchRequest(BaseModel):
    """POST /research request body.

    Conversation modes, in precedence order:
    - `conversation_id` set: the server loads the transcript from its own store, answers
      in-context, and appends the exchange (404 if the id is unknown; `history` is ignored).
    - `conversation_id` unset, `save` true (the default): the server starts a new persisted
      conversation and returns its id in the response for the client to reuse.
    - `save` false: fully stateless -- the optional client-supplied `history` is used for
      condensation/synthesis and nothing is stored.
    """

    question: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    conversation_id: str | None = Field(default=None, max_length=64)
    save: bool = True
    filters: RetrievalFilters = Field(
        default_factory=RetrievalFilters,
        description="Narrows local retrieval. Has no effect on web search results.",
    )

    @field_validator("question")
    @classmethod
    def _clean_and_validate(cls, value: str) -> str:
        stripped = _HTML_TAG_RE.sub("", value).strip()
        if not stripped:
            raise ValueError("question must not be empty")
        alnum_count = len(_ALNUM_RE.findall(stripped))
        if alnum_count / len(stripped) < 0.1:
            raise ValueError("question appears to be gibberish (too few alphanumeric characters)")
        return stripped


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
    # Set only when the question arrived as a follow-up and was rewritten into a standalone
    # form -- lets the UI show "interpreted as: ..." next to what the user literally typed.
    condensed_question: str | None = None
    sub_queries: list[str]
    retrieval_counts: RetrievalCounts
    fused_document_count: int
    # How many fused documents the synthesis context budget dropped (see
    # graph/context_budget.py). Non-zero means the answer was written from a subset of what
    # retrieval found -- worth surfacing, since otherwise a budget decision is
    # indistinguishable from retrieval having found less.
    context_documents_dropped: int = 0
    confidence_score: float | None
    correction_attempted: bool
    node_latencies_ms: list[NodeLatency]
    total_latency_ms: float


class ResearchResponse(BaseModel):
    """POST /research response body. `answer` is the synthesized answer text alone (no
    transparency/citation appendix) -- the piece clients should replay as the assistant's
    turn in `history` on follow-up requests."""

    question: str
    report: str
    answer: str | None = None
    route: str | None
    confidence_score: float | None
    summary: ResearchSummary | None = None
    # The persisted conversation this exchange was appended to (None when save=false).
    # Clients pass it back as `conversation_id` to continue the conversation server-side.
    conversation_id: str | None = None


class IngestUrlRequest(BaseModel):
    """POST /api/v1/ingest/url request body."""

    url: str = Field(..., min_length=1, max_length=2000)

    @field_validator("url")
    @classmethod
    def _must_be_http(cls, value: str) -> str:
        value = value.strip()
        if not value.lower().startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return value


class IngestResponse(BaseModel):
    """POST /api/v1/ingest response body. Indexing runs in a BackgroundTask after this
    response is sent, so this only confirms the upload was validated and persisted to
    `corpus_dir` -- not that embedding/indexing has finished. Poll
    GET /api/v1/ingest/{task_id} with `task_id` for real progress."""

    task_id: str
    filename: str
    original_filename: str
    size_bytes: int
    status: Literal["queued"]
    message: str


class IngestTaskStatus(BaseModel):
    """GET /api/v1/ingest/{task_id} response body. `stage` walks forward through
    queued -> parsing -> indexing -> indexed, or to failed at any point; `error` is only set
    once `stage == "failed"`."""

    task_id: str
    filename: str
    original_filename: str
    stage: Literal["queued", "parsing", "indexing", "indexed", "failed"]
    message: str
    error: str | None = None
    indexed_chunks: int | None = None


class StreamEvent(BaseModel):
    """One SSE frame from POST /research/stream. `type` discriminates which fields are
    populated: "progress" carries node/message, "done" carries the final report fields,
    "error" carries detail. Kept as one flat model (rather than a Union) since the frontend
    parses raw JSON by hand and checks `type` first regardless."""

    type: Literal["progress", "token", "done", "error", "close"]
    node: str | None = None
    message: str | None = None
    # "token" frames carry an incremental piece of the answer as the synthesis LLM generates
    # it; clients append them in order, then replace the whole thing with `report`/`answer`
    # from the "done" frame (which is authoritative -- cached answers emit no tokens at all).
    token: str | None = None
    report: str | None = None
    answer: str | None = None
    route: str | None = None
    confidence_score: float | None = None
    summary: ResearchSummary | None = None
    detail: str | None = None
    conversation_id: str | None = None


class ConversationSummary(BaseModel):
    """One row of GET /api/v1/conversations -- enough for a history sidebar."""

    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


class ConversationMessage(BaseModel):
    """One message of GET /api/v1/conversations/{id}. Assistant messages carry the full
    report and research summary they were rendered with, so reopening a conversation
    restores exactly what the user saw."""

    role: Literal["user", "assistant"]
    content: str
    report: str | None = None
    summary: ResearchSummary | None = None
    created_at: float


class ConversationDetail(BaseModel):
    """GET /api/v1/conversations/{id} response body."""

    id: str
    title: str
    created_at: float
    updated_at: float
    messages: list[ConversationMessage]


class FeedbackRequest(BaseModel):
    """POST /api/v1/feedback body -- one user rating of one answer."""

    question: str = Field(..., min_length=1, max_length=2000)
    rating: Literal["up", "down"]
    conversation_id: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=2000)
    route: str | None = Field(default=None, max_length=32)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)


class FeedbackResponse(BaseModel):
    id: int
    recorded: bool = True


class FeedbackSummary(BaseModel):
    """Aggregate signal, plus the questions worth adding to the eval dataset."""

    up: int
    down: int
    total: int
    satisfaction: float | None
    recent_downvoted_questions: list[str]


class IndexedSource(BaseModel):
    """One file in the knowledge base, as the caller can refer to it in `filters.sources`."""

    source: str
    display_name: str
    chunk_count: int
    owner: str
