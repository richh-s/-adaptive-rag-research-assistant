from typing import Literal

from pydantic import BaseModel


class ResearchRequest(BaseModel):
    """POST /research request body."""

    question: str


class ResearchResponse(BaseModel):
    """POST /research response body."""

    question: str
    report: str
    route: str | None
    confidence_score: float | None


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
    detail: str | None = None
