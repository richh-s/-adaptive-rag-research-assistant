from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag_assistant.graph.build_graph import build_graph
from rag_assistant.logging_conf import configure_logging
from rag_assistant.schemas.api import ResearchRequest, ResearchResponse, StreamEvent

configure_logging()

app = FastAPI(
    title="Adaptive RAG Research Assistant",
    description=(
        "Autonomously routes a research question to local retrieval, web search, or both, "
        "fuses and grades the results, and synthesizes a cited, transparency-reported answer."
    ),
    version="0.1.0",
)

# Allows the Vite dev server (and any local frontend build served on another port) to call
# this API directly from the browser during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Building the graph only wires node functions together -- no API calls happen until
# `.invoke(...)` runs, so one compiled graph can be safely reused across every request.
_graph = build_graph()

_RECURSION_LIMIT = 50


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    try:
        result = _graph.invoke(
            {"question": request.question}, config={"recursion_limit": _RECURSION_LIMIT}
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ResearchResponse(
        question=request.question,
        report=result["research_report"],
        route=result.get("route"),
        confidence_score=result.get("confidence_score"),
    )
