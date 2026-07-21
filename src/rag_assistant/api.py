import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag_assistant.graph.build_graph import build_graph
from rag_assistant.graph.research_summary import build_research_summary
from rag_assistant.logging_conf import configure_logging
from rag_assistant.schemas.api import ResearchRequest, ResearchResponse, StreamEvent

configure_logging()
logger = logging.getLogger(__name__)

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

# Mirrors the `Annotated[..., operator.add]` fields in `graph/state.py`. `stream_mode="updates"`
# yields each node invocation's own delta (e.g. one retrieve_vector call per sub-query), so
# reassembling a final state here must concatenate these keys the same way LangGraph's reducer
# does internally -- a plain dict.update() would silently keep only the last invocation's delta.
_ACCUMULATING_KEYS = {"vector_results", "bm25_results", "web_results", "node_timings"}

# Human-readable progress label per graph node, shown to the client as each node completes.
# `dispatch_retrieval`'s `Send` fan-out means retrieve_vector/retrieve_bm25/web_search can
# each fire multiple times (once per sub-query) and `fuse_results` can fire twice (corrective
# retry loop), so this lookup must stay stateless per event rather than assume 1 event/node.
NODE_MESSAGES: dict[str, str] = {
    "route_query": "Routing question...",
    "decompose_query": "Decomposing into sub-queries...",
    "retrieve_vector": "Retrieving from local knowledge base...",
    "retrieve_bm25": "Searching local corpus by keyword...",
    "web_search": "Searching the web...",
    "fuse_results": "Fusing retrieved results...",
    "grade_and_score": "Grading relevance and confidence...",
    "corrective_web_search": "Confidence low, running corrective web search...",
    "synthesize_answer": "Synthesizing answer...",
    "format_report": "Formatting report...",
}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    try:
        result = _graph.invoke(
            {"question": request.question}, config={"recursion_limit": _RECURSION_LIMIT}
        )
    except Exception as exc:
        logger.exception("research failed for question=%r", request.question)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ResearchResponse(
        question=request.question,
        report=result["research_report"],
        route=result.get("route"),
        confidence_score=result.get("confidence_score"),
        summary=build_research_summary(result),
    )


async def _stream_research_events(question: str) -> AsyncIterator[str]:
    # Once this generator has started, the response is already HTTP 200 with headers flushed
    # -- there is no way to surface an HTTP error status mid-stream. Every failure, including
    # ones from deep inside a graph node (e.g. quota exhaustion), must degrade to a "type":
    # "error" SSE frame instead of propagating and truncating the connection.
    try:
        final_state: dict = {}
        async for update in _graph.astream(
            {"question": question},
            config={"recursion_limit": _RECURSION_LIMIT},
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                for key, value in node_output.items():
                    if key in _ACCUMULATING_KEYS:
                        final_state[key] = final_state.get(key, []) + value
                    else:
                        final_state[key] = value
                event = StreamEvent(
                    type="progress",
                    node=node_name,
                    message=NODE_MESSAGES.get(node_name, node_name),
                )
                yield f"data: {event.model_dump_json()}\n\n"

        done_event = StreamEvent(
            type="done",
            report=final_state.get("research_report", ""),
            route=final_state.get("route"),
            confidence_score=final_state.get("confidence_score"),
            summary=build_research_summary(final_state),
        )
        yield f"data: {done_event.model_dump_json()}\n\n"
    except Exception as exc:
        logger.exception("research_stream failed for question=%r", question)
        detail = str(exc) or f"{type(exc).__name__} (no further detail from the underlying service)"
        error_event = StreamEvent(type="error", detail=detail)
        yield f"data: {error_event.model_dump_json()}\n\n"


@app.post("/research/stream")
async def research_stream(request: ResearchRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_research_events(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
