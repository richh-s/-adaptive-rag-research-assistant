import asyncio
import logging
import signal
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from rag_assistant.config import get_settings
from rag_assistant.graph.build_graph import build_graph
from rag_assistant.graph.research_summary import build_research_summary
from rag_assistant.logging_conf import configure_logging
from rag_assistant.readiness import check_chroma, check_tavily
from rag_assistant.schemas.api import ResearchRequest, ResearchResponse, StreamEvent
from rag_assistant.tracing import get_trace_id, new_trace_id, trace_id_var

configure_logging()
logger = logging.getLogger(__name__)

# Graceful shutdown: SIGTERM sets `_shutdown_event`, which every active SSE stream polls each
# loop iteration so it can send a "close" frame and return cleanly instead of being cut off
# when uvicorn's own shutdown grace period expires. `_active_streams` is a WeakSet used purely
# for observability (how many connections were live at shutdown) -- the actual signal used to
# unblock streams is the Event, since you can't push data into a running generator from outside.
_shutdown_event = asyncio.Event()


class _StreamConnection:
    """Marker object representing one open SSE stream; only its presence in `_active_streams`
    (not any attribute on it) matters."""


_active_streams: "weakref.WeakSet[_StreamConnection]" = weakref.WeakSet()


def _handle_sigterm() -> None:
    logger.info("SIGTERM received; signaling %d active stream(s) to close", len(_active_streams))
    _shutdown_event.set()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


app = FastAPI(
    title="Adaptive RAG Research Assistant",
    description=(
        "Autonomously routes a research question to local retrieval, web search, or both, "
        "fuses and grades the results, and synthesizes a cited, transparency-reported answer."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)

# Per-IP limiter (rate_limit_rpm) and a second limiter keyed on a constant so its bucket is
# shared across every caller (rate_limit_rpm_global) -- together these cap both "one client
# hammering us" and "aggregate load regardless of client" per the production-readiness spec.
# Limit strings are read from settings on every request (not frozen at import time) so tests
# that override RATE_LIMIT_RPM/RATE_LIMIT_RPM_GLOBAL via env vars take effect.
limiter = Limiter(key_func=get_remote_address)
global_limiter = Limiter(key_func=lambda request: "global")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


def _per_ip_limit() -> str:
    return f"{get_settings().rate_limit_rpm}/minute"


def _global_limit() -> str:
    return f"{get_settings().rate_limit_rpm_global}/minute"

# Allows the Vite dev server (and any local frontend build served on another port) to call
# this API directly from the browser during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TraceIdMiddleware:
    """Raw ASGI middleware (not `BaseHTTPMiddleware`, which buffers/consumes the response body
    in a way that's unsafe for our SSE streams) -- generates one UUID4 per request, stores it
    in `trace_id_var` for the lifetime of the request's task, echoes it back as a response
    header, and logs one structured line per request with route + total latency_ms."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        trace_id = new_trace_id()
        token = trace_id_var.set(trace_id)
        start = time.perf_counter()

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"x-trace-id", trace_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.info(
                "request completed",
                extra={"route": scope.get("path", ""), "latency_ms": latency_ms},
            )
            trace_id_var.reset(token)


app.add_middleware(TraceIdMiddleware)

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


@app.get("/ready")
def ready() -> JSONResponse:
    chroma_ok, chroma_err = check_chroma()
    tavily_ok, tavily_err = check_tavily()
    body = {
        "status": "ok" if chroma_ok and tavily_ok else "unavailable",
        "chroma": {"ok": chroma_ok, "error": chroma_err},
        "tavily": {"ok": tavily_ok, "error": tavily_err},
    }
    return JSONResponse(content=body, status_code=200 if chroma_ok and tavily_ok else 503)


@app.post("/research", response_model=ResearchResponse)
@limiter.limit(_per_ip_limit)
@global_limiter.limit(_global_limit)
def research(request: Request, body: ResearchRequest) -> ResearchResponse:
    try:
        result = _graph.invoke(
            {"question": body.question, "trace_id": get_trace_id()},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
    except Exception as exc:
        logger.exception("research failed for question=%r", body.question)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ResearchResponse(
        question=body.question,
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
    timeout_seconds = get_settings().graph_timeout_seconds
    connection = _StreamConnection()
    _active_streams.add(connection)
    try:
        final_state: dict = {}
        graph_iter = _graph.astream(
            {"question": question, "trace_id": get_trace_id()},
            config={"recursion_limit": _RECURSION_LIMIT},
            stream_mode="updates",
        ).__aiter__()
        # Bounds total time spent waiting on the graph, not any single node -- each
        # `__anext__()` gets whatever's left of the overall budget, so a hang anywhere
        # (a slow LLM call, a stuck retry loop) still surfaces an error frame and closes
        # the connection instead of leaving the client waiting indefinitely.
        deadline = time.monotonic() + timeout_seconds
        while True:
            if _shutdown_event.is_set():
                yield f"data: {StreamEvent(type='close').model_dump_json()}\n\n"
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"research timed out after {timeout_seconds}s")
            try:
                update = await asyncio.wait_for(graph_iter.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except TimeoutError:
                raise TimeoutError(f"research timed out after {timeout_seconds}s") from None

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
    finally:
        _active_streams.discard(connection)


@app.post("/research/stream")
@limiter.limit(_per_ip_limit)
@global_limiter.limit(_global_limit)
async def research_stream(request: Request, body: ResearchRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_research_events(body.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
