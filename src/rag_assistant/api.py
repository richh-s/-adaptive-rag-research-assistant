import asyncio
import hashlib
import json
import logging
import re
import signal
import time
import uuid
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from rag_assistant import auth, metrics
from rag_assistant.config import get_settings
from rag_assistant.conversations import store as conversations
from rag_assistant.graph.build_graph import build_graph
from rag_assistant.graph.research_summary import build_research_summary
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.loaders import SUPPORTED_SUFFIXES
from rag_assistant.ingestion.ownership import owner_corpus_dir
from rag_assistant.ingestion.tasks import create_task, get_task, update_task
from rag_assistant.ingestion.url_fetch import UrlIngestError, fetch_page, page_to_markdown
from rag_assistant.logging_conf import configure_logging
from rag_assistant.readiness import (
    check_chroma,
    check_embeddings,
    check_local_llm,
    check_web_search,
)
from rag_assistant.schemas.api import (
    ConversationDetail,
    ConversationMessage,
    ConversationSummary,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSummary,
    IngestResponse,
    IngestTaskStatus,
    IngestUrlRequest,
    ResearchRequest,
    ResearchResponse,
    StreamEvent,
)
from rag_assistant.tracing import get_trace_id, new_trace_id, trace_id_var

configure_logging()
logger = logging.getLogger(__name__)

# Error tracking is opt-in: a blank SENTRY_DSN (the default) means no Sentry import, no
# network calls, no behavior change -- set the DSN in production and unhandled exceptions
# (including ones inside graph nodes) get captured with the request's context.
if get_settings().sentry_dsn:
    import sentry_sdk

    sentry_sdk.init(
        dsn=get_settings().sentry_dsn,
        environment=get_settings().app_env,
        traces_sample_rate=get_settings().sentry_traces_sample_rate,
    )

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
    # Registering this handler REPLACED uvicorn's own SIGTERM handling, so without forwarding,
    # a SIGTERM would leave the process alive but permanently poisoned: `_shutdown_event` never
    # clears, so every future /research/stream instantly emits a "close" frame while /health
    # keeps answering 200 -- a half-dead server. Re-raise as SIGINT (whose uvicorn handler we
    # did not touch) so uvicorn still runs its normal graceful shutdown and actually exits.
    signal.raise_signal(signal.SIGINT)


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


# Per-caller limiter (rate_limit_rpm) and a second limiter keyed on a constant so its bucket
# is shared across every caller (rate_limit_rpm_global) -- together these cap both "one client
# hammering us" and "aggregate load regardless of client" per the production-readiness spec.
# Limit strings are read from settings on every request (not frozen at import time) so tests
# that override RATE_LIMIT_RPM/RATE_LIMIT_RPM_GLOBAL via env vars take effect.
#
# Caller identity: authenticated requests are keyed by (a hash of) their API key, so each
# tenant gets its own budget regardless of network path; anonymous requests fall back to
# client IP. Behind a proxy/load balancer the IP is only meaningful when uvicorn runs with
# --proxy-headers and --forwarded-allow-ips (see Dockerfile CMD) -- without that, every
# visitor arrives as the LB's address and shares one bucket.
def _caller_identity(request: Request) -> str:
    key = auth.extract_key({k.lower(): v for k, v in request.scope.get("headers", [])})
    if key:
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return get_remote_address(request)


limiter = Limiter(key_func=_caller_identity)
global_limiter = Limiter(key_func=lambda request: "global")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


def _per_ip_limit(key: str) -> str:
    """Per-caller limit, honouring a key's own override when it has one.

    slowapi passes the bucket key to a provider that declares a `key` parameter -- that key is
    a fingerprint, never the secret, so the override lookup goes through fingerprints too.
    """
    override = auth.rate_limit_for_identity(key)
    return f"{override or get_settings().rate_limit_rpm}/minute"


def _global_limit() -> str:
    return f"{get_settings().rate_limit_rpm_global}/minute"


# Allows the Vite dev server (and any local frontend build served on another port) to call
# this API directly from the browser during development.
# Origins come from CORS_ALLOW_ORIGINS (see config.py) rather than being hardcoded: the
# defaults cover the Vite dev server, the single-container deploy needs none of them because
# it serves the frontend same-origin, and a split deploy (UI on Vercel, API on Render) is a
# config change instead of a code change. Registering the middleware unconditionally -- with
# an empty list it simply matches no origin -- keeps one code path for every deployment shape.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id"],
)


class ObservabilityMiddleware:
    """Raw ASGI middleware (not `BaseHTTPMiddleware`, which buffers/consumes the response body
    in a way that's unsafe for our SSE streams) carrying all three per-request observability
    concerns: it generates one UUID4 trace ID and stores it in `trace_id_var` for the lifetime
    of the request's task, echoes it back as a response header, logs one structured line per
    request, and records the Prometheus request counter/histogram.

    All three share one timer and one wrapper rather than stacking separate middlewares --
    a second layer would re-wrap `send` for no reason and report a slightly different latency
    than the log line, which is exactly the kind of discrepancy that wastes an incident.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        trace_id = new_trace_id()
        token = trace_id_var.set(trace_id)
        start = time.perf_counter()
        # Captured from the response-start message: a request whose connection drops before
        # any response is sent never sets this, and 499 (nginx's client-closed convention)
        # keeps those out of the 5xx bucket where they would look like server errors.
        status_code = 499

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message.setdefault("headers", [])
                headers.append((b"x-trace-id", trace_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            logger.info(
                "request completed",
                extra={"route": scope.get("path", ""), "latency_ms": round(elapsed * 1000, 1)},
            )
            # Metrics must never break a request that otherwise succeeded, and this runs in a
            # `finally` that is also unwinding whatever exception the app may have raised.
            try:
                metrics.observe_request(scope, status_code, elapsed)
            except Exception:
                logger.warning("failed to record request metrics", exc_info=True)
            trace_id_var.reset(token)


class AuthMiddleware:
    """Raw ASGI middleware (same rationale as TraceIdMiddleware: BaseHTTPMiddleware buffers
    SSE responses). Guards every data/LLM endpoint; liveness (/health, /ready), the API docs,
    and the static frontend stay open. With API_KEYS unset this resolves every request to the
    "public" tenant and never rejects -- open demo mode. OPTIONS passes through so CORS
    preflights (which never carry credentials) reach the CORS layer."""

    # /metrics rides along: with auth enabled a scraper must present a key like any other
    # client, and with auth disabled (open demo) it stays reachable, same as everything else.
    PROTECTED_PREFIXES = ("/research", "/api/v1/", "/metrics")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.startswith(self.PROTECTED_PREFIXES):
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        method = scope["method"]

        async def reject(status: int, detail: str) -> None:
            response_headers = [(b"content-type", b"application/json")]
            if status == 401:
                response_headers.append((b"www-authenticate", b"Bearer"))
            await send(
                {"type": "http.response.start", "status": status, "headers": response_headers}
            )
            await send(
                {"type": "http.response.body", "body": json.dumps({"detail": detail}).encode()}
            )

        if not auth.auth_enabled():
            # Open demo mode: every request is the public tenant and nothing is rejected.
            await self.app(scope, receive, send)
            return

        record = auth.resolve_key(auth.extract_key(headers))
        if record is None:
            auth.audit("auth rejected", path=path, method=method, outcome="invalid_key")
            await reject(401, "Missing, invalid, or expired API key.")
            return

        owner_token = auth.owner_var.set(record.owner)
        key_token = auth.api_key_var.set(record)
        try:
            # 403 rather than 401: the credential is valid, it simply isn't allowed to do
            # this. Returning 401 would tell a read-only client to go re-authenticate, which
            # it cannot fix by presenting the same key again.
            needed = auth.required_scope(method, path)
            if not record.has_scope(needed):
                auth.audit("auth forbidden", path=path, method=method, outcome=f"missing:{needed}")
                await reject(403, f"This API key lacks the {needed!r} scope.")
                return
            auth.audit("auth accepted", path=path, method=method, outcome="ok")
            await self.app(scope, receive, send)
        finally:
            auth.api_key_var.reset(key_token)
            auth.owner_var.reset(owner_token)


app.add_middleware(AuthMiddleware)
app.add_middleware(ObservabilityMiddleware)

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
    "condense_question": "Resolving follow-up references...",
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


if get_settings().metrics_enabled:

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        """Prometheus exposition. Registered conditionally (METRICS_ENABLED) rather than
        always-on-and-404ing, so a scraper pointed at a deployment that disabled metrics gets
        an unambiguous 404 instead of an empty 200 that looks like a healthy zero."""
        payload, content_type = metrics.render()
        return Response(content=payload, media_type=content_type)


@app.get("/ready")
def ready() -> JSONResponse:
    chroma_ok, chroma_err = check_chroma()
    web_search_ok, web_search_err = check_web_search()
    # Part of the verdict, unlike local_llm below: a replica whose configured embedding model
    # doesn't match its index cannot serve a correct answer, only a confident wrong one, so
    # it should be pulled from the load balancer rather than merely reported on.
    embeddings_ok, embeddings_err = check_embeddings()
    # Reported but deliberately NOT part of the ready/unavailable verdict: an unreachable
    # self-hosted endpoint is a cost and latency regression (every call falls through to
    # Anthropic), not an outage, so it shouldn't pull a healthy replica out of a load
    # balancer -- while still being visible to whoever is looking at why the bill moved.
    local_llm_ok, local_llm_err = check_local_llm()
    ready_ok = chroma_ok and web_search_ok and embeddings_ok
    body = {
        "status": "ok" if ready_ok else "unavailable",
        "chroma": {"ok": chroma_ok, "error": chroma_err},
        "embeddings": {"ok": embeddings_ok, "error": embeddings_err},
        "web_search": {"ok": web_search_ok, "error": web_search_err},
        "local_llm": {"ok": local_llm_ok, "error": local_llm_err},
    }
    return JSONResponse(content=body, status_code=200 if ready_ok else 503)


# Generous but bounded -- a stray multi-hundred-page PDF (or a client sending garbage)
# shouldn't be able to fill the disk or tie up a background worker indefinitely.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_UNSAFE_FILENAME_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_stem(filename: str) -> str:
    """Strips everything but alphanumerics/`_`/`-` from the uploaded filename's stem so it
    can't path-traverse (`../../etc`) or otherwise inject path separators into corpus_dir."""
    stem = Path(filename).stem
    cleaned = _UNSAFE_FILENAME_CHARS_RE.sub("_", stem).strip("_")
    return cleaned or "upload"


def _run_ingest_in_background(trace_id: str, task_id: str, owner: str) -> None:
    """Runs in FastAPI's threadpool after the response has already been sent (see
    BackgroundTasks below) -- exceptions here would otherwise vanish silently, so they're
    caught, logged, and reflected onto the task record rather than left to crash the worker
    thread unobserved. `build_index()`'s `on_stage` hook drives the "parsing"/"indexing"
    transitions; this function only owns the terminal "indexed"/"failed" transition, since it's
    the one place that knows whether the whole job actually succeeded.
    """
    token = trace_id_var.set(trace_id)
    update_task(task_id, stage="parsing", message="Starting ingestion...")
    try:
        # Scoped to the uploading tenant: their upload should cost their corpus, not a
        # scan of every other tenant's documents (see ingestion/loaders.iter_corpus_files).
        result = build_index(
            owner=owner,
            on_stage=lambda stage, message: update_task(task_id, stage=stage, message=message),
        )
        logger.info(
            "background ingestion complete",
            extra={
                "indexed_chunks": result.indexed_chunks,
                "changed_files": result.changed_files,
                "skipped_files": result.skipped_files,
                "removed_files": result.removed_files,
            },
        )
        if result.changed_files == 0 and result.removed_files == 0:
            message = "No changes detected -- file content matched what was already indexed."
        else:
            message = (
                f"Indexed {result.indexed_chunks} chunk(s) from {result.changed_files} "
                f"file(s); local search is up to date."
            )
        update_task(task_id, stage="indexed", message=message, indexed_chunks=result.indexed_chunks)
    except Exception as exc:
        logger.exception("background ingestion failed")
        update_task(task_id, stage="failed", message="Indexing failed.", error=str(exc))
    finally:
        trace_id_var.reset(token)


@app.post("/api/v1/ingest", response_model=IngestResponse, status_code=202)
@limiter.limit(_per_ip_limit)
@global_limiter.limit(_global_limit)
async def ingest_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> IngestResponse:
    """Accepts one corpus file (.md/.txt/.pdf), persists it into `corpus_dir`, and schedules
    a background re-index. The file must be written to disk *before* this handler returns --
    FastAPI closes and deletes `UploadFile`'s underlying temp file as soon as the response
    goes out, so the background task is given a durable path, never the UploadFile itself.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Supported types: {sorted(SUPPORTED_SUFFIXES)}.",
        )

    settings = get_settings()
    # Written into the uploading tenant's subtree, which is what makes the document private
    # to them; the public tenant keeps writing flat, so an open demo's layout is unchanged.
    dest_dir = owner_corpus_dir(settings.corpus_dir, auth.get_owner())
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Short UUID suffix avoids collisions between uploads that share a filename (including
    # two concurrent uploads of the exact same file) without needing to inspect existing
    # corpus contents first.
    dest_name = f"{_safe_stem(file.filename)}_{uuid.uuid4().hex[:8]}{suffix}"
    dest_path = dest_dir / dest_name

    size_bytes = 0
    try:
        with dest_path.open("wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                size_bytes += len(chunk)
                if size_bytes > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit.",
                    )
                out.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        logger.exception("failed to persist upload %r", file.filename)
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file.") from exc
    finally:
        await file.close()

    if size_bytes == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("upload persisted", extra={"dest_name": dest_name, "size_bytes": size_bytes})

    task = create_task(filename=dest_name, original_filename=file.filename)
    background_tasks.add_task(
        _run_ingest_in_background, get_trace_id(), task.task_id, auth.get_owner()
    )

    return IngestResponse(
        task_id=task.task_id,
        filename=dest_name,
        original_filename=file.filename,
        size_bytes=size_bytes,
        status="queued",
        message="File saved; indexing has started in the background.",
    )


@app.post("/api/v1/ingest/url", response_model=IngestResponse, status_code=202)
@limiter.limit(_per_ip_limit)
@global_limiter.limit(_global_limit)
def ingest_url(
    request: Request,
    background_tasks: BackgroundTasks,
    body: IngestUrlRequest,
) -> IngestResponse:
    """Fetches a public web page, saves its extracted text into `corpus_dir` as markdown, and
    schedules the same background re-index as a file upload. Sync handler on purpose: FastAPI
    runs it in the threadpool, and the fetch (bounded by FETCH_TIMEOUT_SECONDS) happens before
    the 202 goes out so an unreachable/blocked/empty URL fails the request itself instead of
    a background task the client would have to poll to discover."""
    try:
        page = fetch_page(body.url)
    except UrlIngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("url ingestion failed for url=%r", body.url)
        raise HTTPException(status_code=500, detail="Failed to fetch the URL.") from exc

    settings = get_settings()
    dest_dir = owner_corpus_dir(settings.corpus_dir, auth.get_owner())
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem_source = page.title or Path(str(page.url)).name or "webpage"
    dest_name = f"{_safe_stem(stem_source)[:60]}_{uuid.uuid4().hex[:8]}.md"
    dest_path = dest_dir / dest_name
    markdown = page_to_markdown(page)
    dest_path.write_text(markdown, encoding="utf-8")

    logger.info("url ingested", extra={"dest_name": dest_name, "url": page.url})

    task = create_task(filename=dest_name, original_filename=body.url)
    background_tasks.add_task(
        _run_ingest_in_background, get_trace_id(), task.task_id, auth.get_owner()
    )

    return IngestResponse(
        task_id=task.task_id,
        filename=dest_name,
        original_filename=body.url,
        size_bytes=len(markdown.encode("utf-8")),
        status="queued",
        message="Page fetched; indexing has started in the background.",
    )


@app.get("/api/v1/ingest/{task_id}", response_model=IngestTaskStatus)
async def get_ingest_task_status(task_id: str) -> IngestTaskStatus:
    """Polled by the frontend every ~1-2s while a drawer entry is non-terminal. Deliberately
    not behind `@limiter.limit`/`@global_limiter.limit` -- those budgets are sized for
    LLM-backed endpoints, and a client polling this every second for a multi-minute PDF embed
    would blow through them. Reading an in-memory dict is cheap enough not to need its own
    limit.
    """
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Unknown ingest task.")

    return IngestTaskStatus(
        task_id=task.task_id,
        filename=task.filename,
        original_filename=task.original_filename,
        stage=task.stage,
        message=task.message,
        error=task.error,
        indexed_chunks=task.indexed_chunks,
    )


def _resolve_history(body: ResearchRequest) -> list[dict]:
    """Server-side history wins: with a conversation_id the transcript comes from the store
    (the client can't forge or truncate it); otherwise the client-supplied stateless
    `history` is used as-is. 404s on unknown ids *before* any LLM spend."""
    if body.conversation_id is None:
        return [turn.model_dump() for turn in body.history]
    if conversations.get_conversation(body.conversation_id, owner=auth.get_owner()) is None:
        raise HTTPException(status_code=404, detail="Unknown conversation.")
    return conversations.get_history(body.conversation_id)


def _persist_exchange(body: ResearchRequest, final_state: dict) -> str | None:
    """Appends the completed exchange to its conversation (creating one titled after the
    first question when the client didn't supply an id), or does nothing with save=false.
    Persistence failures are logged, not raised -- the user already has their answer, and
    losing one history entry beats turning a successful research call into a 500."""
    if body.conversation_id is None and not body.save:
        return None
    try:
        conversation_id = body.conversation_id
        if conversation_id is None:
            conversation_id = conversations.create_conversation(
                title=body.question, owner=auth.get_owner()
            ).id
        summary = build_research_summary(final_state)
        conversations.append_turn(
            conversation_id,
            question=body.question,
            answer=final_state.get("final_answer") or final_state.get("research_report", ""),
            report=final_state.get("research_report"),
            summary=summary.model_dump(),
        )
        return conversation_id
    except Exception:
        logger.exception("failed to persist conversation exchange")
        return body.conversation_id


# Two paths, one handler. `/api/v1/research` is canonical -- it matches every other endpoint
# in this app and leaves room to ship a v2 without breaking callers -- while the original
# unversioned `/research` stays registered so existing clients (and the README's curl
# examples from before versioning) keep working. The alias is marked deprecated and hidden
# from the schema so the docs show one obvious path. Decorators apply bottom-up, so both
# registrations sit above the rate limiters and each route gets both budgets applied.
@app.post("/api/v1/research", response_model=ResearchResponse)
@app.post("/research", response_model=ResearchResponse, deprecated=True, include_in_schema=False)
@limiter.limit(_per_ip_limit)
@global_limiter.limit(_global_limit)
def research(request: Request, body: ResearchRequest) -> ResearchResponse:
    history = _resolve_history(body)
    try:
        result = _graph.invoke(
            {
                "question": body.question,
                "chat_history": history,
                "trace_id": get_trace_id(),
                # Retrieval is scoped to this tenant -- see ingestion/ownership.py.
                "owner": auth.get_owner(),
                "filters": body.filters,
            },
            config={"recursion_limit": _RECURSION_LIMIT},
        )
    except Exception as exc:
        logger.exception("research failed for question=%r", body.question)
        metrics.record_graph_run(None, "error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    metrics.record_graph_run(result.get("route"), "ok")
    conversation_id = _persist_exchange(body, result)

    return ResearchResponse(
        question=body.question,
        report=result["research_report"],
        answer=result.get("final_answer"),
        route=result.get("route"),
        confidence_score=result.get("confidence_score"),
        summary=build_research_summary(result),
        conversation_id=conversation_id,
    )


async def _stream_research_events(
    body: ResearchRequest, history: list[dict], owner: str
) -> AsyncIterator[str]:
    question = body.question
    # Once this generator has started, the response is already HTTP 200 with headers flushed
    # -- there is no way to surface an HTTP error status mid-stream. Every failure, including
    # ones from deep inside a graph node (e.g. quota exhaustion), must degrade to a "type":
    # "error" SSE frame instead of propagating and truncating the connection.
    timeout_seconds = get_settings().graph_timeout_seconds
    connection = _StreamConnection()
    _active_streams.add(connection)
    metrics.sse_streams_active.inc()
    try:
        final_state: dict = {}
        # Two stream modes at once: "updates" drives the per-node progress frames, and
        # "messages" relays LLM token callbacks so the answer can render as it's generated.
        # With a mode list, LangGraph yields (mode, payload) tuples instead of bare updates.
        graph_iter = _graph.astream(
            {
                "question": question,
                "chat_history": history,
                "trace_id": get_trace_id(),
                "owner": owner,
                "filters": body.filters,
            },
            config={"recursion_limit": _RECURSION_LIMIT},
            stream_mode=["updates", "messages"],
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

            # Tuple = (mode, payload) from the stream_mode list above. Bare dicts are kept
            # supported so tests can stub astream with plain "updates" output.
            if isinstance(update, tuple):
                mode, payload = update
                if mode == "messages":
                    chunk, metadata = payload
                    # Only the synthesis node's tokens are the user-facing answer -- the
                    # router/decomposer/grader also make LLM calls, and their output is
                    # internal machinery, not something to render in the chat bubble.
                    if metadata.get("langgraph_node") != "synthesize_answer":
                        continue
                    text = chunk.text if isinstance(chunk.text, str) else ""
                    if text:
                        yield f"data: {StreamEvent(type='token', token=text).model_dump_json()}\n\n"
                    continue
                update = payload

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

        metrics.record_graph_run(final_state.get("route"), "ok")
        conversation_id = _persist_exchange(body, final_state)

        done_event = StreamEvent(
            type="done",
            report=final_state.get("research_report", ""),
            answer=final_state.get("final_answer"),
            route=final_state.get("route"),
            confidence_score=final_state.get("confidence_score"),
            summary=build_research_summary(final_state),
            conversation_id=conversation_id,
        )
        yield f"data: {done_event.model_dump_json()}\n\n"
    except Exception as exc:
        logger.exception("research_stream failed for question=%r", question)
        # A timeout is its own outcome, not a generic failure: the two have completely
        # different responses (raise GRAPH_TIMEOUT_SECONDS vs. go find the broken provider),
        # and collapsing them into one counter hides which is happening.
        outcome = "timeout" if isinstance(exc, TimeoutError) else "error"
        metrics.record_graph_run(None, outcome)
        detail = str(exc) or f"{type(exc).__name__} (no further detail from the underlying service)"
        error_event = StreamEvent(type="error", detail=detail)
        yield f"data: {error_event.model_dump_json()}\n\n"
    finally:
        _active_streams.discard(connection)
        metrics.sse_streams_active.dec()


@app.post("/api/v1/research/stream")
@app.post("/research/stream", deprecated=True, include_in_schema=False)
@limiter.limit(_per_ip_limit)
@global_limiter.limit(_global_limit)
async def research_stream(request: Request, body: ResearchRequest) -> StreamingResponse:
    # History (and the conversation_id 404) resolves before streaming starts -- once the
    # generator yields, the status is locked at 200 and errors can only be SSE frames.
    history = _resolve_history(body)
    # The owner is resolved here rather than inside the generator: `auth.owner_var` is a
    # contextvar reset when the request handler returns, and the generator runs *after* that,
    # while the response streams. Reading it lazily would see the default tenant.
    return StreamingResponse(
        _stream_research_events(body, history, auth.get_owner()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Conversation CRUD. Like the ingest task-status endpoint, these are cheap local DB reads
# polled/loaded freely by the UI, so they sit outside the LLM-sized rate-limit budgets.


@app.get("/api/v1/auth/check")
def auth_check() -> dict:
    """Reached only with a valid key (or with auth disabled) -- the middleware rejects the
    rest. The frontend calls this at startup to decide whether to show the access gate."""
    return {"ok": True, "auth_required": auth.auth_enabled(), "owner": auth.get_owner()}


@app.post("/api/v1/feedback", response_model=FeedbackResponse, status_code=201)
def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    """Records one user rating of an answer.

    Outside the LLM rate-limit budgets like the other cheap local writes -- a user clicking
    thumbs-down twice should never be told to slow down, and rate-limiting the one channel
    that reports the system is answering badly is the wrong thing to throttle.
    """
    feedback_id = conversations.record_feedback(
        question=body.question,
        rating=body.rating,
        owner=auth.get_owner(),
        conversation_id=body.conversation_id,
        note=body.note,
        route=body.route,
        confidence_score=body.confidence_score,
    )
    metrics.record_feedback(body.rating)
    logger.info(
        "feedback recorded",
        extra={"route": body.route or "", "node": f"rating={body.rating}"},
    )
    return FeedbackResponse(id=feedback_id)


@app.get("/api/v1/feedback/summary", response_model=FeedbackSummary)
def get_feedback_summary() -> FeedbackSummary:
    """Aggregate ratings for this tenant, plus recently downvoted questions -- the material
    for keeping the golden eval dataset resembling what people actually ask."""
    return FeedbackSummary(**conversations.feedback_summary(owner=auth.get_owner()))


@app.get("/api/v1/conversations", response_model=list[ConversationSummary])
def list_conversations() -> list[ConversationSummary]:
    return [
        ConversationSummary(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=c.message_count,
        )
        for c in conversations.list_conversations(owner=auth.get_owner())
    ]


@app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str) -> ConversationDetail:
    conversation = conversations.get_conversation(conversation_id, owner=auth.get_owner())
    if conversation is None:
        raise HTTPException(status_code=404, detail="Unknown conversation.")
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            ConversationMessage(
                role=m.role,
                content=m.content,
                report=m.report,
                summary=m.summary,
                created_at=m.created_at,
            )
            for m in conversations.get_messages(conversation_id)
        ],
    )


@app.delete("/api/v1/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> None:
    if not conversations.delete_conversation(conversation_id, owner=auth.get_owner()):
        raise HTTPException(status_code=404, detail="Unknown conversation.")


# Single-container deployments (Docker image, Render) bake the built frontend into the image
# and point STATIC_DIR at it, so the API serves the whole app from one origin. Mounted last:
# Starlette matches routes in registration order, so every API route above still wins, and
# `html=True` makes "/" serve index.html. In development this is a no-op (STATIC_DIR unset).
_static_dir = get_settings().static_dir
if _static_dir is not None and _static_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")
