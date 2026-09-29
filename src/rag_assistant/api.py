import anyio
import asyncio
import hashlib
import json
import logging
import re
import signal
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from rag_assistant import auth, budget, metrics, oidc, tenancy
from rag_assistant.config import get_settings
from rag_assistant.ingestion.generations import active_index_dir
from rag_assistant.conversations import store as conversations
from rag_assistant.graph.build_graph import build_graph
from rag_assistant.graph.research_summary import build_research_summary
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.acl import (
    DocumentAcl,
    entry_readable,
    parse_acl_fields,
    sidecar_path,
    write_acl,
)
from rag_assistant.ingestion.loaders import SUPPORTED_SUFFIXES
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.ingestion.ownership import display_source, owner_corpus_dir, visible_owners
from rag_assistant.ingestion import tasks as ingest_tasks
from rag_assistant.ingestion.tasks import create_task, get_task, update_task
from rag_assistant.ingestion.url_fetch import UrlIngestError, fetch_page, page_to_markdown
from rag_assistant.llm import embedding_backend_unavailable
from rag_assistant.logging_conf import configure_logging
from rag_assistant.readiness import (
    check_chroma,
    check_embeddings,
    check_index_generation,
    check_local_llm,
    check_rate_limit_storage,
    check_row_security,
    check_web_search,
)
from rag_assistant.schemas.api import (
    ConversationDetail,
    ConversationMessage,
    ConversationSummary,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSummary,
    IndexedSource,
    IngestResponse,
    IngestTaskStatus,
    IngestUrlRequest,
    ResearchRequest,
    ResearchResponse,
    ActivateGenerationRequest,
    ReindexRequest,
    SourceAclRequest,
    SourceAclResponse,
    SourceDeleteResponse,
    StreamEvent,
    TenantPurgeResponse,
    TenantUsageResponse,
)
from rag_assistant.tracing import configure_otel, get_trace_id, new_trace_id, trace_id_var

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
    # An ingest runs inside the process that accepted the upload, so a deploy or a crash
    # leaves its task record stuck in a non-terminal stage with nothing working on it. There
    # is nothing to resume, so the startup pass fails those loudly rather than leaving a
    # client polling a job nobody is doing. Best-effort: a broken task backend must not stop
    # the service from starting.
    try:
        for stale in ingest_tasks.reconcile_stale_tasks():
            # Off the event loop: the resumed work is the same synchronous graph-and-embed
            # job a request would have run, and doing it inline would block startup until
            # every orphaned corpus finished re-indexing.
            threading.Thread(
                target=_run_ingest_in_background,
                args=(new_trace_id(), stale.task_id, stale.owner),
                name=f"resume-ingest-{stale.task_id[:8]}",
                daemon=True,
            ).start()
    except Exception:
        logger.warning("Could not reconcile in-flight ingest tasks at startup", exc_info=True)
    # Sized explicitly rather than left at AnyIO's default, because this is the concurrency
    # ceiling for `/api/v1/research` -- a sync handler holding one thread for a multi-second
    # graph run. Set here, inside the running loop, since the limiter is loop-scoped.
    # Before anything else that might emit a span.
    try:
        configure_otel()
    except Exception:
        logger.warning("Could not configure OpenTelemetry tracing", exc_info=True)
    # Probed at startup rather than discovered during an incident: a multi-replica
    # deployment whose limiter store is unreachable still serves, but each replica enforces
    # its own private copy of a cap documented as global, and nothing else says so.
    try:
        storage_uri = rate_limit_storage_uri()
        if storage_uri.startswith("memory://"):
            logger.info(
                "rate limiter using in-process counters; with more than one replica the "
                "global cap is enforced per replica (set RATE_LIMIT_STORAGE_URI to share it)"
            )
        else:
            ok, detail = check_rate_limit_storage()
            if ok:
                logger.info("rate limiter sharing counters via %s", storage_uri)
            else:
                logger.warning("rate limiter storage unreachable at startup: %s", detail)
    except Exception:
        logger.warning("Could not probe the rate-limit storage", exc_info=True)
    if get_settings().connector_scheduler:
        _start_connector_scheduler()
    try:
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = get_settings().api_threadpool_size
        logger.info("request threadpool sized to %d threads", limiter.total_tokens)
    except Exception:
        logger.warning("Could not size the request threadpool", exc_info=True)
    try:
        yield
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


_SCHEDULER_TICK_SECONDS = 60.0


def _start_connector_scheduler() -> None:
    """Runs due connector syncs on a daemon thread. Each tick is independent and every error
    is caught: a scheduler that died on the first bad sync would stop every other connector
    too, silently. The per-connector lock in sync.py keeps two replicas that both run the
    scheduler from syncing the same source at once."""
    from rag_assistant.connectors.sync import run_due_syncs

    def _loop() -> None:
        while not _shutdown_event.is_set():
            try:
                run_due_syncs()
            except Exception:
                logger.exception("connector scheduler tick failed")
            time.sleep(_SCHEDULER_TICK_SECONDS)

    threading.Thread(target=_loop, name="connector-scheduler", daemon=True).start()
    logger.info("connector scheduler started")


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
    # An SSO caller is bucketed by who they are, not by the token they hold: tokens are
    # refreshed every few minutes, and a bucket per token would reset the limit with each one.
    principal = auth.get_principal()
    if principal is not None and principal.method == "oidc":
        return "user:" + principal.fingerprint
    key = auth.extract_key({k.lower(): v for k, v in request.scope.get("headers", [])})
    if key:
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return get_remote_address(request)


def rate_limit_storage_uri() -> str:
    """The `limits` storage URI for the limiter buckets.

    Blank (and the explicit `memory://`) mean per-process counters, which is the right answer
    for the single-container default: there is one process, so its buckets already are the
    whole deployment's.
    """
    configured = get_settings().rate_limit_storage_uri.strip()
    return configured or "memory://"


def _build_limiter(key_func, prefix: str) -> Limiter:
    """A limiter over the configured storage.

    `in_memory_fallback_enabled` is the important argument. Shared storage turns Redis into a
    dependency of every request, and a limiter that 500s the API because its bookkeeping
    store is unreachable has inverted its own purpose -- it exists to keep the service up.
    With the fallback on, a Redis outage downgrades enforcement to per-process (slowapi flips
    to a memory limiter and re-checks the backend periodically), which is strictly the old
    behaviour and strictly better than either failing open or failing the request.

    `key_prefix` keeps the two limiters' buckets apart in a shared Redis, where they would
    otherwise be distinguishable only by their key strings, and makes them legible to anyone
    reading the keyspace during an incident.
    """
    return Limiter(
        key_func=key_func,
        storage_uri=rate_limit_storage_uri(),
        in_memory_fallback_enabled=True,
        key_prefix=prefix,
    )


limiter = _build_limiter(_caller_identity, "rag:caller")
global_limiter = _build_limiter(lambda request: "global", "rag:global")
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
    the SSO bootstrap config and the static frontend stay open. With neither API keys nor SSO
    configured this resolves every request to the "public" tenant and never rejects -- open
    demo mode. OPTIONS passes through so CORS preflights (which never carry credentials) reach
    the CORS layer.

    A credential is an API key or an SSO access token, presented the same way; see
    `auth.resolve_principal` for the order they are tried in."""

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
            # Open demo mode: every request is the public tenant and nothing is rejected --
            # except operating the deployment. Rebuilding the index re-embeds the whole corpus
            # at the operator's expense, and an open demo must not offer that to anyone who
            # finds the URL.
            if auth.required_scope(method, path) == auth.ADMIN:
                await reject(403, "Admin endpoints require authentication to be configured.")
                return
            await self.app(scope, receive, send)
            return

        presented = auth.extract_key(headers)
        try:
            # Token verification may fetch the identity provider's signing keys on a cache
            # miss, so it runs off the event loop rather than stalling every other request.
            # API keys are a few in-memory comparisons and stay inline.
            if presented and get_settings().oidc_issuer and oidc.looks_like_jwt(presented):
                principal = await anyio.to_thread.run_sync(auth.resolve_principal, presented)
            else:
                principal = auth.resolve_principal(presented)
            rejection = "invalid_key"
        except auth.CredentialRejected as exc:
            principal = None
            rejection = f"invalid_token:{exc}"
        if principal is None:
            auth.audit("auth rejected", path=path, method=method, outcome=rejection)
            await reject(401, "Missing, invalid, or expired credentials.")
            return

        owner_token = auth.owner_var.set(principal.owner)
        principal_token = auth.principal_var.set(principal)
        key_token = auth.api_key_var.set(
            auth.resolve_key(presented) if principal.method == "api_key" else None
        )
        try:
            # 403 rather than 401: the credential is valid, it simply isn't allowed to do
            # this. Returning 401 would tell a read-only client to go re-authenticate, which
            # it cannot fix by presenting the same key again.
            needed = auth.required_scope(method, path)
            if not principal.has_scope(needed):
                auth.audit("auth forbidden", path=path, method=method, outcome=f"missing:{needed}")
                await reject(403, f"This credential lacks the {needed!r} scope.")
                return
            auth.audit("auth accepted", path=path, method=method, outcome="ok")
            await self.app(scope, receive, send)
        finally:
            auth.api_key_var.reset(key_token)
            auth.principal_var.reset(principal_token)
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
    "refine_retrieval": "Confidence low, rephrasing and searching the corpus again...",
    "corrective_web_search": "Confidence low, running corrective web search...",
    "synthesize_answer": "Synthesizing answer...",
    "verify_groundedness": "Checking the answer against its sources...",
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
    # Same category as local_llm: reported, not part of the verdict. See check_rate_limit_storage.
    rate_limit_ok, rate_limit_err = check_rate_limit_storage()
    # Both reported, neither part of the verdict: which generation serves, and whether the
    # database is enforcing tenant isolation as well as the queries.
    generation_ok, generation = check_index_generation()
    row_security_ok, row_security_err = check_row_security()
    ready_ok = chroma_ok and web_search_ok and embeddings_ok and generation_ok
    body = {
        "status": "ok" if ready_ok else "unavailable",
        "chroma": {"ok": chroma_ok, "error": chroma_err},
        "embeddings": {"ok": embeddings_ok, "error": embeddings_err},
        "web_search": {"ok": web_search_ok, "error": web_search_err},
        "local_llm": {"ok": local_llm_ok, "error": local_llm_err},
        "rate_limit_storage": {"ok": rate_limit_ok, "error": rate_limit_err},
        "index_generation": {
            "ok": generation_ok,
            "error": None if generation_ok else generation,
            "generation": generation if generation_ok else None,
        },
        "row_security": {"ok": row_security_ok, "error": row_security_err},
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

    Retries in a loop up to INGEST_MAX_ATTEMPTS. In-place rather than re-queued because the
    work is already here and the staged file is already on disk, and `build_index` decides
    what to do from a fingerprint -- so a retry costs only what the failed attempt did not
    finish. Bounded because a file that crashes the parser will crash it again, and an
    unbounded retry turns one bad upload into a loop that reads as an unstable service.

    The attempt counter lives on the task record rather than in this frame, so a restart
    midway resumes at the right attempt instead of granting a fresh budget every crash.
    """
    settings = get_settings()
    token = trace_id_var.set(trace_id)
    try:
        existing = get_task(task_id)
        attempt = (existing.attempts if existing else 0) + 1
        while True:
            update_task(task_id, stage="parsing", message="Starting ingestion...", attempts=attempt)
            try:
                # Scoped to the uploading tenant: their upload should cost their corpus, not
                # a scan of every other tenant's documents (see ingestion/loaders).
                result = build_index(
                    owner=owner,
                    on_stage=lambda stage, message: update_task(
                        task_id, stage=stage, message=message
                    ),
                )
            except Exception as exc:
                logger.exception("background ingestion failed (attempt %d)", attempt)
                if attempt >= settings.ingest_max_attempts:
                    update_task(
                        task_id,
                        stage="failed",
                        message=f"Indexing failed after {attempt} attempt(s).",
                        error=str(exc),
                        attempts=attempt,
                    )
                    return
                update_task(
                    task_id,
                    stage="queued",
                    message=f"Attempt {attempt} failed; retrying.",
                    error=str(exc),
                    attempts=attempt,
                )
                # What makes this a retry rather than a tight loop against a provider that is
                # rate-limiting us.
                time.sleep(settings.ingest_retry_delay_seconds)
                attempt += 1
                continue

            # Charged here rather than at the endpoint, because the endpoint returns 202
            # before any of this has happened and the cost is not knowable until it has.
            # Embeddings and vision calls do not report usage the way chat completions do,
            # so the amounts are estimates -- see budget.charge_ingest.
            budget.charge_ingest(
                owner, result.embedded_chars, result.vision_calls, result.description_calls
            )
            logger.info(
                "background ingestion complete",
                extra={
                    "indexed_chunks": result.indexed_chunks,
                    "changed_files": result.changed_files,
                    "skipped_files": result.skipped_files,
                    "removed_files": result.removed_files,
                    "embedded_chars": result.embedded_chars,
                    "vision_calls": result.vision_calls,
                    "attempts": attempt,
                },
            )
            if result.changed_files == 0 and result.removed_files == 0:
                message = "No changes detected -- file content matched what was already indexed."
            else:
                message = (
                    f"Indexed {result.indexed_chunks} chunk(s) from {result.changed_files} "
                    f"file(s); local search is up to date."
                )
            update_task(
                task_id,
                stage="indexed",
                message=message,
                indexed_chunks=result.indexed_chunks,
                attempts=attempt,
            )
            return
    finally:
        trace_id_var.reset(token)


@app.post("/api/v1/ingest", response_model=IngestResponse, status_code=202)
@limiter.limit(_per_ip_limit)
@global_limiter.limit(_global_limit)
async def ingest_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    allowed_users: str | None = Form(None),
    allowed_groups: str | None = Form(None),
) -> IngestResponse:
    """Accepts one corpus file (.md/.txt/.pdf), persists it into `corpus_dir`, and schedules
    a background re-index. The file must be written to disk *before* this handler returns --
    FastAPI closes and deletes `UploadFile`'s underlying temp file as soon as the response
    goes out, so the background task is given a durable path, never the UploadFile itself.

    `allowed_users` / `allowed_groups` (comma-separated) restrict the document to those
    people inside the tenant; omitted, it is visible to the whole tenant (see ingestion/acl.py).
    """
    acl = parse_acl_fields(allowed_users, allowed_groups)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Supported types: {sorted(SUPPORTED_SUFFIXES)}.",
        )

    # Before the upload is accepted, not after: ingest is the expensive path (an embedding
    # per chunk, a vision call per figure), and streaming 25MB to disk first only to reject
    # it wastes the one resource the budget is protecting.
    try:
        budget.enforce(auth.get_owner())
    except budget.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

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
    # Hashed while streaming rather than by re-reading the file afterwards: the bytes are
    # already passing through, and a second full read of a 25MB upload to learn something
    # this loop could have computed for free is pure latency.
    digest = hashlib.sha256()
    try:
        with dest_path.open("wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                size_bytes += len(chunk)
                if size_bytes > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit.",
                    )
                digest.update(chunk)
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

    # Written before the ingest is scheduled, never after: an ingest that ran first would
    # index the document as visible to the whole tenant, and it would stay that way until the
    # next run noticed the sidecar.
    write_acl(dest_path, acl)

    logger.info("upload persisted", extra={"dest_name": dest_name, "size_bytes": size_bytes})

    # Owner-scoped: two tenants uploading identical bytes are two separate ingests into two
    # separate subtrees, and collapsing them would put one tenant's document in the other's
    # corpus. Same bytes from the same tenant is the case worth collapsing -- that is a retry
    # after a timeout, or a double-clicked upload button.
    content_hash = f"{auth.get_owner()}:{acl.fingerprint()}:{digest.hexdigest()}"
    existing = ingest_tasks.find_task_by_content(content_hash)
    if existing is not None and existing.stage != "failed":
        # The duplicate upload is discarded rather than indexed. `build_index` would skip it
        # by fingerprint anyway, but only after the file was written into the corpus, where
        # it would sit forever as a second copy under a different UUID suffix.
        dest_path.unlink(missing_ok=True)
        sidecar_path(dest_path).unlink(missing_ok=True)
        logger.info(
            "duplicate upload collapsed onto the existing task",
            extra={"task_id": existing.task_id, "original_filename": file.filename},
        )
        return IngestResponse(
            task_id=existing.task_id,
            filename=existing.filename,
            original_filename=existing.original_filename,
            size_bytes=size_bytes,
            status=existing.stage,
            message="This file was already uploaded; returning the original ingest task.",
        )

    task = create_task(
        filename=dest_name,
        original_filename=file.filename,
        content_hash=content_hash,
        owner=auth.get_owner(),
    )
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
    # Before the fetch, for the same reason as the upload path: the network round trip and
    # the indexing behind it are what the budget is protecting.
    try:
        budget.enforce(auth.get_owner())
    except budget.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
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

    # The owner is recorded on the task so a restart resumes it in the right tenant's scope;
    # without it, a URL ingest orphaned by a deploy was resumed as the public tenant.
    task = create_task(filename=dest_name, original_filename=body.url, owner=auth.get_owner())
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
    # Another tenant's task is reported as unknown rather than returned: its record names the
    # uploaded file, and task ids are only unguessable, not secret.
    if task is None or task.owner != auth.get_owner():
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
    if (
        conversations.get_conversation(body.conversation_id, owner=auth.get_conversation_owner())
        is None
    ):
        raise HTTPException(status_code=404, detail="Unknown conversation.")
    return conversations.get_history(body.conversation_id)


def _state_principals() -> list[str] | None:
    """The caller's ACL principals in the form graph state carries them: a sorted list, or
    None for a caller that bypasses document ACLs. A list rather than a frozenset because
    graph state is plain data that LangGraph copies into every `Send` payload."""
    principals = auth.get_principals()
    return None if principals is None else sorted(principals)


def _persist_exchange(
    body: ResearchRequest, final_state: dict, conversation_owner: str | None = None
) -> str | None:
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
                title=body.question, owner=conversation_owner or auth.get_conversation_owner()
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
async def research(request: Request, body: ResearchRequest) -> ResearchResponse:
    # Async, and that is the point rather than a style choice. This used to be a sync `def`,
    # so FastAPI ran it on the AnyIO worker pool and each in-flight question held one thread
    # for the whole graph -- seconds, not milliseconds -- making API_THREADPOOL_SIZE the hard
    # concurrency ceiling of the service no matter how idle the CPU was. The graph's own
    # latency is almost entirely provider latency, which is I/O: the LLM-bound nodes now await
    # it on the event loop, and only the nodes doing blocking library work (retrieval, web
    # search) take a thread, for milliseconds each.
    history = _resolve_history(body)
    owner = auth.get_owner()
    principals = _state_principals()
    # 429 before any model is called, rather than after the bill. Disabled by default; see
    # budget.py for why the charge lands after the run rather than being pre-authorised.
    try:
        budget.enforce(owner)
    except budget.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    accountant = budget.TokenAccountant()
    # Incremented around the graph run only, not the whole handler: the gauge answers
    # "how many questions are in flight right now". It no longer tracks worker-thread
    # saturation, because a graph run no longer occupies a worker thread end to end -- see
    # the note on this handler being async.
    metrics.research_in_flight.inc()
    try:
        result = await _graph.ainvoke(
            {
                "question": body.question,
                "chat_history": history,
                "trace_id": get_trace_id(),
                # Retrieval is scoped to this tenant -- see ingestion/ownership.py -- and,
                # within it, to the documents this caller may read (ingestion/acl.py).
                "owner": owner,
                "principals": principals,
                "filters": body.filters,
            },
            # The accountant rides the config so it reaches every node and nested LLM call;
            # a contextvar would not reliably survive LangGraph's thread scheduling.
            config={"recursion_limit": _RECURSION_LIMIT, "callbacks": [accountant]},
        )
    except Exception as exc:
        logger.exception("research failed for question=%r", body.question)
        metrics.record_graph_run(None, "error")
        # Charged even on failure: a run that errored after four LLM calls cost exactly as
        # much as one that succeeded, and not charging it would make failure the cheap way to
        # exhaust a provider quota.
        budget.charge(owner, accountant.total_tokens)
        # 503, not 500, when the self-hosted embedding server is simply gone: it is a
        # dependency outage a client should retry and an orchestrator should route around,
        # not a defect in the request. Everything else stays a 500.
        unavailable = embedding_backend_unavailable(exc)
        raise HTTPException(
            status_code=503 if unavailable else 500, detail=unavailable or str(exc)
        ) from exc
    finally:
        metrics.research_in_flight.dec()

    budget.charge(owner, accountant.total_tokens)
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
    body: ResearchRequest,
    history: list[dict],
    owner: str,
    principals: list[str] | None = None,
    conversation_owner: str | None = None,
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
    accountant = budget.TokenAccountant()
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
                "principals": principals,
                "filters": body.filters,
            },
            config={"recursion_limit": _RECURSION_LIMIT, "callbacks": [accountant]},
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
        conversation_id = _persist_exchange(body, final_state, conversation_owner)

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
        # In `finally` so every exit charges: success, error, timeout, and the shutdown path
        # that returns early mid-stream. A client disconnecting halfway through still spent
        # whatever the graph had already spent by then.
        budget.charge(owner, accountant.total_tokens)
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
    owner = auth.get_owner()
    # Same for the caller's identity: both are contextvars the generator cannot read.
    principals = _state_principals()
    conversation_owner = auth.get_conversation_owner()
    # Checked here, not in the generator, for the same reason the 404 above is: once the
    # generator yields its first frame the status is locked at 200, and an exhausted budget
    # would have to be reported as an SSE error frame that no HTTP client can act on.
    try:
        budget.enforce(owner)
    except budget.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return StreamingResponse(
        _stream_research_events(body, history, owner, principals, conversation_owner),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Conversation CRUD. Like the ingest task-status endpoint, these are cheap local DB reads
# polled/loaded freely by the UI, so they sit outside the LLM-sized rate-limit budgets.


@app.get("/api/v1/auth/check")
def auth_check() -> dict:
    """Reached only with a valid credential (or with auth disabled) -- the middleware rejects
    the rest. The frontend calls this at startup to decide whether to show the access gate,
    and to show who is signed in."""
    principal = auth.get_principal()
    return {
        "ok": True,
        "auth_required": auth.auth_enabled(),
        "owner": auth.get_owner(),
        "method": principal.method if principal else "open",
        "user": (principal.email or principal.subject) if principal else None,
        "groups": sorted(principal.groups) if principal else [],
        "scopes": sorted(principal.scopes) if principal else [],
    }


@app.get("/auth/config")
def auth_config() -> dict:
    """How a browser should authenticate, before it has any credential to present.

    Deliberately outside the protected prefixes: the sign-in page needs to know the identity
    provider and public client id in order to *obtain* a credential. Nothing here is secret --
    a browser-based client is public by definition, which is why it uses PKCE.
    """
    return {
        "auth_required": auth.auth_enabled(),
        "api_keys": bool(auth.load_api_keys()),
        "oidc": oidc.public_client_config(),
    }


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
        owner=auth.get_conversation_owner(),
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
    return FeedbackSummary(**conversations.feedback_summary(owner=auth.get_conversation_owner()))


@app.get("/api/v1/sources", response_model=list[IndexedSource])
def list_sources() -> list[IndexedSource]:
    """The indexed files this tenant can retrieve from.

    Read from the ingestion manifest rather than by querying Chroma: the manifest already
    records one row per source with its chunk ids and owner, so this is a file read rather
    than a scan of every chunk in the collection.

    Exists so a client can offer a source filter without the user having to know exact
    filenames -- `filters.sources` matches on the `source` field returned here, while
    `display_name` is what a person should be shown (a tenant's own path prefix is noise to
    the tenant it belongs to).
    """
    owner = auth.get_owner()
    allowed = set(visible_owners(owner))
    principals = auth.get_principals()
    manifest = load_manifest(active_index_dir())
    sources = [
        IndexedSource(
            source=source,
            display_name=display_source(source),
            chunk_count=len(entry.get("chunk_ids", [])),
            owner=entry.get("owner", auth.PUBLIC_OWNER),
            restricted=bool(entry.get("acl")),
        )
        for source, entry in sorted(manifest.items())
        # A document the caller may not read is not listed either: its filename is itself
        # information ("layoffs-2027-q1.pdf"), and listing it would leak exactly what the
        # retrieval filter withholds.
        if entry.get("owner", auth.PUBLIC_OWNER) in allowed and entry_readable(entry, principals)
    ]
    return sources


@app.delete("/api/v1/sources/{source_id:path}", response_model=SourceDeleteResponse)
@limiter.limit(_per_ip_limit)
def delete_source(request: Request, source_id: str) -> SourceDeleteResponse:
    """Removes one indexed document -- chunks, parent sections, manifest entry and the
    uploaded file.

    The unit that was missing. Erasure existed only per tenant, so a takedown request, a
    document past its retention date or one badly-parsed upload had no answer short of
    deleting everything the tenant owned and re-ingesting the rest.

    `{source_id:path}` because a source key carries slashes for tenant-owned files
    (`_t/alice/report.md`), and the un-suffixed converter would stop at the first one and
    404 every tenant document. Scoped to the caller's own sources, and a source belonging to
    someone else is reported as absent rather than as forbidden -- "forbidden" would confirm
    the file exists to anyone willing to guess at names.
    """
    owner = auth.get_owner()
    try:
        result = tenancy.purge_source(source_id, owner, principals=auth.get_principals())
    except tenancy.SourceNotFound as exc:
        raise HTTPException(status_code=404, detail=f"No indexed source {source_id!r}.") from exc
    except Exception as exc:
        logger.exception("source delete failed for source=%r owner=%r", source_id, owner)
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc

    auth.audit(
        "source deleted",
        path="/api/v1/sources",
        method="DELETE",
        outcome=f"source={source_id},chunks={result.chunks}",
    )
    return SourceDeleteResponse(
        source=result.source,
        display_name=display_source(result.source),
        chunks_removed=result.chunks,
        file_removed=result.file_removed,
    )


@app.put("/api/v1/sources/{source_id:path}/acl", response_model=SourceAclResponse)
@limiter.limit(_per_ip_limit)
def set_source_acl(
    request: Request,
    source_id: str,
    body: SourceAclRequest,
    background_tasks: BackgroundTasks,
) -> SourceAclResponse:
    """Replaces who may read one document. Empty lists make it visible to the whole tenant.

    Applied without re-embedding: the sidecar is rewritten and an ingest scoped to the tenant
    notices that only the permissions changed and rewrites the chunks' ACL metadata in place
    (see build_index). Until that finishes -- normally well under a second -- retrieval still
    applies the old permissions, which is why the response reports the task to poll.

    The caller must be able to read the document now, or it is reported as absent: a user
    who could lock others out of a document they cannot see, or grant themselves access to
    one, would make permissions meaningless.
    """
    owner = auth.get_owner()
    principals = auth.get_principals()
    manifest = load_manifest(active_index_dir())
    entry = manifest.get(source_id)
    if (
        entry is None
        or entry.get("owner", auth.PUBLIC_OWNER) != owner
        or not entry_readable(entry, principals)
    ):
        raise HTTPException(status_code=404, detail=f"No indexed source {source_id!r}.")

    document = tenancy.corpus_path(source_id)
    if document is None or not document.is_file():
        raise HTTPException(status_code=404, detail=f"No indexed source {source_id!r}.")
    acl = DocumentAcl(users=frozenset(body.users), groups=frozenset(body.groups))
    if principals is not None and acl.restricted and not (set(acl.principals()) & principals):
        # Refused rather than applied: the caller would immediately lose access to the
        # document they just changed, with no way back short of an administrator.
        raise HTTPException(
            status_code=400,
            detail="That ACL would exclude you; include one of your own groups or your user id.",
        )
    write_acl(document, acl)
    task = create_task(filename=source_id, original_filename=source_id, owner=owner)
    background_tasks.add_task(_run_ingest_in_background, get_trace_id(), task.task_id, owner)
    auth.audit(
        "source acl changed",
        path="/api/v1/sources",
        method="PUT",
        outcome=f"source={source_id},acl={acl.fingerprint()}",
    )
    return SourceAclResponse(
        source=source_id,
        users=sorted(acl.users),
        groups=sorted(acl.groups),
        restricted=acl.restricted,
        task_id=task.task_id,
    )


# ---- source connectors (see connectors/) ----


def _tenant_connectors() -> list:
    """The configured connectors that sync into the caller's tenant. Another tenant's
    connectors are not listed: their names and schedules describe that tenant's sources."""
    from rag_assistant.connectors.base import load_connector_configs

    owner = auth.get_owner()
    return [c for c in load_connector_configs() if c.owner == owner]


@app.get("/api/v1/connectors")
def list_connectors() -> list[dict]:
    from rag_assistant.connectors.sync import connector_status

    return [connector_status(config) for config in _tenant_connectors()]


@app.post("/api/v1/connectors/{name}/sync", status_code=202)
@limiter.limit(_per_ip_limit)
def trigger_connector_sync(request: Request, name: str, force: bool = False) -> dict:
    """Starts a sync of one of the caller's connectors now, rather than at its next interval.

    `force` overrides the deletion guard and is an operator decision -- it can delete most of
    a tenant's synced corpus -- so it needs the admin scope, not merely write."""
    from rag_assistant.connectors.sync import SyncInProgress, sync_connector

    config = next((c for c in _tenant_connectors() if c.name == name), None)
    if config is None:
        raise HTTPException(status_code=404, detail=f"No connector {name!r}.")
    principal = auth.get_principal()
    if force and principal is not None and not principal.has_scope(auth.ADMIN):
        raise HTTPException(status_code=403, detail="force needs the 'admin' scope.")

    trace_id = get_trace_id()

    def _run() -> None:
        token = trace_id_var.set(trace_id)
        try:
            sync_connector(config, force=force)
        except SyncInProgress:
            logger.info("connector %s already syncing; on-demand run skipped", name)
        except Exception:
            logger.exception("on-demand sync of connector %s failed", name)
        finally:
            trace_id_var.reset(token)

    threading.Thread(target=_run, name=f"connector-sync-{name}", daemon=True).start()
    auth.audit(
        "connector sync requested",
        path="/api/v1/connectors",
        method="POST",
        outcome=f"connector={name},force={force}",
    )
    return {"connector": name, "status": "started"}


# ---- index administration (see ingestion/generations.py, ingestion/reindex.py) ----
#
# Deployment-wide rather than per tenant, so they need the `admin` scope, which only an API
# key can carry (SSO admin groups are *tenant* admins). They exist because with embedded
# Chroma the serving process holds the index open, and the rebuild has to run inside it.

_reindex_job: dict = {}
_reindex_lock = threading.Lock()


@app.get("/api/v1/admin/index")
def index_status() -> dict:
    from dataclasses import asdict

    from rag_assistant.ingestion.reindex import describe_generations

    return {
        "generations": [asdict(info) for info in describe_generations()],
        "job": dict(_reindex_job),
    }


@app.post("/api/v1/admin/index/reindex", status_code=202)
def start_reindex(body: ReindexRequest) -> dict:
    from rag_assistant.ingestion.reindex import activate, build_generation

    with _reindex_lock:
        if _reindex_job.get("status") == "running":
            raise HTTPException(status_code=409, detail="A rebuild is already running.")
        _reindex_job.clear()
        _reindex_job.update(status="running", started_at=time.time(), progress=[])

    trace_id = get_trace_id()

    def _run() -> None:
        token = trace_id_var.set(trace_id)
        try:
            result = build_generation(
                embedding_model=body.embedding_model,
                from_corpus=body.from_corpus,
                on_progress=lambda m: _reindex_job["progress"].append(m),
            )
            _reindex_job.update(generation=result.generation, sources=result.sources)
            if body.activate:
                activate(result.generation)
                _reindex_job["activated"] = True
            _reindex_job["status"] = "built"
        except Exception as exc:
            logger.exception("index rebuild failed")
            _reindex_job.update(status="failed", error=str(exc))
        finally:
            _reindex_job["finished_at"] = time.time()
            trace_id_var.reset(token)

    threading.Thread(target=_run, name="index-rebuild", daemon=True).start()
    auth.audit("index rebuild started", path="/api/v1/admin/index", method="POST", outcome="ok")
    return {"status": "running"}


@app.post("/api/v1/admin/index/activate")
def activate_generation(body: ActivateGenerationRequest) -> dict:
    from rag_assistant.ingestion.reindex import ReindexError, activate

    try:
        pointer = activate(body.generation, settle=False)
    except (ReindexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    auth.audit(
        "index generation activated",
        path="/api/v1/admin/index",
        method="POST",
        outcome=f"generation={pointer.generation or 'legacy'}",
    )
    return {"generation": pointer.generation, "previous": pointer.previous}


@app.post("/api/v1/admin/index/rollback")
def rollback_generation() -> dict:
    from rag_assistant.ingestion.reindex import ReindexError, rollback

    try:
        pointer = rollback()
    except ReindexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    auth.audit(
        "index generation rolled back",
        path="/api/v1/admin/index",
        method="POST",
        outcome=f"generation={pointer.generation or 'legacy'}",
    )
    return {"generation": pointer.generation, "previous": pointer.previous}


@app.get("/api/v1/tenant/usage", response_model=TenantUsageResponse)
@limiter.limit(_per_ip_limit)
def tenant_usage(request: Request) -> TenantUsageResponse:
    """What the calling tenant currently occupies in the index, and what it has spent today.

    Scoped to the caller for the same reason the purge endpoint is: reading another tenant's
    footprint is an administrative question, and it should not share an endpoint with reading
    your own.
    """
    owner = auth.get_owner()
    usage = tenancy.tenant_usage(owner)
    return TenantUsageResponse(
        owner=owner,
        sources=usage.sources,
        chunks=usage.chunks,
        corpus_bytes=usage.corpus_bytes,
        tokens_used_today=usage.tokens_used_today,
        daily_token_budget=usage.daily_token_budget,
    )


@app.delete("/api/v1/tenant/data", response_model=TenantPurgeResponse)
@limiter.limit(_per_ip_limit)
async def purge_tenant_data(request: Request) -> TenantPurgeResponse:
    """Erases everything belonging to the calling tenant: indexed documents, embeddings,
    parent sections, manifest entries, conversations and feedback.

    Scoped to the caller rather than taking an owner parameter. A tenant may erase their own
    data; erasing someone else's is an administrative action that should not share an
    endpoint with it, where a single wrong argument is the difference. DELETE maps to the
    `write` scope through the usual middleware, so a read-only key gets a 403.
    """
    owner = auth.get_owner()
    try:
        result = tenancy.purge_tenant(owner)
    except Exception as exc:
        logger.exception("tenant purge failed for owner=%r", owner)
        raise HTTPException(status_code=500, detail=f"Purge failed: {exc}") from exc

    auth.audit(
        "tenant data purged",
        path="/api/v1/tenant/data",
        method="DELETE",
        outcome=f"sources={result.sources},conversations={result.conversations}",
    )
    return TenantPurgeResponse(
        owner=owner,
        sources_removed=result.sources,
        chunks_removed=result.chunks,
        conversations_removed=result.conversations,
        feedback_removed=result.feedback,
        corpus_files_removed=result.files_removed,
    )


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
        for c in conversations.list_conversations(owner=auth.get_conversation_owner())
    ]


@app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str) -> ConversationDetail:
    conversation = conversations.get_conversation(
        conversation_id, owner=auth.get_conversation_owner()
    )
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
    if not conversations.delete_conversation(conversation_id, owner=auth.get_conversation_owner()):
        raise HTTPException(status_code=404, detail="Unknown conversation.")


# Single-container deployments (Docker image, Render) bake the built frontend into the image
# and point STATIC_DIR at it, so the API serves the whole app from one origin. Mounted last:
# Starlette matches routes in registration order, so every API route above still wins, and
# `html=True` makes "/" serve index.html. In development this is a no-op (STATIC_DIR unset).
_static_dir = get_settings().static_dir
if _static_dir is not None and _static_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")
