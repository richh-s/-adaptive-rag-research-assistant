"""Prometheus metrics.

The structured logs (logging_conf.py) already answer "what happened on this one request" --
this module answers the questions logs are bad at: what is the p95 latency of /research right
now, what fraction of LLM calls are falling back to the secondary provider, is the cache
actually being hit, how much is this thing costing per hour. Those are aggregates, and
grepping JSON logs for them is a worse version of a counter.

Cardinality is the failure mode a metrics layer usually dies of, so every label here is drawn
from a bounded set: route labels use Starlette's *templated* path (`/api/v1/conversations/
{conversation_id}`, never the UUID-bearing real path), provider/model come from settings, and
anything unrecognized collapses to a single sentinel bucket rather than minting a new series.

Metrics are process-local. That is exactly right here -- the deployment is pinned to one
uvicorn worker (embedded Chroma's SQLite lock allows nothing else; see docker-compose.yml),
so one process is the whole service and there is no multiprocess collector to reconcile.
"""

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

# Label used whenever a value would otherwise be unbounded (a 404 path, an unknown provider).
# Collapsing to one sentinel series is the difference between a metrics endpoint and a memory
# leak that a scraper walks into.
UNKNOWN = "<unknown>"
UNMATCHED_ROUTE = "<unmatched>"

# Latency buckets are tuned for this workload, not Prometheus' web-server defaults: a research
# call runs a multi-node LangGraph with several LLM round-trips, so the interesting region is
# seconds-to-a-minute, and GRAPH_TIMEOUT_SECONDS (45s default) should land inside a bucket
# edge rather than in an open +Inf tail where a timeout is indistinguishable from a hang.
_REQUEST_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 120.0)
_LLM_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 45.0, 90.0, 180.0)

http_requests_total = Counter(
    "rag_http_requests_total",
    "HTTP requests completed, by templated route, method and status class.",
    ["route", "method", "status"],
)

http_request_duration_seconds = Histogram(
    "rag_http_request_duration_seconds",
    "Wall-clock time to serve an HTTP request, by templated route.",
    ["route", "method"],
    buckets=_REQUEST_BUCKETS,
)

# SSE streams are long-lived, so they are counted separately from the request histogram --
# a /research/stream connection's "duration" is the length of the answer, not server latency,
# and mixing the two would make the request histogram meaningless.
sse_streams_active = Gauge(
    "rag_sse_streams_active",
    "SSE research streams currently open.",
)

graph_runs_total = Counter(
    "rag_graph_runs_total",
    "Completed graph executions, by chosen route and outcome.",
    ["route", "outcome"],
)

graph_node_duration_seconds = Histogram(
    "rag_graph_node_duration_seconds",
    "Per-node execution time inside the LangGraph pipeline.",
    ["node"],
    buckets=_REQUEST_BUCKETS,
)

# `outcome` separates a provider that answered from one that raised and got fallen back to --
# the fallback path is silent by design (llm.py degrades rather than failing), so without this
# counter a total Anthropic outage looks identical to normal operation apart from the bill.
llm_calls_total = Counter(
    "rag_llm_calls_total",
    "LLM invocations, by provider, model and outcome (ok/error).",
    ["provider", "model", "outcome"],
)

llm_call_duration_seconds = Histogram(
    "rag_llm_call_duration_seconds",
    "LLM invocation latency, by provider and model.",
    ["provider", "model"],
    buckets=_LLM_BUCKETS,
)

# The one metric that maps directly onto money. Everything else here answers "is it healthy";
# this answers "what is it costing", which for an LLM app is the question that actually
# arrives unannounced at the end of a month.
llm_tokens_total = Counter(
    "rag_llm_tokens_total",
    "Tokens consumed, by provider, model and kind (input/output).",
    ["provider", "model", "kind"],
)

cache_operations_total = Counter(
    "rag_cache_operations_total",
    "Cache lookups by namespace and result (hit/miss/error/disabled).",
    ["namespace", "result"],
)

ingest_tasks_total = Counter(
    "rag_ingest_tasks_total",
    "Background ingestion jobs reaching a terminal stage.",
    ["stage"],
)


# The only metric sourced from a human rather than from the system's own behaviour. Everything
# else here measures whether the service is working; this measures whether it is any *good*,
# which no amount of latency and error-rate data can tell you.
feedback_total = Counter(
    "rag_feedback_total",
    "User ratings of answers, by rating.",
    ["rating"],
)


def record_feedback(rating: str) -> None:
    feedback_total.labels(rating=rating).inc()


def render() -> tuple[bytes, str]:
    """The exposition payload plus its content type, for the /metrics handler."""
    return generate_latest(), CONTENT_TYPE_LATEST


def route_label(scope: dict) -> str:
    """The templated path for a matched request (`/api/v1/conversations/{conversation_id}`).

    Starlette stores the matched route on the ASGI scope during routing, which happens
    *inside* the app call -- so this is only meaningful once the response has started, and
    unmatched requests (404s, and any probe hammering random URLs) collapse to one sentinel
    series instead of minting a label value per path.
    """
    route = scope.get("route")
    path_format = getattr(route, "path_format", None) or getattr(route, "path", None)
    return path_format or UNMATCHED_ROUTE


def observe_request(scope: dict, status_code: int, duration_seconds: float) -> None:
    route = route_label(scope)
    method = scope.get("method", UNKNOWN)
    # Status *class* ("2xx"), not the exact code: the useful alert is "error rate rose", and
    # a per-code label would fan out across every status the app can emit for little gain.
    http_requests_total.labels(route=route, method=method, status=f"{status_code // 100}xx").inc()
    http_request_duration_seconds.labels(route=route, method=method).observe(duration_seconds)


def _extract_token_usage(response: LLMResult) -> tuple[int, int]:
    """Input/output token counts out of an LLMResult, across provider shapes.

    There is no single place to read this: Anthropic and Gemini put `usage_metadata` on the
    message, OpenAI-compatible servers report `token_usage` in `llm_output`, and a
    self-hosted server may report nothing at all. Missing counts return (0, 0) -- an
    unreported usage is a gap in the cost graph, never a failed request.
    """
    llm_output = response.llm_output or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        return (
            int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        )
    for generations in response.generations:
        for generation in generations:
            message_usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if message_usage:
                return (
                    int(message_usage.get("input_tokens") or 0),
                    int(message_usage.get("output_tokens") or 0),
                )
    return 0, 0


class MetricsCallbackHandler(BaseCallbackHandler):
    """Records latency, outcome and token usage for one provider's calls.

    A callback handler rather than a wrapper around the call site, because the call sites are
    `RunnableWithFallbacks` chains (see llm.py): by the time a request reaches `get_chat_model()`
    the identity of the provider that actually served it is decided inside LangChain, and a
    timer around the outer runnable would attribute a Gemini fallback's latency to the local
    model that failed first. Each provider is constructed with its *own* handler instance, so
    the labels are correct by construction and a fallback shows up as `outcome="error"` on the
    provider that failed plus `outcome="ok"` on the one that rescued it.

    Never raises: an exception escaping a callback would propagate into the LLM call it was
    only supposed to observe, turning a metrics bug into a user-visible 500.
    """

    # LangChain calls handlers on the same thread as the run it's reporting, but a single
    # handler instance is shared across every call to that provider, so start times are keyed
    # by run_id rather than held in one attribute -- concurrent sub-query retrievals fan out
    # several simultaneous calls through the same model object.
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self._starts: dict[UUID, float] = {}

    def _finish(self, run_id: UUID, outcome: str) -> None:
        start = self._starts.pop(run_id, None)
        llm_calls_total.labels(provider=self.provider, model=self.model, outcome=outcome).inc()
        if start is not None:
            llm_call_duration_seconds.labels(provider=self.provider, model=self.model).observe(
                time.perf_counter() - start
            )

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, **kwargs) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_chat_model_start(
        self, serialized: dict, messages: list, *, run_id: UUID, **kwargs
    ) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs) -> None:
        try:
            self._finish(run_id, "ok")
            input_tokens, output_tokens = _extract_token_usage(response)
            if input_tokens:
                llm_tokens_total.labels(provider=self.provider, model=self.model, kind="input").inc(
                    input_tokens
                )
            if output_tokens:
                llm_tokens_total.labels(
                    provider=self.provider, model=self.model, kind="output"
                ).inc(output_tokens)
        except Exception:
            logger.warning("failed to record LLM metrics", exc_info=True)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            self._finish(run_id, "error")
        except Exception:
            logger.warning("failed to record LLM error metrics", exc_info=True)


def record_cache(namespace: str, result: str) -> None:
    cache_operations_total.labels(namespace=namespace, result=result).inc()


def record_graph_run(route: str | None, outcome: str) -> None:
    graph_runs_total.labels(route=route or UNKNOWN, outcome=outcome).inc()


def record_node_timing(node: str, duration_seconds: float) -> None:
    graph_node_duration_seconds.labels(node=node).observe(duration_seconds)


def record_ingest_task(stage: str) -> None:
    ingest_tasks_total.labels(stage=stage).inc()
