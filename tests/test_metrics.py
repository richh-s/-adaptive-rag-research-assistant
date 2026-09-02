"""Tests for the Prometheus metrics layer.

The property worth defending here is bounded label cardinality: a metrics endpoint that mints
a new time series per request URL is worse than no metrics endpoint, because it degrades the
process rather than merely failing to inform. So these assert on the *labels*, not just that
numbers went up.
"""

import importlib
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from prometheus_client import REGISTRY

from rag_assistant import api, metrics
from rag_assistant.config import get_settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def sample(name: str, labels: dict[str, str]) -> float:
    """Current value of one series, or 0.0 when it doesn't exist yet. Counters start absent
    rather than at zero in prometheus_client, so tests compare deltas against this."""
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


# ---- exposition endpoint ----


def test_metrics_endpoint_exposes_registered_metric_families(client: TestClient):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    for family in (
        "rag_http_requests_total",
        "rag_http_request_duration_seconds",
        "rag_llm_calls_total",
        "rag_cache_operations_total",
        "rag_sse_streams_active",
    ):
        assert family in body


def test_metrics_endpoint_absent_when_disabled(monkeypatch):
    """METRICS_ENABLED=false removes the route rather than serving an empty 200 -- a scraper
    pointed at a deployment with metrics off should get an unambiguous 404.

    The route is registered at import time, so this reloads the api module under the flag and
    reloads it back afterwards. Only `api` is reloaded, never `metrics`: re-executing the
    metric definitions would register a second copy of every collector in the global registry
    and raise `Duplicated timeseries`.
    """
    monkeypatch.setenv("METRICS_ENABLED", "false")
    get_settings.cache_clear()
    try:
        importlib.reload(api)
        assert TestClient(api.app).get("/metrics").status_code == 404
    finally:
        # Order matters: undoing the env var is not enough, because `get_settings` is
        # lru_cached and would hand the reloaded module the stale disabled Settings.
        monkeypatch.undo()
        get_settings.cache_clear()
        importlib.reload(api)

    assert TestClient(api.app).get("/metrics").status_code == 200


# ---- label cardinality ----


def test_request_metrics_label_uses_templated_route_not_the_real_path(client: TestClient):
    """A UUID in the path must not become a label value. This is the whole cardinality story:
    100k conversation reads should produce one series, not 100k."""
    labels = {"route": "/api/v1/conversations/{conversation_id}", "method": "GET"}
    before = sample("rag_http_requests_total", {**labels, "status": "4xx"})

    response = client.get("/api/v1/conversations/0123456789abcdef0123456789abcdef")
    assert response.status_code == 404

    assert sample("rag_http_requests_total", {**labels, "status": "4xx"}) == before + 1
    # The concrete id must appear in no series at all.
    assert (
        REGISTRY.get_sample_value(
            "rag_http_requests_total",
            {
                "route": "/api/v1/conversations/0123456789abcdef0123456789abcdef",
                "method": "GET",
                "status": "4xx",
            },
        )
        is None
    )


def test_unmatched_paths_collapse_to_one_series(client: TestClient):
    """404s on random URLs are exactly what an internet-facing service gets scanned with;
    each one must not mint its own label value."""
    labels = {"route": metrics.UNMATCHED_ROUTE, "method": "GET", "status": "4xx"}
    before = sample("rag_http_requests_total", labels)

    for path in ("/wp-login.php", "/.env", "/admin/config"):
        assert client.get(path).status_code == 404

    assert sample("rag_http_requests_total", labels) == before + 3
    assert (
        REGISTRY.get_sample_value("rag_http_requests_total", {**labels, "route": "/.env"}) is None
    )


def test_status_is_recorded_as_a_class_not_an_exact_code(client: TestClient):
    before = sample(
        "rag_http_requests_total", {"route": "/health", "method": "GET", "status": "2xx"}
    )

    client.get("/health")

    after = sample(
        "rag_http_requests_total", {"route": "/health", "method": "GET", "status": "2xx"}
    )
    assert after == before + 1
    assert (
        REGISTRY.get_sample_value(
            "rag_http_requests_total", {"route": "/health", "method": "GET", "status": "200"}
        )
        is None
    )


def test_request_duration_is_observed(client: TestClient):
    labels = {"route": "/health", "method": "GET"}
    before = sample("rag_http_request_duration_seconds_count", labels)

    client.get("/health")

    assert sample("rag_http_request_duration_seconds_count", labels) == before + 1


# ---- LLM callback handler ----


def _llm_result(input_tokens: int, output_tokens: int) -> LLMResult:
    message = AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_callback_handler_records_success_latency_and_tokens():
    handler = metrics.MetricsCallbackHandler("anthropic", "claude-sonnet-5")
    labels = {"provider": "anthropic", "model": "claude-sonnet-5"}
    before_calls = sample("rag_llm_calls_total", {**labels, "outcome": "ok"})
    before_input = sample("rag_llm_tokens_total", {**labels, "kind": "input"})
    before_output = sample("rag_llm_tokens_total", {**labels, "kind": "output"})

    run_id = uuid.uuid4()
    handler.on_chat_model_start({}, [], run_id=run_id)
    handler.on_llm_end(_llm_result(120, 45), run_id=run_id)

    assert sample("rag_llm_calls_total", {**labels, "outcome": "ok"}) == before_calls + 1
    assert sample("rag_llm_tokens_total", {**labels, "kind": "input"}) == before_input + 120
    assert sample("rag_llm_tokens_total", {**labels, "kind": "output"}) == before_output + 45
    assert sample("rag_llm_call_duration_seconds_count", labels) >= 1


def test_callback_handler_records_errors_separately():
    """The fallback chain swallows provider failures by design, so this counter is the only
    place a total outage of the primary provider becomes visible."""
    handler = metrics.MetricsCallbackHandler("local", "gemma-4-26b")
    labels = {"provider": "local", "model": "gemma-4-26b", "outcome": "error"}
    before = sample("rag_llm_calls_total", labels)

    run_id = uuid.uuid4()
    handler.on_chat_model_start({}, [], run_id=run_id)
    handler.on_llm_error(RuntimeError("connection refused"), run_id=run_id)

    assert sample("rag_llm_calls_total", labels) == before + 1


def test_callback_handler_never_raises_into_the_llm_call():
    """A metrics bug must not become a user-visible 500, so the handler swallows its own
    failures -- here, a response shaped in a way the token extractor doesn't expect."""
    handler = metrics.MetricsCallbackHandler("gemini", "gemini-2.5-flash")
    run_id = uuid.uuid4()
    handler.on_chat_model_start({}, [], run_id=run_id)

    handler.on_llm_end(object(), run_id=run_id)  # type: ignore[arg-type]


def test_token_extraction_handles_openai_style_usage():
    """Self-hosted OpenAI-compatible servers report usage in llm_output, not on the message."""
    response = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="hi"))]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    )

    assert metrics._extract_token_usage(response) == (10, 3)


def test_token_extraction_returns_zeros_when_usage_is_unreported():
    """A server that reports nothing leaves a gap in the cost graph -- never an exception."""
    response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="hi"))]])

    assert metrics._extract_token_usage(response) == (0, 0)


# ---- cache + graph counters ----


def test_cache_namespace_label_excludes_the_key_digest():
    """`v1:router:<sha256>` must label as `router`; labelling by full key would be one series
    per distinct question asked."""
    assert metrics.route_label({}) == metrics.UNMATCHED_ROUTE

    from rag_assistant.cache import _namespace_of, cache_key

    assert _namespace_of(cache_key("router", "who founded anthropic")) == "router"
    assert _namespace_of("malformed") == "unknown"


def test_disabled_cache_is_recorded_as_its_own_result():
    """USE_CACHE=false is the test/offline default -- distinguishing it from a genuine miss
    keeps a hit-rate dashboard honest instead of showing 0% and looking broken."""
    from rag_assistant.cache import cache_get, cache_key

    labels = {"namespace": "synthesis", "result": "disabled"}
    before = sample("rag_cache_operations_total", labels)

    assert cache_get(cache_key("synthesis", "q")) is None

    assert sample("rag_cache_operations_total", labels) == before + 1
