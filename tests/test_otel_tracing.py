"""Tests for optional OpenTelemetry tracing.

The property that matters most is the negative one: with no endpoint configured -- the
default, and what the great majority of deployments will run -- nothing is imported, nothing
is exported, and no call site behaves differently. Instrumentation that can break the
un-instrumented path is worse than none.
"""

import pytest

from rag_assistant import tracing
from rag_assistant.config import get_settings

try:
    import opentelemetry.sdk.trace  # noqa: F401

    HAS_OTEL_SDK = True
except ImportError:
    HAS_OTEL_SDK = False

# The SDK is an optional extra, so the tests that exercise real spans skip without it. The
# tests above them deliberately do not: "tracing is off and everything still works" is the
# configuration CI's default job runs, and is the one that must never skip.
needs_sdk = pytest.mark.skipif(not HAS_OTEL_SDK, reason="requires `uv sync --extra otel`")


@pytest.fixture(autouse=True)
def _reset_otel():
    tracing.reset_otel()
    yield
    tracing.reset_otel()


def test_tracing_is_off_by_default(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    get_settings.cache_clear()

    assert tracing.configure_otel() is False
    assert tracing.otel_enabled() is False


def test_span_is_a_usable_no_op_when_tracing_is_off():
    """Call sites use the same `with span(...)` in both modes, so the disabled path has to be
    a working context manager rather than something that needs guarding."""
    with tracing.span("node.route_query", **{"rag.node": "route_query"}) as current:
        assert current is None


def test_a_no_op_span_does_not_swallow_exceptions():
    with pytest.raises(ValueError):
        with tracing.span("node.boom"):
            raise ValueError("node failed")


@needs_sdk
def test_configuring_with_an_endpoint_enables_tracing(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    get_settings.cache_clear()

    assert tracing.configure_otel() is True
    assert tracing.otel_enabled() is True


@needs_sdk
def test_configuration_is_idempotent(monkeypatch):
    """Called from the lifespan, which a test client may start more than once per process --
    a second provider would silently orphan the first one's exporter."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    get_settings.cache_clear()

    assert tracing.configure_otel() is True
    assert tracing.configure_otel() is True


@needs_sdk
def test_spans_are_recorded_when_enabled(monkeypatch):
    """Exported through an in-memory exporter rather than a real collector: this asserts the
    spans are produced and named, which is what the code here controls."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer(__name__))
    monkeypatch.setattr(tracing, "_otel_enabled", True)

    with tracing.span("node.retrieve_vector", **{"rag.node": "retrieve_vector"}):
        pass

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["node.retrieve_vector"]
    assert spans[0].attributes["rag.node"] == "retrieve_vector"


@needs_sdk
def test_none_attributes_are_dropped(monkeypatch):
    """`route` is None until the router has run, and OTel rejects a None attribute value."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer(__name__))
    monkeypatch.setattr(tracing, "_otel_enabled", True)

    with tracing.span("node.condense_question", **{"rag.route": None, "rag.node": "condense"}):
        pass

    attributes = exporter.get_finished_spans()[0].attributes
    assert "rag.route" not in attributes
    assert attributes["rag.node"] == "condense"


@needs_sdk
def test_graph_nodes_emit_one_span_each(monkeypatch):
    """Instrumentation lives in the shared `_timed` wrapper, so this checks it is actually on
    the path a node takes rather than merely available."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from rag_assistant.graph.build_graph import _timed

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer(__name__))
    monkeypatch.setattr(tracing, "_otel_enabled", True)

    wrapped = _timed("route_query", lambda state: {"route": "vector"})
    result = wrapped({"trace_id": "abc", "route": None})

    assert [s.name for s in exporter.get_finished_spans()] == ["node.route_query"]
    # The wrapper's existing contract is unchanged by the instrumentation.
    assert result["route"] == "vector"
    assert result["node_timings"][0]["node"] == "route_query"


@needs_sdk
def test_a_node_raising_still_closes_its_span(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from rag_assistant.graph.build_graph import _timed

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer(__name__))
    monkeypatch.setattr(tracing, "_otel_enabled", True)

    def _boom(state):
        raise RuntimeError("node failed")

    with pytest.raises(RuntimeError):
        _timed("web_search", _boom)({"trace_id": "abc"})

    assert [s.name for s in exporter.get_finished_spans()] == ["node.web_search"]
