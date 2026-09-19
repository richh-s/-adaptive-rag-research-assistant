"""Per-request trace ID plumbing. An ASGI middleware in `api.py` generates one UUID4 per
request and stores it in this ContextVar so any log line -- including ones from deep inside
a synchronous LangGraph node that never sees the request object -- can be tagged with it for
correlation. The graph itself also threads `trace_id` through its state (see `graph/state.py`)
rather than relying solely on contextvar propagation across LangGraph's own task/thread pool
scheduling, which isn't guaranteed to preserve context the same way asyncio.create_task does."""

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


def get_trace_id() -> str | None:
    return trace_id_var.get()


def new_trace_id() -> str:
    return str(uuid.uuid4())


class TraceIdLogFilter(logging.Filter):
    """Stamps every log record with the current request's trace_id unless the call site
    already supplied one via `extra=`, so lines from any module -- not just the ones that
    explicitly pass trace_id -- can still be correlated back to one request."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = get_trace_id()
        return True


# ---- OpenTelemetry ----
#
# The trace_id above correlates *logs*. It cannot tell you where the time went: a request
# that took nine seconds shows up as a set of log lines that each know their own duration and
# nothing about each other. Node timings are recorded too, but only as a flat list on the
# response -- there is no parent/child structure, so "the corrective web search ran because
# grading was slow and returned nothing" is a story you assemble by hand.
#
# Spans give that structure. Kept optional (`uv sync --extra otel`) and off unless an endpoint
# is configured, because a collector is infrastructure and the default deployment has none --
# the same stance as Redis, Postgres and Chroma server mode. Every function below is a no-op
# when the packages are missing or the endpoint is blank, so nothing here can break a
# deployment that never opted in.

_tracer = None
_otel_enabled = False


def configure_otel(service_name: str = "rag-assistant") -> bool:
    """Sets up OTLP export. Returns whether tracing is actually on.

    Idempotent, and deliberately quiet about the ordinary "not configured" case: an operator
    who has not set OTEL_EXPORTER_OTLP_ENDPOINT is not misconfigured, they are running the
    default, and warning about it every boot trains people to ignore warnings.
    """
    global _tracer, _otel_enabled
    if _otel_enabled:
        return True

    from rag_assistant.config import get_settings

    endpoint = get_settings().otel_exporter_otlp_endpoint
    if not endpoint:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        # Configured but not installed is a real misconfiguration, unlike the case above.
        logging.getLogger(__name__).warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but OpenTelemetry is not installed. "
            "Install it with `uv sync --extra otel`; tracing stays off until then."
        )
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    # Batched rather than simple: a span export on the request path would add a network round
    # trip to every node, which is a strange price to pay for observing latency.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)
    _otel_enabled = True
    logging.getLogger(__name__).info("OpenTelemetry tracing enabled, exporting to %s", endpoint)
    return True


def otel_enabled() -> bool:
    return _otel_enabled


def reset_otel() -> None:
    """For tests."""
    global _tracer, _otel_enabled
    _tracer, _otel_enabled = None, False


@contextmanager
def span(name: str, **attributes):
    """A span, or nothing at all when tracing is off.

    A context manager either way so call sites read the same in both cases -- the alternative
    is an `if otel_enabled()` branch around every instrumented block, which is how
    instrumentation drifts out of sync with the code it describes.
    """
    if not _otel_enabled or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current
