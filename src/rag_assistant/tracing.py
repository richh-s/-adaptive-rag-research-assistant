"""Per-request trace ID plumbing. An ASGI middleware in `api.py` generates one UUID4 per
request and stores it in this ContextVar so any log line -- including ones from deep inside
a synchronous LangGraph node that never sees the request object -- can be tagged with it for
correlation. The graph itself also threads `trace_id` through its state (see `graph/state.py`)
rather than relying solely on contextvar propagation across LangGraph's own task/thread pool
scheduling, which isn't guaranteed to preserve context the same way asyncio.create_task does."""

import logging
import uuid
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
