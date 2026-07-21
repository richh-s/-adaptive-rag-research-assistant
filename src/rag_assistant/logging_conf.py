import logging

from pythonjsonlogger.json import JsonFormatter

from rag_assistant.tracing import TraceIdLogFilter

# Every line gets these keys regardless of call site (null if not applicable to that line) --
# trace_id via TraceIdLogFilter, node/route/latency_ms via extra= at the specific call sites
# that have them (the trace middleware and the per-node timing wrapper).
_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s %(node)s %(route)s %(latency_ms)s"
)


def configure_logging(level: int | None = None) -> None:
    if level is None:
        from rag_assistant.config import get_settings

        level = getattr(logging, get_settings().log_level.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(_LOG_FORMAT))
    handler.addFilter(TraceIdLogFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
