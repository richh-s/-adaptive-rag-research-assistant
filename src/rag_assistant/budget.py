"""Per-tenant token budgets.

Rate limiting caps how *often* a tenant can ask. It says nothing about what an ask costs, and
in this pipeline the spread is wide: a question the router sends to `none` is one short
completion, while a compound question decomposed into four sub-queries, retrieved down two
paths, graded per document, corrected via web search and synthesized over a 6,000-token
context is orders of magnitude more. Ten requests a minute is a sentence that means nothing
to a bill.

Token usage was already *metered* -- `metrics.llm_tokens_total` has carried it by provider
and model from the start -- but metering is not enforcement, and a Prometheus counter cannot
be consulted to decide whether to serve a request.

Two design points worth stating, because both are trade-offs rather than oversights:

* **Checked before, charged after.** The cost of a run is not knowable until it has run, so a
  tenant on their last thousand tokens can overshoot by one request. Pre-authorising an
  estimate would bound that, at the price of a reservation protocol, refunds on the common
  path, and an estimate that is still wrong. Overshooting by at most one request is the
  cheaper error, and the budget is a cost control rather than a hard quota.

* **Redis when present, per-process otherwise.** Redis makes the counter shared, which is what
  a multi-replica deployment needs. Without it the counter is per-process, which under N
  replicas means the effective budget is N times the configured one -- so the fallback is
  logged once at startup rather than silently pretending to enforce. A budget that is
  quietly wrong is worse than one that is off, because it is trusted.

`TENANT_DAILY_TOKEN_BUDGET=0` (the default) disables all of this, and the code path is never
reached.
"""

import logging
import threading
import time
from datetime import UTC, datetime
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from rag_assistant.cache import get_redis_client
from rag_assistant.config import get_settings
from rag_assistant.metrics import _extract_token_usage

logger = logging.getLogger(__name__)

# Two days, so a counter written just before midnight UTC survives long enough to be read
# back on the same calendar day it belongs to, and expires on its own afterwards.
_KEY_TTL_SECONDS = 2 * 24 * 60 * 60

_LOCK = threading.Lock()
_local_counters: dict[str, tuple[int, float]] = {}
_warned_local = False


class BudgetExceeded(Exception):
    """Raised when a tenant has spent its allowance for the current UTC day."""

    def __init__(self, owner: str, used: int, budget: int):
        self.owner = owner
        self.used = used
        self.budget = budget
        super().__init__(
            f"Daily token budget exhausted: {used:,} of {budget:,} tokens used. "
            "The budget resets at 00:00 UTC."
        )


def _budget_key(owner: str, day: str | None = None) -> str:
    return f"v1:budget:{owner}:{day or datetime.now(UTC).strftime('%Y-%m-%d')}"


def reset_budget_state() -> None:
    """Drops the per-process counters. For tests; the Redis keys expire on their own."""
    global _warned_local
    with _LOCK:
        _local_counters.clear()
        _warned_local = False


def _warn_once_about_local_counters() -> None:
    global _warned_local
    if _warned_local:
        return
    _warned_local = True
    logger.warning(
        "TENANT_DAILY_TOKEN_BUDGET is set but Redis is unavailable, so the budget is counted "
        "per process. Across N replicas the effective budget is N times the configured one."
    )


def used_tokens(owner: str) -> int:
    """Tokens charged to `owner` so far today. Best-effort: a Redis failure reports 0 rather
    than failing the request, because a monitoring dependency must not become an availability
    dependency -- the budget is a cost control, not an authorization boundary."""
    key = _budget_key(owner)
    client = get_redis_client()
    if client is not None:
        try:
            return int(client.get(key) or 0)
        except Exception:
            logger.warning("Could not read the token budget from Redis", exc_info=True)
            return 0
    _warn_once_about_local_counters()
    with _LOCK:
        entry = _local_counters.get(key)
        if entry is None or entry[1] < time.time():
            return 0
        return entry[0]


def charge(owner: str, tokens: int) -> int:
    """Adds `tokens` to today's total and returns the new total."""
    if tokens <= 0:
        return used_tokens(owner)
    key = _budget_key(owner)
    client = get_redis_client()
    if client is not None:
        try:
            # INCRBY then EXPIRE rather than a read-modify-write: two replicas charging the
            # same tenant at the same moment must not each read the same total and write the
            # same successor. EXPIRE is reapplied on every charge, which is harmless and
            # cheaper than checking whether it is already set.
            total = int(client.incrby(key, tokens))
            client.expire(key, _KEY_TTL_SECONDS)
            return total
        except Exception:
            logger.warning("Could not record token spend in Redis", exc_info=True)
            return 0
    _warn_once_about_local_counters()
    with _LOCK:
        entry = _local_counters.get(key)
        current = entry[0] if entry and entry[1] >= time.time() else 0
        total = current + tokens
        _local_counters[key] = (total, time.time() + _KEY_TTL_SECONDS)
        return total


def charge_ingest(owner: str, embedded_chars: int, vision_calls: int) -> int:
    """Charges an ingest's cost: embeddings plus vision calls.

    Neither reports usage the way a chat completion does. Embedding models do not surface
    token counts through LangChain's callbacks at all, and a vision call's cost depends on
    image resolution rather than anything visible here -- so both are estimated, and the
    estimate lives in one place instead of at every call site.

    Estimated deliberately high rather than low. This is a spend *cap*: an estimate that
    under-counts lets a tenant exceed the budget they were given, which is the failure the
    budget exists to prevent, while over-counting only makes the cap slightly conservative.

    Ingest is the expensive path, not the cheap one -- a 25MB PDF with PDF_VISION on is one
    vision call per figure and per scanned page, which is why leaving it unbudgeted made the
    budget cover mostly the wrong thing.
    """
    settings = get_settings()
    embedding_tokens = int(embedded_chars / max(settings.synthesis_chars_per_token, 1))
    vision_tokens = vision_calls * settings.vision_call_token_estimate
    total = embedding_tokens + vision_tokens
    if total <= 0:
        return used_tokens(owner)
    logger.info(
        "charging ingest to %s: %d embedding + %d vision tokens",
        owner,
        embedding_tokens,
        vision_tokens,
    )
    return charge(owner, total)


def enforce(owner: str) -> None:
    """Raises `BudgetExceeded` when `owner` has already spent its allowance today."""
    budget = get_settings().tenant_daily_token_budget
    if budget <= 0:
        return
    used = used_tokens(owner)
    if used >= budget:
        raise BudgetExceeded(owner=owner, used=used, budget=budget)


class TokenAccountant(BaseCallbackHandler):
    """Totals the tokens one graph run spends, across every node and provider.

    Handed to the graph through `config={"callbacks": [...]}` rather than read from a
    contextvar. LangGraph propagates config callbacks into every node and every nested LLM
    call, while contextvars are not guaranteed to survive its internal thread scheduling --
    the same reason `trace_id` is threaded through the graph state explicitly rather than
    relied upon from context (see api.py's TraceIdMiddleware).
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def on_llm_end(self, response: LLMResult, *, run_id: UUID | None = None, **kwargs) -> None:
        try:
            # Reuses the metrics module's extractor rather than reimplementing it: the shapes
            # differ per provider (Anthropic reports usage on the message, OpenAI-compatible
            # servers in llm_output) and two copies of that knowledge would drift.
            input_tokens, output_tokens = _extract_token_usage(response)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
        except Exception:
            # Never fail a served answer over accounting.
            logger.warning("Could not total token usage for this run", exc_info=True)
