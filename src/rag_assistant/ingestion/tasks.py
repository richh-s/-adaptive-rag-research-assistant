"""Registry tracking the lifecycle of a background ingest job, so `POST /api/v1/ingest` can
hand the caller a `task_id` immediately and the caller can poll `GET /api/v1/ingest/{task_id}`
for real progress instead of guessing when the background task finishes.

Two backends. The default is an in-memory dict, which is correct and free for a single-worker
deployment. `TASK_BACKEND=redis` moves the same state into Redis so every replica sees it --
without which a client polling a load-balanced deployment gets "unknown ingest task" 404s
roughly (workers - 1) / workers of the time, because the task exists only in the process that
happened to accept the upload.

Both implement the same four functions, so nothing that uses them knows which is running.
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from rag_assistant.cache import get_redis_client
from rag_assistant.config import get_settings
from rag_assistant.metrics import record_ingest_task

logger = logging.getLogger(__name__)

IngestStage = Literal["queued", "parsing", "indexing", "indexed", "failed"]

TERMINAL_STAGES: frozenset[IngestStage] = frozenset({"indexed", "failed"})

# Bounds memory growth across the life of a long-running process -- old entries are evicted
# oldest-first once the registry fills up, same tradeoff as any fixed-size cache.
_MAX_TASKS = 500


@dataclass
class IngestTask:
    task_id: str
    filename: str
    original_filename: str
    stage: IngestStage = "queued"
    message: str = "Waiting to start."
    error: str | None = None
    indexed_chunks: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


_tasks: dict[str, IngestTask] = {}
_task_order: list[str] = []
_lock = threading.Lock()

# Redis keys are namespaced and expire on their own: an ingest task is only interesting while
# a client is polling it, and a TTL is a simpler bound than the in-memory registry's
# oldest-first eviction because Redis applies it without anything having to run.
_REDIS_PREFIX = "v1:ingest-task"
_REDIS_TTL_SECONDS = 24 * 3600


def _use_redis() -> bool:
    return get_settings().task_backend == "redis"


def _redis_key(task_id: str) -> str:
    return f"{_REDIS_PREFIX}:{task_id}"


def _to_payload(task: IngestTask) -> str:
    return json.dumps(asdict(task))


def _from_payload(payload: str) -> IngestTask:
    return IngestTask(**json.loads(payload))


def create_task(filename: str, original_filename: str) -> IngestTask:
    task = IngestTask(
        task_id=uuid.uuid4().hex, filename=filename, original_filename=original_filename
    )
    if _use_redis():
        client = get_redis_client()
        if client is not None:
            client.setex(_redis_key(task.task_id), _REDIS_TTL_SECONDS, _to_payload(task))
            return task
        # Redis configured but unreachable: fall through to memory rather than failing the
        # upload. The task becomes invisible to other replicas, which degrades polling on
        # this one job -- strictly better than refusing to ingest at all.
        logger.warning("TASK_BACKEND=redis but Redis is unavailable; using in-memory tasks")
    with _lock:
        _tasks[task.task_id] = task
        _task_order.append(task.task_id)
        while len(_task_order) > _MAX_TASKS:
            _tasks.pop(_task_order.pop(0), None)
    return task


def _apply_update(
    task: IngestTask,
    stage: IngestStage | None,
    message: str | None,
    error: str | None,
    indexed_chunks: int | None,
) -> IngestTask:
    if stage is not None:
        # Counted on the transition, not on every update, so a task that reports progress
        # while already "indexing" doesn't inflate the count.
        if stage != task.stage and stage in TERMINAL_STAGES:
            record_ingest_task(stage)
        task.stage = stage
    if message is not None:
        task.message = message
    if error is not None:
        task.error = error
    if indexed_chunks is not None:
        task.indexed_chunks = indexed_chunks
    task.updated_at = time.time()
    return task


def update_task(
    task_id: str,
    *,
    stage: IngestStage | None = None,
    message: str | None = None,
    error: str | None = None,
    indexed_chunks: int | None = None,
) -> None:
    if _use_redis():
        client = get_redis_client()
        if client is not None:
            raw = client.get(_redis_key(task_id))
            if raw is None:
                return
            task = _apply_update(_from_payload(raw), stage, message, error, indexed_chunks)
            client.setex(_redis_key(task_id), _REDIS_TTL_SECONDS, _to_payload(task))
            return
    with _lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        _apply_update(task, stage, message, error, indexed_chunks)


def get_task(task_id: str) -> IngestTask | None:
    """Returns a snapshot copy, not the live object, so a caller iterating over its fields
    can't observe a partial update landing concurrently from `update_task`."""
    if _use_redis():
        client = get_redis_client()
        if client is not None:
            raw = client.get(_redis_key(task_id))
            return _from_payload(raw) if raw else None
    with _lock:
        task = _tasks.get(task_id)
        return replace(task) if task is not None else None


def reset_tasks() -> None:
    """Drops in-memory task state. For tests; the Redis backend expires on its own."""
    with _lock:
        _tasks.clear()
        _task_order.clear()
