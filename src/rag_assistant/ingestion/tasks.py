"""Thread-safe in-memory registry tracking the lifecycle of a background ingest job, so
`POST /api/v1/ingest` can hand the caller a `task_id` immediately and the caller can poll
`GET /api/v1/ingest/{task_id}` for real progress instead of guessing when the background task
finishes. In-memory and per-process by design: this project runs a single uvicorn worker (see
`cli.py`'s `serve` command), so there's no cross-process visibility problem to solve. Scaling to
multiple workers/replicas would need this moved to Redis (already used elsewhere in this
codebase for caching) so every worker can see the same task state.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Literal

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


def create_task(filename: str, original_filename: str) -> IngestTask:
    task = IngestTask(task_id=uuid.uuid4().hex, filename=filename, original_filename=original_filename)
    with _lock:
        _tasks[task.task_id] = task
        _task_order.append(task.task_id)
        while len(_task_order) > _MAX_TASKS:
            _tasks.pop(_task_order.pop(0), None)
    return task


def update_task(
    task_id: str,
    *,
    stage: IngestStage | None = None,
    message: str | None = None,
    error: str | None = None,
    indexed_chunks: int | None = None,
) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        if stage is not None:
            task.stage = stage
        if message is not None:
            task.message = message
        if error is not None:
            task.error = error
        if indexed_chunks is not None:
            task.indexed_chunks = indexed_chunks
        task.updated_at = time.time()


def get_task(task_id: str) -> IngestTask | None:
    """Returns a snapshot copy, not the live object, so a caller iterating over its fields
    can't observe a partial update landing concurrently from `update_task`."""
    with _lock:
        task = _tasks.get(task_id)
        return replace(task) if task is not None else None
