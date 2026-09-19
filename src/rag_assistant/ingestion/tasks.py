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
    # sha256 of the uploaded bytes, when the caller supplied it. Used to make a repeated
    # upload of identical content return the original task instead of parsing and embedding
    # the same file twice -- see `find_task_by_content`.
    content_hash: str | None = None
    # Who uploaded it, so a restart can resume the work without a request to read it from,
    # and how many times it has been attempted, so retries are bounded rather than a loop.
    owner: str = "public"
    attempts: int = 0
    stage: IngestStage = "queued"
    message: str = "Waiting to start."
    error: str | None = None
    indexed_chunks: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


_tasks: dict[str, IngestTask] = {}
_task_order: list[str] = []
_content_index: dict[str, str] = {}
_lock = threading.Lock()

# Redis keys are namespaced and expire on their own: an ingest task is only interesting while
# a client is polling it, and a TTL is a simpler bound than the in-memory registry's
# oldest-first eviction because Redis applies it without anything having to run.
_REDIS_PREFIX = "v1:ingest-task"
_REDIS_CONTENT_PREFIX = "v1:ingest-content"
_REDIS_TTL_SECONDS = 24 * 3600


def _use_redis() -> bool:
    return get_settings().task_backend == "redis"


def _redis_key(task_id: str) -> str:
    return f"{_REDIS_PREFIX}:{task_id}"


def _content_key(content_hash: str) -> str:
    return f"{_REDIS_CONTENT_PREFIX}:{content_hash}"


def _to_payload(task: IngestTask) -> str:
    return json.dumps(asdict(task))


def _from_payload(payload: str) -> IngestTask:
    return IngestTask(**json.loads(payload))


def create_task(
    filename: str,
    original_filename: str,
    content_hash: str | None = None,
    owner: str = "public",
) -> IngestTask:
    task = IngestTask(
        task_id=uuid.uuid4().hex,
        filename=filename,
        original_filename=original_filename,
        content_hash=content_hash,
        owner=owner,
    )
    if _use_redis():
        client = get_redis_client()
        if client is not None:
            client.setex(_redis_key(task.task_id), _REDIS_TTL_SECONDS, _to_payload(task))
            if content_hash:
                client.setex(_content_key(content_hash), _REDIS_TTL_SECONDS, task.task_id)
            return task
        # Redis configured but unreachable: fall through to memory rather than failing the
        # upload. The task becomes invisible to other replicas, which degrades polling on
        # this one job -- strictly better than refusing to ingest at all.
        logger.warning("TASK_BACKEND=redis but Redis is unavailable; using in-memory tasks")
    with _lock:
        _tasks[task.task_id] = task
        _task_order.append(task.task_id)
        if content_hash:
            _content_index[content_hash] = task.task_id
        while len(_task_order) > _MAX_TASKS:
            evicted = _task_order.pop(0)
            evicted_task = _tasks.pop(evicted, None)
            # The content index has to shrink with the registry, or it accumulates hashes
            # pointing at tasks that no longer exist and every lookup misses anyway.
            if evicted_task is not None and evicted_task.content_hash:
                _content_index.pop(evicted_task.content_hash, None)
    return task


def _apply_update(
    task: IngestTask,
    stage: IngestStage | None,
    message: str | None,
    error: str | None,
    indexed_chunks: int | None,
    attempts: int | None = None,
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
    if attempts is not None:
        task.attempts = attempts
    task.updated_at = time.time()
    return task


def update_task(
    task_id: str,
    *,
    stage: IngestStage | None = None,
    message: str | None = None,
    error: str | None = None,
    indexed_chunks: int | None = None,
    attempts: int | None = None,
) -> None:
    if _use_redis():
        client = get_redis_client()
        if client is not None:
            raw = client.get(_redis_key(task_id))
            if raw is None:
                return
            task = _apply_update(
                _from_payload(raw), stage, message, error, indexed_chunks, attempts
            )
            client.setex(_redis_key(task_id), _REDIS_TTL_SECONDS, _to_payload(task))
            return
    with _lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        _apply_update(task, stage, message, error, indexed_chunks, attempts)


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


def find_task_by_content(content_hash: str) -> IngestTask | None:
    """An existing task for identical bytes, if one is still on record.

    Uploading the same file twice is not hypothetical -- it is what a client does after a
    timeout, a lost connection, or a user double-clicking. Without this, the second upload
    re-parses the file (pymupdf4llm, plus a vision call per figure with PDF_VISION on) and
    re-embeds every chunk, to arrive at exactly the state the first one produced.

    Deliberately keyed on content rather than filename: the same name may hold new bytes, and
    new bytes are precisely when the work *is* needed. `build_index` already skips unchanged
    files by fingerprint, so this is not the only guard -- but it saves the upload, the stage
    and the duplicate task record on top of that.
    """
    if not content_hash:
        return None
    if _use_redis():
        client = get_redis_client()
        if client is not None:
            task_id = client.get(_content_key(content_hash))
            if task_id is None:
                return None
            if isinstance(task_id, bytes):
                task_id = task_id.decode()
            return get_task(task_id)
    with _lock:
        task_id = _content_index.get(content_hash)
    return get_task(task_id) if task_id else None


def reconcile_stale_tasks() -> list["IngestTask"]:
    """Recovers tasks left mid-flight by a restart. Returns the ones worth resuming.

    An ingest runs as a FastAPI background task inside the process that accepted the upload.
    If that process dies -- deploy, OOM, crash -- the work stops but the record does not: the
    task sits at `parsing` forever while a client polls a job nobody is doing.

    The work *is* resumable, which is the thing worth noticing. The uploaded file was written
    into the corpus before the task was created, and `build_index` is incremental and decides
    what to do from a fingerprint -- so re-running it after a crash costs only the files that
    were not finished, and re-running it after a *completed* ingest costs nothing at all.
    That makes "resume" strictly better than the honest-but-unhelpful "re-upload and start
    over" this used to do.

    Retries are bounded by INGEST_MAX_ATTEMPTS, because a file that crashes the parser will
    crash it again: without a ceiling, a poison upload becomes an infinite restart loop that
    looks like an unstable deployment. A task that exhausts its attempts goes to a terminal
    `failed` carrying the attempt count -- the record is the dead-letter queue, already
    queryable through the status endpoint.

    Called at startup. Stale means "non-terminal and untouched for longer than
    INGEST_STALE_AFTER_SECONDS", which has to exceed the longest legitimate gap between
    stage updates, or a large corpus gets resumed out from under itself mid-index.
    """
    settings = get_settings()
    cutoff = time.time() - settings.ingest_stale_after_seconds
    resumable: list[IngestTask] = []
    failed = 0
    for task in _all_tasks():
        if task.stage in TERMINAL_STAGES or task.updated_at >= cutoff:
            continue
        if task.attempts >= settings.ingest_max_attempts:
            update_task(
                task.task_id,
                stage="failed",
                message="Interrupted by a restart, and out of attempts.",
                error=(
                    f"This ingest was interrupted {task.attempts} time(s) and has reached "
                    f"INGEST_MAX_ATTEMPTS. Re-upload the file to try again."
                ),
            )
            failed += 1
            continue
        update_task(
            task.task_id,
            stage="queued",
            message="Interrupted by a restart; queued to resume.",
            attempts=task.attempts,
        )
        resumable.append(task)
    if failed:
        logger.warning("Failed %d ingest task(s) that had exhausted their attempts", failed)
    if resumable:
        logger.info("Resuming %d ingest task(s) left in flight by a restart", len(resumable))
    return resumable


def _all_tasks() -> list[IngestTask]:
    """Every task currently on record, across whichever backend is active."""
    if _use_redis():
        client = get_redis_client()
        if client is not None:
            try:
                tasks = []
                # SCAN rather than KEYS: KEYS blocks the server for the whole keyspace, which
                # is exactly the wrong thing to do at startup on a shared Redis.
                for key in client.scan_iter(match=f"{_REDIS_PREFIX}:*", count=100):
                    raw = client.get(key)
                    if raw:
                        tasks.append(_from_payload(raw))
                return tasks
            except Exception:
                logger.warning("Could not enumerate ingest tasks from Redis", exc_info=True)
                return []
    with _lock:
        return [replace(task) for task in _tasks.values()]


def reset_tasks() -> None:
    """Drops in-memory task state. For tests; the Redis backend expires on its own."""
    with _lock:
        _content_index.clear()
        _tasks.clear()
        _task_order.clear()
