"""Tests for ingest durability: idempotent uploads and restart reconciliation.

An ingest runs as a FastAPI background task inside the process that accepted the upload.
That has two consequences a client can observe, and both are covered here: the same file
uploaded twice must not be parsed and embedded twice, and a task the process was working on
when it died must not sit at `parsing` forever with nobody doing it.
"""

import time

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api
from rag_assistant.config import get_settings
from rag_assistant.ingestion import tasks
from rag_assistant.ingestion.build_index import IndexResult

FILE = ("cohere.md", b"Cohere builds enterprise retrieval models.", "text/markdown")


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    (tmp_path / "corpus").mkdir(parents=True, exist_ok=True)
    return TestClient(api.app)


@pytest.fixture
def counted_build(monkeypatch):
    """Counts how many times the expensive path actually runs."""
    calls = []

    def _fake_build_index(*, on_stage=None, **kwargs):
        calls.append(kwargs)
        if on_stage:
            on_stage("parsing", "parsing")
            on_stage("indexing", "indexing")
        return IndexResult(indexed_chunks=3, changed_files=1, skipped_files=0, removed_files=0)

    monkeypatch.setattr(api, "build_index", _fake_build_index)
    return calls


# ---- idempotency ----


def test_uploading_the_same_file_twice_reuses_the_first_task(client, counted_build):
    """The retry case: a timeout, a lost connection, or a double-clicked upload button. The
    second upload must not re-parse (pymupdf4llm, plus a vision call per figure) and re-embed
    to arrive at the state the first one already produced."""
    first = client.post("/api/v1/ingest", files={"file": FILE})
    second = client.post("/api/v1/ingest", files={"file": FILE})

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["task_id"] == first.json()["task_id"]
    assert len(counted_build) == 1
    assert "already uploaded" in second.json()["message"]


def test_the_duplicate_is_not_left_behind_in_the_corpus(client, counted_build, tmp_path):
    """Each upload is written under its own UUID suffix before the duplicate is detected, so
    the collapsed one has to be cleaned up or it sits in the corpus as a second copy."""
    client.post("/api/v1/ingest", files={"file": FILE})
    client.post("/api/v1/ingest", files={"file": FILE})

    written = list((tmp_path / "corpus").rglob("cohere*.md"))

    assert len(written) == 1


def test_different_content_under_the_same_name_still_ingests(client, counted_build):
    """Keyed on bytes, not filename -- the same name holding new bytes is exactly when the
    work *is* needed."""
    client.post("/api/v1/ingest", files={"file": ("a.md", b"first version", "text/markdown")})
    second = client.post(
        "/api/v1/ingest", files={"file": ("a.md", b"second version", "text/markdown")}
    )

    assert len(counted_build) == 2
    assert second.json()["message"].startswith("File saved")


def test_a_failed_ingest_can_be_retried_by_re_uploading(client, monkeypatch):
    """A prior failure must not make the content permanently un-ingestable."""
    attempts = []

    def _failing_build(*, on_stage=None, **kwargs):
        attempts.append(1)
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(api, "build_index", _failing_build)
    max_attempts = get_settings().ingest_max_attempts
    first = client.post("/api/v1/ingest", files={"file": FILE})
    first_task = tasks.get_task(first.json()["task_id"])
    assert first_task.stage == "failed"
    # Each upload now exhausts its retry budget before giving up.
    assert first_task.attempts == max_attempts
    assert len(attempts) == max_attempts

    second = client.post("/api/v1/ingest", files={"file": FILE})

    assert second.json()["task_id"] != first.json()["task_id"]
    assert len(attempts) == max_attempts * 2


def test_a_transient_failure_succeeds_on_retry(client, monkeypatch):
    """The case retries exist for: a provider blip should not cost the user their upload."""
    from rag_assistant.ingestion.build_index import IndexResult

    calls = []

    def _flaky_build(*, on_stage=None, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("embedding provider rate-limited")
        return IndexResult(indexed_chunks=3, changed_files=1, skipped_files=0, removed_files=0)

    monkeypatch.setattr(api, "build_index", _flaky_build)

    response = client.post("/api/v1/ingest", files={"file": FILE})
    task = tasks.get_task(response.json()["task_id"])

    assert task.stage == "indexed"
    assert task.attempts == 2
    assert len(calls) == 2


def test_retries_are_bounded(client, monkeypatch):
    """A file that crashes the parser crashes it again; an unbounded retry turns one bad
    upload into a loop that reads as an unstable service."""

    calls = []

    def _always_failing(*, on_stage=None, **kwargs):
        calls.append(1)
        raise RuntimeError("poison file")

    monkeypatch.setattr(api, "build_index", _always_failing)
    monkeypatch.setenv("INGEST_MAX_ATTEMPTS", "2")
    get_settings.cache_clear()

    response = client.post("/api/v1/ingest", files={"file": FILE})
    task = tasks.get_task(response.json()["task_id"])

    assert len(calls) == 2
    assert task.stage == "failed"
    assert "2 attempt(s)" in task.message


def test_two_tenants_uploading_identical_bytes_are_not_collapsed(
    counted_build, monkeypatch, tmp_path
):
    """Collapsing these would put one tenant's document in the other's corpus.

    Authenticated with real API keys rather than by setting `auth.owner_var` directly: the
    contextvar is set by the auth middleware inside the request, and TestClient runs that on
    its own thread, so a value set out here never reaches the handler and both uploads would
    arrive as the public tenant -- the test would pass or fail for reasons unrelated to
    tenancy.
    """
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    (tmp_path / "corpus").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("API_KEYS", "alice:secret-a,bob:secret-b")
    get_settings.cache_clear()
    client = TestClient(api.app)

    alice = client.post("/api/v1/ingest", files={"file": FILE}, headers={"X-API-Key": "secret-a"})
    bob = client.post("/api/v1/ingest", files={"file": FILE}, headers={"X-API-Key": "secret-b"})

    assert alice.status_code == 202
    assert bob.status_code == 202
    assert alice.json()["task_id"] != bob.json()["task_id"]
    assert len(counted_build) == 2


# ---- restart reconciliation ----


def test_a_task_orphaned_by_a_restart_is_queued_for_resumption(monkeypatch):
    """The work is resumable, which is the point.

    The uploaded file was written into the corpus before the task existed, and `build_index`
    decides what to do from a fingerprint -- so re-running after a crash costs only what was
    not finished, and re-running after a completed ingest costs nothing. That makes resuming
    strictly better than the honest-but-unhelpful "re-upload and start over".
    """
    monkeypatch.setenv("INGEST_STALE_AFTER_SECONDS", "60")
    get_settings.cache_clear()
    task = tasks.create_task(filename="big.pdf", original_filename="big.pdf", owner="alice")
    tasks.update_task(task.task_id, stage="parsing", message="Parsing...")
    tasks._tasks[task.task_id].updated_at = time.time() - 3600

    resumable = tasks.reconcile_stale_tasks()

    assert [t.task_id for t in resumable] == [task.task_id]
    # The owner rides the record, because a restart has no request to read it from and
    # resuming under the wrong tenant would index into the wrong corpus.
    assert resumable[0].owner == "alice"
    assert tasks.get_task(task.task_id).stage == "queued"


def test_an_orphan_that_has_exhausted_its_attempts_is_failed_not_resumed(monkeypatch):
    """Without a ceiling a poison upload becomes an infinite restart loop, which presents as
    an unstable deployment rather than a bad file."""
    monkeypatch.setenv("INGEST_STALE_AFTER_SECONDS", "60")
    monkeypatch.setenv("INGEST_MAX_ATTEMPTS", "3")
    get_settings.cache_clear()
    task = tasks.create_task(filename="poison.pdf", original_filename="poison.pdf")
    tasks.update_task(task.task_id, stage="parsing", message="Parsing...", attempts=3)
    tasks._tasks[task.task_id].updated_at = time.time() - 3600

    resumable = tasks.reconcile_stale_tasks()

    assert resumable == []
    recovered = tasks.get_task(task.task_id)
    assert recovered.stage == "failed"
    assert "attempts" in recovered.error.lower()


def test_a_slow_but_live_ingest_is_not_resumed(monkeypatch):
    """The threshold has to exceed the longest legitimate gap between stage updates, or a
    large corpus gets resumed out from under itself mid-index -- two ingests at once."""
    monkeypatch.setenv("INGEST_STALE_AFTER_SECONDS", "900")
    get_settings.cache_clear()
    task = tasks.create_task(filename="big.pdf", original_filename="big.pdf")
    tasks.update_task(task.task_id, stage="indexing", message="Embedding...")

    assert tasks.reconcile_stale_tasks() == []
    assert tasks.get_task(task.task_id).stage == "indexing"


def test_already_terminal_tasks_are_left_alone(monkeypatch):
    monkeypatch.setenv("INGEST_STALE_AFTER_SECONDS", "60")
    get_settings.cache_clear()
    task = tasks.create_task(filename="done.md", original_filename="done.md")
    tasks.update_task(task.task_id, stage="indexed", message="Done", indexed_chunks=4)
    tasks._tasks[task.task_id].updated_at = time.time() - 3600

    assert tasks.reconcile_stale_tasks() == []
    assert tasks.get_task(task.task_id).stage == "indexed"


def _run_lifespan() -> None:
    """Drives the app's lifespan start-to-finish.

    Directly rather than through `with TestClient(...)`: the lifespan installs a SIGTERM
    handler via `loop.add_signal_handler`, and TestClient runs it on a portal thread where
    that raises "set_wakeup_fd only works in main thread". `asyncio.run` here keeps it on the
    main thread, which is where uvicorn runs it in production anyway.
    """
    import asyncio

    async def _drive():
        async with api._lifespan(api.app):
            pass

    asyncio.run(_drive())


def test_reconciliation_runs_at_startup(monkeypatch):
    """Wired into the lifespan, not merely available to call."""
    called = []
    monkeypatch.setattr(api.ingest_tasks, "reconcile_stale_tasks", lambda: called.append(1) or [])

    _run_lifespan()

    assert called == [1]


def test_a_broken_task_backend_does_not_stop_startup(monkeypatch):
    """Reconciliation is housekeeping. Failing to start the service over it would turn a
    Redis blip into an outage."""

    def _explode():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(api.ingest_tasks, "reconcile_stale_tasks", _explode)

    _run_lifespan()  # must not raise
