import pytest

from rag_assistant.ingestion import tasks


@pytest.fixture(autouse=True)
def _clear_registry():
    # The registry is process-global module state (by design -- see tasks.py's docstring), so
    # tests must reset it themselves rather than relying on pytest isolation.
    tasks._tasks.clear()
    tasks._task_order.clear()
    yield
    tasks._tasks.clear()
    tasks._task_order.clear()


def test_create_task_starts_in_queued_stage():
    task = tasks.create_task(filename="doc_ab12cd34.md", original_filename="doc.md")

    assert task.stage == "queued"
    assert task.filename == "doc_ab12cd34.md"
    assert task.original_filename == "doc.md"
    assert task.error is None
    assert task.indexed_chunks is None


def test_update_task_only_overwrites_provided_fields():
    task = tasks.create_task(filename="doc_ab12cd34.md", original_filename="doc.md")

    tasks.update_task(task.task_id, stage="parsing", message="Loading...")
    fetched = tasks.get_task(task.task_id)
    assert fetched.stage == "parsing"
    assert fetched.message == "Loading..."

    tasks.update_task(task.task_id, indexed_chunks=5)
    fetched = tasks.get_task(task.task_id)
    # stage/message from the previous call must survive a call that only sets indexed_chunks.
    assert fetched.stage == "parsing"
    assert fetched.message == "Loading..."
    assert fetched.indexed_chunks == 5


def test_update_task_on_unknown_id_is_a_no_op():
    tasks.update_task("does-not-exist", stage="failed", error="boom")

    assert tasks.get_task("does-not-exist") is None


def test_get_task_returns_a_snapshot_not_the_live_object():
    task = tasks.create_task(filename="doc_ab12cd34.md", original_filename="doc.md")

    snapshot = tasks.get_task(task.task_id)
    tasks.update_task(task.task_id, stage="indexed", message="Done")

    assert snapshot.stage == "queued"
    assert tasks.get_task(task.task_id).stage == "indexed"


def test_registry_evicts_oldest_task_once_full(monkeypatch):
    monkeypatch.setattr(tasks, "_MAX_TASKS", 3)

    created = [tasks.create_task(filename=f"f{i}.md", original_filename=f"f{i}.md") for i in range(5)]

    assert tasks.get_task(created[0].task_id) is None
    assert tasks.get_task(created[1].task_id) is None
    assert tasks.get_task(created[4].task_id) is not None
