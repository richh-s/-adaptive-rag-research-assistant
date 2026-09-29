"""The migration advisory-lock ids must be identical in every process.

This is the test that the original code could not have passed. It computed its lock id with
`hash("rag_assistant_migrations") % 2**31`, and `str.__hash__` is salted per interpreter, so
every replica took a *different* advisory lock and the mutual exclusion the comment promised
did not exist. Nothing caught it because a single-process test compares a value to itself.

So the assertion has to cross a process boundary, with the hash seed explicitly varied --
that is the only way the property under test ("two replicas agree") is actually exercised.
"""

import subprocess
import sys

import pytest

from rag_assistant.advisory_lock import LOCK_ID_SPACE, advisory_lock_id

# The two independent migration chains. Named here rather than imported from the stores so
# this file has no psycopg dependency -- and so a renamed lock constant shows up as a
# failure here rather than as a silently un-run test.
LOCK_NAMES = ["rag_assistant_migrations", "rag_assistant_pgvector_migrations"]


def _lock_ids_in_subprocess(seed: str) -> list[int]:
    """Computes the ids in a fresh interpreter with PYTHONHASHSEED pinned to `seed`."""
    program = (
        "from rag_assistant.advisory_lock import advisory_lock_id;"
        f"print(' '.join(str(advisory_lock_id(n)) for n in {LOCK_NAMES!r}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
    )
    return [int(value) for value in completed.stdout.split()]


def test_lock_ids_are_identical_across_processes_with_different_hash_seeds():
    """Two replicas starting at once must contend on the same lock, or it locks nothing."""
    first = _lock_ids_in_subprocess("1")
    second = _lock_ids_in_subprocess("2")

    assert first == second, (
        "advisory lock ids differ between processes -- every replica would take its own "
        "lock and the migration mutual exclusion would be decorative"
    )


def test_the_builtin_hash_really_is_unstable_across_processes():
    """Guards the reason this module exists.

    If a future Python made `hash()` stable, `advisory_lock_id` would look like ceremony and
    somebody would inline `hash()` back. This fails first, and says why.
    """
    program = "print(hash('rag_assistant_migrations') % 2**31)"
    env_a = {"PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin"}
    env_b = {"PYTHONHASHSEED": "2", "PATH": "/usr/bin:/bin"}
    a = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True, env=env_a
    )
    b = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True, env=env_b
    )

    assert a.stdout != b.stdout, (
        "hash() is stable here; advisory_lock_id's rationale needs revisiting"
    )


def test_the_two_migration_chains_do_not_share_a_lock():
    """Sharing one id would serialise unrelated startups and make a stall in one chain look
    like a hang in the other."""
    ids = {name: advisory_lock_id(name) for name in LOCK_NAMES}

    assert len(set(ids.values())) == len(ids), f"lock id collision: {ids}"


@pytest.mark.parametrize("name", LOCK_NAMES)
def test_lock_ids_fit_postgres_advisory_lock_parameter(name):
    """Positive and inside int32: `pg_advisory_lock` takes a signed value, and a negative or
    oversized id is a runtime error on a path that only runs at startup under concurrency."""
    lock_id = advisory_lock_id(name)

    assert 0 <= lock_id < LOCK_ID_SPACE


def test_the_stores_use_the_shared_helper():
    """The ids the stores actually pass to Postgres, not just the helper in isolation."""
    psycopg = pytest.importorskip("psycopg")  # noqa: F841
    from rag_assistant.conversations import postgres_store
    from rag_assistant.retrieval import pgvector_store

    assert postgres_store._MIGRATION_LOCK_ID == advisory_lock_id("rag_assistant_migrations")
    assert pgvector_store._MIGRATION_LOCK_ID == advisory_lock_id(
        "rag_assistant_pgvector_migrations"
    )
