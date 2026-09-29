"""Deterministic Postgres advisory-lock ids.

Both migration chains (conversations and the pgvector index) serialise themselves with
`pg_advisory_lock` so several replicas can start at once without racing onto the same
migration. That only works if every replica computes the *same* id from the same name.

Python's `hash()` does not do that. `str.__hash__` is salted per interpreter process
(PYTHONHASHSEED, on by default since 3.3), so `hash("migrations")` returns a different
number in every process -- which is precisely the situation the lock exists for. Each
replica would take a lock nobody else contends on, and the mutual exclusion would be
decorative. `CREATE TABLE IF NOT EXISTS` hides that for the migrations written so far; the
first one that ALTERs or INSERTs is where it surfaces, as a startup crash on whichever
replica loses a race that was never supposed to happen.

CRC32 is used instead, for the one property that matters here: it is a pure function of the
bytes, fixed by the algorithm rather than by a runtime seed, so it is stable across
processes, restarts and Python versions. It is not a secure hash and does not need to be --
nothing here is guarding against an adversary choosing a colliding lock name.

The result is masked to 31 bits so it fits Postgres's signed `bigint`/`int` advisory-lock
parameter as a positive number, keeping the ids readable in `pg_locks`.
"""

import zlib

LOCK_ID_SPACE = 2**31


def advisory_lock_id(name: str) -> int:
    """A stable, positive advisory-lock id for `name`, identical in every process."""
    return zlib.crc32(name.encode("utf-8")) % LOCK_ID_SPACE
