"""Section bodies for small-to-big retrieval.

Retrieval and synthesis want different things from a chunk. Retrieval wants it small: a tight,
single-topic passage embeds to a precise point and matches a specific question. Synthesis
wants it large: the surrounding section carries the qualifiers, the units, the "as of" date,
and the sentence that says which company the paragraph is about.

Chunking for one hurts the other. Small-to-big retrieval refuses the trade — it searches the
small chunks and then hands synthesis the section each winner came from.

The sections live in their own SQLite file beside the index rather than in chunk metadata,
because a section produces many chunks and duplicating the full section into each of them
would multiply stored corpus size by the chunks-per-section factor. They are keyed by the
`parent_id` the splitter stamps onto every chunk.
"""

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

PARENT_DB_FILENAME = "parents.db"

_LOCK = threading.Lock()
_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parents (
    parent_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    owner TEXT NOT NULL DEFAULT 'public',
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parents_source ON parents(source);
"""


def parent_db_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / PARENT_DB_FILENAME


def _get_conn(persist_dir: Path) -> sqlite3.Connection:
    """One connection per process, reopened when the persist directory changes (which is what
    happens between tests). Mirrors conversations/store.py's shape deliberately -- one
    concurrency story for every SQLite file in this project, not two."""
    global _conn, _conn_path
    path = parent_db_path(persist_dir)
    if _conn is None or _conn_path != path:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        _conn, _conn_path = conn, path
    return _conn


def reset_parent_store_cache() -> None:
    global _conn, _conn_path
    if _conn is not None:
        _conn.close()
    _conn, _conn_path = None, None


def replace_parents_for_source(
    persist_dir: Path, source: str, owner: str, parents: dict[str, str]
) -> None:
    """Replaces every section recorded for one source file.

    Delete-then-insert rather than upsert: a re-indexed file may produce *fewer* sections than
    before (a heading removed), and an upsert would leave the vanished sections behind as
    orphans that no chunk points at and nothing ever cleans up.
    """
    with _LOCK:
        conn = _get_conn(persist_dir)
        conn.execute("DELETE FROM parents WHERE source = ?", (source,))
        conn.executemany(
            "INSERT INTO parents (parent_id, source, owner, content) VALUES (?, ?, ?, ?)",
            [(parent_id, source, owner, content) for parent_id, content in parents.items()],
        )
        conn.commit()


def delete_parents_for_source(persist_dir: Path, source: str) -> None:
    with _LOCK:
        conn = _get_conn(persist_dir)
        conn.execute("DELETE FROM parents WHERE source = ?", (source,))
        conn.commit()


def get_parents(persist_dir: Path, parent_ids: list[str]) -> dict[str, str]:
    """Section bodies for the given ids. Missing ids are simply absent from the result --
    callers fall back to the chunk they already have, so a parent store that is stale or was
    never populated degrades to ordinary chunk retrieval rather than losing the answer."""
    if not parent_ids:
        return {}
    with _LOCK:
        conn = _get_conn(persist_dir)
        placeholders = ",".join("?" * len(parent_ids))
        rows = conn.execute(
            f"SELECT parent_id, content FROM parents WHERE parent_id IN ({placeholders})",
            parent_ids,
        ).fetchall()
    return {row[0]: row[1] for row in rows}


def count_parents(persist_dir: Path) -> int:
    with _LOCK:
        return _get_conn(persist_dir).execute("SELECT COUNT(*) FROM parents").fetchone()[0]
