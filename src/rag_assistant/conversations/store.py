"""SQLite-backed conversation persistence.

SQLite (not Redis/Postgres) is deliberate: this project already commits to single-process
SQLite-backed storage for Chroma (see docker-compose.yml's volume comment), so conversations
follow the same operational model -- durable across restarts via the same volume, zero extra
infrastructure, and honest about the single-writer constraint. WAL mode + a process-wide lock
serialize writers across FastAPI's threadpool; a multi-replica deployment would move this to
Postgres, same as it would move Chroma to server mode.

The server owns conversation history: clients send a `conversation_id` and the API loads the
transcript from here, rather than trusting the client to replay it. (The stateless `history`
field on /research still works for callers that opt out of persistence.)
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.config import get_settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None


def _migration_001_baseline(conn: sqlite3.Connection) -> None:
    """Everything the schema was before migrations existed, made idempotent.

    This has to converge three different databases onto one shape, because versioning was
    introduced after the fact and all of them report `user_version = 0`: a brand-new file, a
    pre-multi-tenancy file whose `conversations` table has no `owner` column, and a file
    already carrying `owner` from the ad-hoc ALTER that used to run on every connect. Hence
    `IF NOT EXISTS` throughout plus the guarded ALTER -- after this runs, all three are
    identical and stamped as version 1.

    The owner index is created *after* the ALTER, not alongside the table: on a pre-owner
    database the column doesn't exist until that statement lands, and an index referencing it
    any earlier fails.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            owner TEXT NOT NULL DEFAULT 'public',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            report TEXT,
            summary_json TEXT,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
        """
    )
    columns = [row[1] for row in conn.execute("PRAGMA table_info(conversations)")]
    if "owner" not in columns:
        conn.execute("ALTER TABLE conversations ADD COLUMN owner TEXT NOT NULL DEFAULT 'public'")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner, updated_at)"
    )


# Ordered, append-only. Index i in this list is migration number i+1, and `PRAGMA user_version`
# records how many have been applied -- the standard SQLite idiom, and the reason a schema
# change from here on is "append a function" rather than "hope CREATE TABLE IF NOT EXISTS
# happens to cover it". Never edit or reorder an entry that has shipped; append a new one.
_MIGRATIONS: list = [
    _migration_001_baseline,
]

MAX_TITLE_CHARS = 80
# Matches the ResearchRequest.history cap: the graph never needs more than the recent window.
HISTORY_TURN_LIMIT = 20


@dataclass
class ConversationRow:
    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


@dataclass
class MessageRow:
    role: str
    content: str
    report: str | None
    summary: dict | None
    created_at: float


def _get_conn() -> sqlite3.Connection:
    """One connection per process, shared across threads (serialized by _LOCK). Callers must
    hold _LOCK for the read-modify-write sequences; simple single-statement reads ride on
    SQLite's own locking plus WAL."""
    global _conn, _conn_path
    path = get_settings().conversations_db_path
    if _conn is None or _conn_path != path:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _migrate(conn)
        _conn, _conn_path = conn, path
    return _conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Applies every migration the database hasn't seen, each in its own transaction.

    Per-migration commits rather than one big one: if migration N+1 fails, N stays applied and
    recorded, so a retry resumes instead of replaying work that already succeeded. The version
    stamp is written inside the same transaction as the migration it describes, so the two
    can't disagree.

    `user_version` takes a literal -- SQLite doesn't accept a bound parameter in a PRAGMA --
    but the value is a list index here, never anything caller-supplied.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= len(_MIGRATIONS):
        return
    for version in range(current, len(_MIGRATIONS)):
        migration = _MIGRATIONS[version]
        logger.info(
            "applying conversation store migration %d (%s)",
            version + 1,
            migration.__name__,
        )
        migration(conn)
        conn.execute(f"PRAGMA user_version = {version + 1}")
        conn.commit()


def reset_store_cache() -> None:
    """Tests point CONVERSATIONS_DB_PATH at tmp dirs per-test; this drops the cached
    connection so the next call reopens against the current settings (mirrors
    cache.reset_client_cache)."""
    global _conn, _conn_path
    if _conn is not None:
        _conn.close()
    _conn, _conn_path = None, None


def create_conversation(title: str, owner: str = "public") -> ConversationRow:
    now = time.time()
    conversation_id = uuid.uuid4().hex
    clean_title = (title or "New conversation").strip()[:MAX_TITLE_CHARS] or "New conversation"
    with _LOCK:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO conversations (id, title, owner, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, clean_title, owner, now, now),
        )
        conn.commit()
    return ConversationRow(
        id=conversation_id, title=clean_title, created_at=now, updated_at=now, message_count=0
    )


def list_conversations(owner: str = "public", limit: int = 100) -> list[ConversationRow]:
    with _LOCK:
        rows = _get_conn().execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id)
            FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.owner = ?
            GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?
            """,
            (owner, limit),
        ).fetchall()
    return [
        ConversationRow(id=r[0], title=r[1], created_at=r[2], updated_at=r[3], message_count=r[4])
        for r in rows
    ]


def get_conversation(conversation_id: str, owner: str = "public") -> ConversationRow | None:
    """Owner-scoped on purpose: a valid id belonging to another tenant returns None, which
    the API surfaces as the same 404 as a nonexistent id -- no cross-tenant existence oracle."""
    with _LOCK:
        row = _get_conn().execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id)
            FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.id = ? AND c.owner = ? GROUP BY c.id
            """,
            (conversation_id, owner),
        ).fetchone()
    if row is None:
        return None
    return ConversationRow(
        id=row[0], title=row[1], created_at=row[2], updated_at=row[3], message_count=row[4]
    )


def get_messages(conversation_id: str) -> list[MessageRow]:
    with _LOCK:
        rows = _get_conn().execute(
            """
            SELECT role, content, report, summary_json, created_at
            FROM messages WHERE conversation_id = ? ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
    return [
        MessageRow(
            role=r[0],
            content=r[1],
            report=r[2],
            summary=json.loads(r[3]) if r[3] else None,
            created_at=r[4],
        )
        for r in rows
    ]


def get_history(conversation_id: str) -> list[dict]:
    """The transcript in the {"role", "content"} shape the graph's chat_history expects,
    capped to the most recent HISTORY_TURN_LIMIT messages."""
    messages = get_messages(conversation_id)
    return [{"role": m.role, "content": m.content} for m in messages[-HISTORY_TURN_LIMIT:]]


def append_turn(
    conversation_id: str,
    question: str,
    answer: str,
    report: str | None = None,
    summary: dict | None = None,
) -> None:
    """Persists one completed exchange (user question + assistant answer) atomically, so a
    crash between the two writes can't leave a dangling user message that would skew the
    next request's condensation history."""
    now = time.time()
    with _LOCK:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
            (conversation_id, question, now),
        )
        conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, report, summary_json, created_at)
            VALUES (?, 'assistant', ?, ?, ?, ?)
            """,
            (conversation_id, answer, report, json.dumps(summary) if summary else None, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
        )
        conn.commit()
        # Pruned here rather than on a timer: the only moment the table can grow is a write,
        # and this scopes the work to the one tenant that just wrote instead of sweeping the
        # whole database. Failures are logged, not raised -- the turn is already durably
        # committed above, and losing a retention pass beats losing the user's answer.
        owner_row = conn.execute(
            "SELECT owner FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if owner_row is not None:
            try:
                _prune_locked(conn, owner_row[0], now=now)
            except Exception:
                logger.warning("conversation retention pass failed", exc_info=True)


def _prune_locked(conn: sqlite3.Connection, owner: str, now: float | None = None) -> int:
    """Enforces the retention policy for one tenant. Caller must hold `_LOCK`.

    Two independent ceilings, either disabled by setting it to 0: an age cutoff
    (CONVERSATION_RETENTION_DAYS) and a per-tenant count cap (CONVERSATION_MAX_PER_OWNER,
    keeping the most recently updated). Messages go with their conversation via the schema's
    ON DELETE CASCADE, which is live because `_get_conn` turns `foreign_keys` on.

    Both statements are driven by `idx_conversations_owner (owner, updated_at)`, so this stays
    an indexed range delete rather than a scan that gets slower as other tenants accumulate.
    """
    settings = get_settings()
    now = time.time() if now is None else now
    removed = 0

    if settings.conversation_retention_days > 0:
        cutoff = now - settings.conversation_retention_days * 86400
        cursor = conn.execute(
            "DELETE FROM conversations WHERE owner = ? AND updated_at < ?", (owner, cutoff)
        )
        removed += cursor.rowcount or 0

    if settings.conversation_max_per_owner > 0:
        cursor = conn.execute(
            """
            DELETE FROM conversations WHERE id IN (
                SELECT id FROM conversations WHERE owner = ?
                ORDER BY updated_at DESC LIMIT -1 OFFSET ?
            )
            """,
            (owner, settings.conversation_max_per_owner),
        )
        removed += cursor.rowcount or 0

    if removed:
        conn.commit()
        logger.info(
            "pruned %d conversation(s) past the retention policy", removed, extra={"owner": owner}
        )
    return removed


def prune_conversations(owner: str = "public") -> int:
    """Public entry point for the retention policy -- used by tests and available for an
    operator-triggered sweep. `append_turn` prunes inline and does not go through here."""
    with _LOCK:
        return _prune_locked(_get_conn(), owner)


def delete_conversation(conversation_id: str, owner: str = "public") -> bool:
    with _LOCK:
        conn = _get_conn()
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND owner = ?", (conversation_id, owner)
        )
        conn.commit()
    return cursor.rowcount > 0
