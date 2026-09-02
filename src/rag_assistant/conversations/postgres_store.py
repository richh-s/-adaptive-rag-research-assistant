"""Postgres-backed conversation persistence, for running more than one replica.

SQLite is the default and the right one for a single container: durable across restarts on
the same volume, zero infrastructure, honest about its single-writer constraint. What it
cannot do is let two processes write, which makes it the thing standing between this service
and a second replica.

This is the same schema, the same public functions, and the same semantics, against Postgres.
`CONVERSATIONS_BACKEND=postgres` plus `DATABASE_URL` switches to it; nothing that calls
`conversations.store` knows which is running.

The migration mechanism is the same idea as the SQLite side but not the same implementation:
`PRAGMA user_version` doesn't exist here, so applied migrations are recorded in a table. The
two version counters are independent — a database is only ever one or the other.
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass

from rag_assistant.config import get_settings

logger = logging.getLogger(__name__)

MAX_TITLE_CHARS = 80
HISTORY_TURN_LIMIT = 20

_LOCK = threading.Lock()
_pool = None


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


@dataclass
class FeedbackRow:
    id: int
    conversation_id: str | None
    question: str
    rating: str
    note: str | None
    route: str | None
    confidence_score: float | None
    created_at: float


def _migration_001_baseline(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            owner TEXT NOT NULL DEFAULT 'public',
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id BIGSERIAL PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            report TEXT,
            summary_json TEXT,
            created_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner, updated_at)"
    )


def _migration_002_feedback(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id BIGSERIAL PRIMARY KEY,
            conversation_id TEXT,
            owner TEXT NOT NULL DEFAULT 'public',
            question TEXT NOT NULL,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            note TEXT,
            route TEXT,
            confidence_score DOUBLE PRECISION,
            created_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_owner ON feedback(owner, created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")


_MIGRATIONS: list = [_migration_001_baseline, _migration_002_feedback]


def _get_pool():
    """One connection pool per process.

    A pool rather than a single shared connection, because the point of this backend is
    concurrency: FastAPI's threadpool runs handlers in parallel and Postgres connections are
    not safe to share across threads.
    """
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool

        settings = get_settings()
        if not settings.database_url:
            raise RuntimeError("CONVERSATIONS_BACKEND=postgres requires DATABASE_URL to be set.")
        _pool = ConnectionPool(settings.database_url, min_size=1, open=True)
        _migrate()
    return _pool


def reset_store_cache() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
    _pool = None


def _migrate() -> None:
    """Applies pending migrations, each in its own transaction.

    The advisory lock is what makes this safe with several replicas starting at once: without
    it, two processes both read version 0 and both run migration 1, and whichever loses the
    race fails on an object that already exists. `CREATE TABLE IF NOT EXISTS` hides most of
    that, but not a future migration that inserts or alters.
    """
    with _pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (hash("rag_assistant_migrations") % 2**31,))
            try:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations "
                    "(version INTEGER PRIMARY KEY, applied_at DOUBLE PRECISION NOT NULL)"
                )
                conn.commit()
                cur.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
                current = cur.fetchone()[0]
                for version in range(current, len(_MIGRATIONS)):
                    migration = _MIGRATIONS[version]
                    logger.info(
                        "applying postgres conversation migration %d (%s)",
                        version + 1,
                        migration.__name__,
                    )
                    migration(cur)
                    cur.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (%s, %s)",
                        (version + 1, time.time()),
                    )
                    conn.commit()
            finally:
                cur.execute(
                    "SELECT pg_advisory_unlock(%s)", (hash("rag_assistant_migrations") % 2**31,)
                )
                conn.commit()


# ---- conversations ----


def create_conversation(title: str, owner: str = "public") -> ConversationRow:
    now = time.time()
    conversation_id = uuid.uuid4().hex
    clean_title = (title or "New conversation").strip()[:MAX_TITLE_CHARS] or "New conversation"
    with _get_pool().connection() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, owner, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (conversation_id, clean_title, owner, now, now),
        )
    return ConversationRow(
        id=conversation_id, title=clean_title, created_at=now, updated_at=now, message_count=0
    )


def list_conversations(owner: str = "public", limit: int = 100) -> list[ConversationRow]:
    with _get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id)
            FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.owner = %s
            GROUP BY c.id ORDER BY c.updated_at DESC LIMIT %s
            """,
            (owner, limit),
        ).fetchall()
    return [
        ConversationRow(id=r[0], title=r[1], created_at=r[2], updated_at=r[3], message_count=r[4])
        for r in rows
    ]


def get_conversation(conversation_id: str, owner: str = "public") -> ConversationRow | None:
    """Owner-scoped: a valid id belonging to another tenant returns None, which the API
    surfaces as the same 404 as a nonexistent id -- no cross-tenant existence oracle."""
    with _get_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id)
            FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.id = %s AND c.owner = %s GROUP BY c.id
            """,
            (conversation_id, owner),
        ).fetchone()
    if row is None:
        return None
    return ConversationRow(
        id=row[0], title=row[1], created_at=row[2], updated_at=row[3], message_count=row[4]
    )


def get_messages(conversation_id: str) -> list[MessageRow]:
    with _get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT role, content, report, summary_json, created_at FROM messages "
            "WHERE conversation_id = %s ORDER BY id",
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
    messages = get_messages(conversation_id)
    return [{"role": m.role, "content": m.content} for m in messages[-HISTORY_TURN_LIMIT:]]


def append_turn(
    conversation_id: str,
    question: str,
    answer: str,
    report: str | None = None,
    summary: dict | None = None,
) -> None:
    """Persists one completed exchange atomically -- a crash between the two writes can't
    leave a dangling user message that would skew the next request's condensation history."""
    now = time.time()
    with _get_pool().connection() as conn:
        with conn.transaction():
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (%s, 'user', %s, %s)",
                (conversation_id, question, now),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, report, summary_json, created_at) "
                "VALUES (%s, 'assistant', %s, %s, %s, %s)",
                (conversation_id, answer, report, json.dumps(summary) if summary else None, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = %s WHERE id = %s", (now, conversation_id)
            )
        owner_row = conn.execute(
            "SELECT owner FROM conversations WHERE id = %s", (conversation_id,)
        ).fetchone()
    if owner_row is not None:
        try:
            prune_conversations(owner_row[0])
        except Exception:
            logger.warning("conversation retention pass failed", exc_info=True)


def delete_conversation(conversation_id: str, owner: str = "public") -> bool:
    with _get_pool().connection() as conn:
        result = conn.execute(
            "DELETE FROM conversations WHERE id = %s AND owner = %s", (conversation_id, owner)
        )
        return result.rowcount > 0


def prune_conversations(owner: str = "public") -> int:
    """Same two ceilings as the SQLite backend: an age cutoff and a per-tenant count cap."""
    settings = get_settings()
    now = time.time()
    removed = 0
    with _get_pool().connection() as conn:
        if settings.conversation_retention_days > 0:
            cutoff = now - settings.conversation_retention_days * 86400
            result = conn.execute(
                "DELETE FROM conversations WHERE owner = %s AND updated_at < %s", (owner, cutoff)
            )
            removed += result.rowcount or 0
        if settings.conversation_max_per_owner > 0:
            result = conn.execute(
                """
                DELETE FROM conversations WHERE id IN (
                    SELECT id FROM conversations WHERE owner = %s
                    ORDER BY updated_at DESC OFFSET %s
                )
                """,
                (owner, settings.conversation_max_per_owner),
            )
            removed += result.rowcount or 0
    return removed


# ---- feedback ----


def record_feedback(
    question: str,
    rating: str,
    owner: str = "public",
    conversation_id: str | None = None,
    note: str | None = None,
    route: str | None = None,
    confidence_score: float | None = None,
) -> int:
    now = time.time()
    with _get_pool().connection() as conn:
        row = conn.execute(
            """
            INSERT INTO feedback
                (conversation_id, owner, question, rating, note, route, confidence_score, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (conversation_id, owner, question, rating, note, route, confidence_score, now),
        ).fetchone()
    return int(row[0])


def list_feedback(owner: str = "public", limit: int = 100) -> list[FeedbackRow]:
    with _get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, conversation_id, question, rating, note, route, confidence_score, created_at "
            "FROM feedback WHERE owner = %s ORDER BY created_at DESC LIMIT %s",
            (owner, limit),
        ).fetchall()
    return [FeedbackRow(*row) for row in rows]


def feedback_summary(owner: str = "public") -> dict:
    with _get_pool().connection() as conn:
        counts = dict(
            conn.execute(
                "SELECT rating, COUNT(*) FROM feedback WHERE owner = %s GROUP BY rating", (owner,)
            ).fetchall()
        )
        downvoted = [
            row[0]
            for row in conn.execute(
                "SELECT question FROM feedback WHERE owner = %s AND rating = 'down' "
                "ORDER BY created_at DESC LIMIT 20",
                (owner,),
            ).fetchall()
        ]
    up, down = counts.get("up", 0), counts.get("down", 0)
    total = up + down
    return {
        "up": up,
        "down": down,
        "total": total,
        "satisfaction": (up / total) if total else None,
        "recent_downvoted_questions": downvoted,
    }
