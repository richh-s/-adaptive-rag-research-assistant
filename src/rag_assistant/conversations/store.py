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
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.config import get_settings

_LOCK = threading.Lock()
_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None

_SCHEMA = """
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
        conn.executescript(_SCHEMA)
        # Migration for databases created before multi-tenancy: pre-owner rows belong to the
        # "public" tenant, which is exactly who could see them back then. The owner index is
        # created here, NOT in _SCHEMA -- on an old database the column doesn't exist until
        # this ALTER runs, and an index in the schema script would reference it too early.
        columns = [row[1] for row in conn.execute("PRAGMA table_info(conversations)")]
        if "owner" not in columns:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN owner TEXT NOT NULL DEFAULT 'public'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner, updated_at)"
        )
        conn.commit()
        _conn, _conn_path = conn, path
    return _conn


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


def delete_conversation(conversation_id: str, owner: str = "public") -> bool:
    with _LOCK:
        conn = _get_conn()
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND owner = ?", (conversation_id, owner)
        )
        conn.commit()
    return cursor.rowcount > 0
