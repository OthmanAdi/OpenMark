"""
SQLite chat history for the Gradio chat tab.

Stores every chat session and every message exactly as the user saw it,
so a page refresh / browser close / restart never loses what was on
screen. The agent's runtime memory (MemorySaver) is separate — this is
for replay and review.

Schema:
    sessions(id, title, created_at, last_message_at)
    messages(id, session_id, role, content, thinking, tool_calls_json, ts)

API (all sync; SQLite is fast enough for chat-scale traffic):
    init_db()                                    one-shot at app start
    create_session(title='New chat') -> int      returns session id
    update_session_title(sid, title)
    append_message(sid, role, content, thinking='', tool_calls=None)
    list_sessions(limit=50) -> [dict]            newest first
    get_messages(sid) -> [dict]                  chronological
    delete_session(sid)
    auto_title(first_user_message) -> str        derive a short title

The DB file lives at data/openmark_chat.db so it's outside the repo
(data/ is already gitignored) and survives `pip install`-like resets.
"""

from __future__ import annotations

import os
import json
import time
import sqlite3
from typing import Optional


DB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "openmark_chat.db")
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL DEFAULT 'New chat',
    created_at      REAL NOT NULL,
    last_message_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL,
    role            TEXT NOT NULL,   -- 'user' or 'assistant'
    content         TEXT NOT NULL,   -- markdown exactly as the user saw it
    thinking        TEXT,            -- concatenated turn_thinking text (assistant only)
    tool_calls_json TEXT,            -- JSON list of tool events
    ts              REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_recent  ON sessions(last_message_at DESC);
"""


def _conn() -> sqlite3.Connection:
    """One short-lived connection per call — sqlite handles concurrency on the file."""
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(title: str = "New chat") -> int:
    now = time.time()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO sessions (title, created_at, last_message_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        return int(cur.lastrowid)


def update_session_title(session_id: int, title: str) -> None:
    with _conn() as c:
        c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))


def list_sessions(limit: int = 100) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT s.id, s.title, s.created_at, s.last_message_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n_msgs
            FROM sessions s
            ORDER BY s.last_message_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "title": r[1] or "New chat",
            "created_at": r[2],
            "last_message_at": r[3],
            "n_msgs": r[4],
        }
        for r in rows
    ]


def delete_session(session_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# ── Messages ──────────────────────────────────────────────────────────────────

def append_message(
    session_id: int,
    role: str,
    content: str,
    thinking: str = "",
    tool_calls: list | None = None,
) -> int:
    now = time.time()
    tc_json = json.dumps(tool_calls or [], ensure_ascii=False)
    with _conn() as c:
        cur = c.execute(
            """
            INSERT INTO messages (session_id, role, content, thinking, tool_calls_json, ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, content, thinking, tc_json, now),
        )
        c.execute(
            "UPDATE sessions SET last_message_at = ? WHERE id = ?", (now, session_id)
        )
        return int(cur.lastrowid)


def get_messages(session_id: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT role, content, thinking, tool_calls_json, ts
            FROM messages WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    out = []
    for r in rows:
        tc = []
        if r[3]:
            try:
                tc = json.loads(r[3])
            except Exception:
                tc = []
        out.append({"role": r[0], "content": r[1], "thinking": r[2] or "",
                    "tool_calls": tc, "ts": r[4]})
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def auto_title(text: str, n: int = 60) -> str:
    """Derive a short session title from the first user message."""
    text = (text or "").strip().replace("\n", " ")
    # Strip a leading slash command — keep the actual question
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        text = parts[1] if len(parts) > 1 else (parts[0].lstrip("/") if parts else "")
    if not text:
        return "New chat"
    if len(text) > n:
        text = text[: n - 1].rstrip() + "…"
    return text


def session_label(s: dict) -> str:
    """Render a session row as a dropdown label."""
    import datetime as _dt
    ts = _dt.datetime.fromtimestamp(s["last_message_at"]).strftime("%m/%d %H:%M")
    return f"#{s['id']} · {s['title']} · {s['n_msgs']} msgs · {ts}"
