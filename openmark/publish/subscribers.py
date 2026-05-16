"""
Subscriber list — SQLite, double-opt-in, unsubscribe tokens.

Schema:
    subscribers(
        id, email, status, source, confirm_token, unsubscribe_token,
        created_at, confirmed_at, unsubscribed_at, last_sent_at, send_count
    )

Status state machine:
    pending    -> waiting on confirm email click
    active     -> opted in, eligible for sends
    bounced    -> Resend bounced; do not re-send
    unsubscribed -> clicked unsubscribe link; do not re-send

API (sync; SQLite is plenty for personal newsletter scale):
    init_subscribers_db()
    add_subscriber(email, source=...)            -> Subscriber  (status='pending')
    confirm_subscriber(token)                     -> Subscriber|None
    unsubscribe(token)                            -> Subscriber|None
    list_active(limit=10_000)                     -> list[Subscriber]
    mark_sent(subscriber_id)
    mark_bounced(email, reason)
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import time
from dataclasses import dataclass


DB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "subscribers.db")
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS subscribers (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    email              TEXT NOT NULL UNIQUE,
    status             TEXT NOT NULL DEFAULT 'pending',
    source             TEXT,
    confirm_token      TEXT NOT NULL UNIQUE,
    unsubscribe_token  TEXT NOT NULL UNIQUE,
    created_at         REAL NOT NULL,
    confirmed_at       REAL,
    unsubscribed_at    REAL,
    bounce_reason      TEXT,
    last_sent_at       REAL,
    send_count         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_subscribers_status ON subscribers(status);
CREATE INDEX IF NOT EXISTS idx_subscribers_email  ON subscribers(email);
"""


@dataclass
class Subscriber:
    id: int
    email: str
    status: str
    source: str | None
    confirm_token: str
    unsubscribe_token: str
    created_at: float
    confirmed_at: float | None
    unsubscribed_at: float | None
    bounce_reason: str | None
    last_sent_at: float | None
    send_count: int

    @classmethod
    def from_row(cls, r) -> "Subscriber":
        return cls(
            id=r[0], email=r[1], status=r[2], source=r[3],
            confirm_token=r[4], unsubscribe_token=r[5],
            created_at=r[6], confirmed_at=r[7], unsubscribed_at=r[8],
            bounce_reason=r[9], last_sent_at=r[10], send_count=r[11],
        )


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_subscribers_db() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def add_subscriber(email: str, source: str | None = None) -> Subscriber:
    """
    Insert a new subscriber in 'pending' state. If the email already exists
    in active state, return the existing row (idempotent). If existing in
    'unsubscribed' state, recycle the row back to 'pending' with new tokens.
    """
    email = email.strip().lower()
    if "@" not in email or "." not in email.split("@", 1)[1]:
        raise ValueError(f"invalid email: {email!r}")

    now = time.time()
    with _conn() as c:
        row = c.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone()
        if row:
            existing = Subscriber.from_row(row)
            if existing.status == "active":
                return existing
            # recycle pending/unsubscribed/bounced rows with fresh tokens
            new_confirm = _new_token()
            new_unsub = _new_token()
            c.execute(
                """
                UPDATE subscribers
                SET status='pending', source=?, confirm_token=?, unsubscribe_token=?,
                    created_at=?, confirmed_at=NULL, unsubscribed_at=NULL, bounce_reason=NULL
                WHERE email = ?
                """,
                (source, new_confirm, new_unsub, now, email),
            )
            row = c.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone()
            return Subscriber.from_row(row)

        confirm = _new_token()
        unsub = _new_token()
        c.execute(
            """
            INSERT INTO subscribers
                (email, status, source, confirm_token, unsubscribe_token, created_at)
            VALUES (?, 'pending', ?, ?, ?, ?)
            """,
            (email, source, confirm, unsub, now),
        )
        row = c.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone()
        return Subscriber.from_row(row)


def confirm_subscriber(token: str) -> Subscriber | None:
    """Click on confirm link → moves 'pending' to 'active'. Idempotent."""
    if not token:
        return None
    now = time.time()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM subscribers WHERE confirm_token = ?", (token,)
        ).fetchone()
        if not row:
            return None
        sub = Subscriber.from_row(row)
        if sub.status == "active":
            return sub
        c.execute(
            "UPDATE subscribers SET status='active', confirmed_at=? WHERE id=?",
            (now, sub.id),
        )
        return Subscriber.from_row(
            c.execute("SELECT * FROM subscribers WHERE id=?", (sub.id,)).fetchone()
        )


def unsubscribe(token: str) -> Subscriber | None:
    """Click on unsubscribe link → moves to 'unsubscribed'. Idempotent."""
    if not token:
        return None
    now = time.time()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM subscribers WHERE unsubscribe_token = ?", (token,)
        ).fetchone()
        if not row:
            return None
        c.execute(
            "UPDATE subscribers SET status='unsubscribed', unsubscribed_at=? WHERE id=?",
            (now, row[0]),
        )
        return Subscriber.from_row(
            c.execute("SELECT * FROM subscribers WHERE id=?", (row[0],)).fetchone()
        )


def mark_sent(subscriber_id: int) -> None:
    now = time.time()
    with _conn() as c:
        c.execute(
            """
            UPDATE subscribers
            SET last_sent_at = ?, send_count = send_count + 1
            WHERE id = ?
            """,
            (now, subscriber_id),
        )


def mark_bounced(email: str, reason: str = "") -> None:
    with _conn() as c:
        c.execute(
            "UPDATE subscribers SET status='bounced', bounce_reason=? WHERE email=?",
            (reason[:300], email.strip().lower()),
        )


def list_active(limit: int = 10_000) -> list[Subscriber]:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT * FROM subscribers
            WHERE status='active'
            ORDER BY confirmed_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [Subscriber.from_row(r) for r in rows]


def stats() -> dict:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT status, count(*) FROM subscribers GROUP BY status
            """
        ).fetchall()
    return {r[0]: r[1] for r in rows}
