"""SQLite subscriber list tests — no network, no LLM."""

from __future__ import annotations

import os
import uuid

import pytest


@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    """Each test gets its own temp DB file in a pytest-managed tempdir."""
    path = str(tmp_path / f"subs-{uuid.uuid4().hex[:8]}.db")
    import openmark.publish.subscribers as subs
    monkeypatch.setattr(subs, "DB_PATH", path)
    subs.init_subscribers_db()
    yield subs
    # pytest tmp_path is auto-cleaned; no manual unlink needed


def test_add_then_confirm_flow(isolated_db):
    s = isolated_db.add_subscriber("ahmad@example.com", source="form")
    assert s.status == "pending"
    assert s.confirm_token and s.unsubscribe_token
    confirmed = isolated_db.confirm_subscriber(s.confirm_token)
    assert confirmed is not None
    assert confirmed.status == "active"
    assert confirmed.email == "ahmad@example.com"


def test_double_subscribe_idempotent_when_active(isolated_db):
    s1 = isolated_db.add_subscriber("user@example.com")
    isolated_db.confirm_subscriber(s1.confirm_token)
    s2 = isolated_db.add_subscriber("user@example.com")
    # Active subscribers stay as-is on re-subscribe; no re-issued tokens
    assert s2.status == "active"
    assert s2.email == "user@example.com"


def test_resubscribe_recycles_after_unsubscribe(isolated_db):
    s1 = isolated_db.add_subscriber("recycle@example.com")
    isolated_db.confirm_subscriber(s1.confirm_token)
    isolated_db.unsubscribe(s1.unsubscribe_token)
    s2 = isolated_db.add_subscriber("recycle@example.com")
    # Recycled row -> back to pending with FRESH tokens
    assert s2.status == "pending"
    assert s2.confirm_token != s1.confirm_token


def test_unsubscribe_idempotent(isolated_db):
    s = isolated_db.add_subscriber("unsub@example.com")
    isolated_db.confirm_subscriber(s.confirm_token)
    r1 = isolated_db.unsubscribe(s.unsubscribe_token)
    r2 = isolated_db.unsubscribe(s.unsubscribe_token)
    assert r1.status == r2.status == "unsubscribed"


def test_invalid_token_returns_none(isolated_db):
    assert isolated_db.confirm_subscriber("not-a-real-token") is None
    assert isolated_db.unsubscribe("not-a-real-token") is None
    assert isolated_db.confirm_subscriber("") is None


def test_email_validation(isolated_db):
    with pytest.raises(ValueError):
        isolated_db.add_subscriber("not-an-email")
    with pytest.raises(ValueError):
        isolated_db.add_subscriber("missing@dot")


def test_list_active_returns_only_active(isolated_db):
    s1 = isolated_db.add_subscriber("a@example.com")
    isolated_db.confirm_subscriber(s1.confirm_token)
    s2 = isolated_db.add_subscriber("b@example.com")  # pending, not confirmed
    s3 = isolated_db.add_subscriber("c@example.com")
    isolated_db.confirm_subscriber(s3.confirm_token)
    isolated_db.unsubscribe(s3.unsubscribe_token)
    active = isolated_db.list_active()
    emails = {s.email for s in active}
    assert emails == {"a@example.com"}


def test_mark_sent_increments(isolated_db):
    s = isolated_db.add_subscriber("track@example.com")
    isolated_db.confirm_subscriber(s.confirm_token)
    sid = s.id
    isolated_db.mark_sent(sid)
    isolated_db.mark_sent(sid)
    after = isolated_db.list_active()
    assert len(after) == 1
    assert after[0].send_count == 2
    assert after[0].last_sent_at is not None


def test_stats_groups_by_status(isolated_db):
    s1 = isolated_db.add_subscriber("a@example.com")
    isolated_db.confirm_subscriber(s1.confirm_token)
    isolated_db.add_subscriber("b@example.com")  # pending
    s3 = isolated_db.add_subscriber("c@example.com")
    isolated_db.confirm_subscriber(s3.confirm_token)
    isolated_db.unsubscribe(s3.unsubscribe_token)
    s = isolated_db.stats()
    assert s.get("active") == 1
    assert s.get("pending") == 1
    assert s.get("unsubscribed") == 1
