-- Cloudflare D1 schema for the subscriber list.
-- Mirror of openmark/publish/subscribers.py — keep the two in sync.
--
-- Apply with:
--   wrangler d1 execute openmark-subscribers --file=migrations/001_subscribers.sql
--

CREATE TABLE IF NOT EXISTS subscribers (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    email              TEXT NOT NULL UNIQUE,
    status             TEXT NOT NULL DEFAULT 'pending',   -- pending | active | unsubscribed | bounced
    source             TEXT,                              -- 'site', 'manual', 'import', ...
    confirm_token      TEXT NOT NULL UNIQUE,
    unsubscribe_token  TEXT NOT NULL UNIQUE,
    created_at         REAL NOT NULL,                     -- unix epoch seconds
    confirmed_at       REAL,
    unsubscribed_at    REAL,
    bounce_reason      TEXT,
    last_sent_at       REAL,
    send_count         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_subscribers_status ON subscribers(status);
CREATE INDEX IF NOT EXISTS idx_subscribers_email  ON subscribers(email);
CREATE INDEX IF NOT EXISTS idx_subscribers_confirm_token ON subscribers(confirm_token);
CREATE INDEX IF NOT EXISTS idx_subscribers_unsubscribe_token ON subscribers(unsubscribe_token);
