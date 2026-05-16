/**
 * D1 query helpers shared by every /api/* function. Keeps SQL in one place.
 *
 * Env binding: D1Database `DB` (declared in wrangler.toml [[d1_databases]]).
 */

export interface Subscriber {
  id: number;
  email: string;
  status: 'pending' | 'active' | 'unsubscribed' | 'bounced';
  source: string | null;
  confirm_token: string;
  unsubscribe_token: string;
  created_at: number;
  confirmed_at: number | null;
  unsubscribed_at: number | null;
  bounce_reason: string | null;
  last_sent_at: number | null;
  send_count: number;
}

export interface Env {
  DB: D1Database;
  RESEND_API_KEY?: string;
  PUBLISH_FROM_EMAIL?: string;
  PUBLISH_BASE_URL?: string;
  PUBLICATION_NAME?: string;
}

/**
 * 256-bit random URL-safe token. Web Crypto is available in the Workers runtime.
 */
export function newToken(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

export function isValidEmail(s: string): boolean {
  if (!s || typeof s !== 'string') return false;
  const trimmed = s.trim().toLowerCase();
  // Conservative but practical: at least one @ and one . in the domain part
  if (!trimmed.includes('@')) return false;
  const [_, domain] = trimmed.split('@', 2);
  return domain.includes('.') && domain.length > 2;
}

export function normalizeEmail(s: string): string {
  return s.trim().toLowerCase();
}

/** Find subscriber by email (case-insensitive). */
export async function findByEmail(env: Env, email: string): Promise<Subscriber | null> {
  const row = await env.DB
    .prepare('SELECT * FROM subscribers WHERE email = ?')
    .bind(normalizeEmail(email))
    .first<Subscriber>();
  return row ?? null;
}

/** Insert or recycle a subscriber in pending state. Returns the row. */
export async function addOrRecycle(
  env: Env,
  email: string,
  source: string | null,
): Promise<Subscriber> {
  email = normalizeEmail(email);
  const existing = await findByEmail(env, email);
  const now = Date.now() / 1000;

  if (existing) {
    if (existing.status === 'active') {
      // Already subscribed; idempotent — return as-is.
      return existing;
    }
    // Recycle pending/unsubscribed/bounced with FRESH tokens.
    const confirm = newToken();
    const unsubscribe = newToken();
    await env.DB
      .prepare(
        `UPDATE subscribers
           SET status = 'pending', source = ?,
               confirm_token = ?, unsubscribe_token = ?,
               created_at = ?, confirmed_at = NULL,
               unsubscribed_at = NULL, bounce_reason = NULL
         WHERE email = ?`,
      )
      .bind(source, confirm, unsubscribe, now, email)
      .run();
    return (await findByEmail(env, email))!;
  }

  const confirm = newToken();
  const unsubscribe = newToken();
  await env.DB
    .prepare(
      `INSERT INTO subscribers
         (email, status, source, confirm_token, unsubscribe_token, created_at)
       VALUES (?, 'pending', ?, ?, ?, ?)`,
    )
    .bind(email, source, confirm, unsubscribe, now)
    .run();
  return (await findByEmail(env, email))!;
}

export async function findByConfirmToken(env: Env, token: string): Promise<Subscriber | null> {
  if (!token) return null;
  const row = await env.DB
    .prepare('SELECT * FROM subscribers WHERE confirm_token = ?')
    .bind(token)
    .first<Subscriber>();
  return row ?? null;
}

export async function findByUnsubscribeToken(env: Env, token: string): Promise<Subscriber | null> {
  if (!token) return null;
  const row = await env.DB
    .prepare('SELECT * FROM subscribers WHERE unsubscribe_token = ?')
    .bind(token)
    .first<Subscriber>();
  return row ?? null;
}

export async function setActive(env: Env, id: number): Promise<void> {
  await env.DB
    .prepare("UPDATE subscribers SET status='active', confirmed_at=? WHERE id=?")
    .bind(Date.now() / 1000, id)
    .run();
}

export async function setUnsubscribed(env: Env, id: number): Promise<void> {
  await env.DB
    .prepare("UPDATE subscribers SET status='unsubscribed', unsubscribed_at=? WHERE id=?")
    .bind(Date.now() / 1000, id)
    .run();
}
