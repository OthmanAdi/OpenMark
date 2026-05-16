"""
Resend API client — minimal, hardened, dependency-light.

Resend.com is the chosen outbound provider. Free tier: 3000 emails/month,
100/day. API: https://resend.com/docs/api-reference/emails/send-email .
Auth: Bearer <RESEND_API_KEY>.

We avoid the `resend` Python SDK because it's a thin wrapper over httpx and
has a habit of pinning weird transitive versions. Direct httpx calls are
trivial and we already have httpx in requirements.

Usage:
    from openmark.publish.resend_client import ResendClient, send_one, send_batch

    client = ResendClient()
    client.send_one(
        to="ahmad@example.com",
        subject="OpenMark Weekly — RAG patterns",
        html=mjml_compiled_html,
        text="Plain text fallback",
        from_email="ahmad@openmark.dev",
        reply_to="ahmad@openmark.dev",
    )

    client.send_batch(
        [{"to": s.email, "subject": ..., "html": ..., "text": ...}
         for s in active_subscribers],
        from_email="ahmad@openmark.dev",
    )
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

import httpx


log = logging.getLogger("openmark.publish.resend")


RESEND_API_BASE = "https://api.resend.com"
RESEND_BATCH_MAX = 100         # Resend hard cap per batch call
RESEND_DEFAULT_TIMEOUT = 30.0  # seconds


@dataclass
class SendResult:
    ok: bool
    id: str | None
    to: str
    error: str | None = None


class ResendClient:
    """
    Stateless-ish wrapper. Re-uses one httpx.Client across calls.

    Env:
        RESEND_API_KEY    (required for sends)
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = RESEND_DEFAULT_TIMEOUT,
        max_retries: int = 3,
        base_url: str = RESEND_API_BASE,
    ) -> None:
        self.api_key = api_key or os.environ.get("RESEND_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self._client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()

    # ── sending ─────────────────────────────────────────────────────────────

    def send_one(
        self,
        *,
        to: str,
        subject: str,
        html: str,
        text: str | None = None,
        from_email: str,
        reply_to: str | None = None,
        headers: dict | None = None,
        tags: dict[str, str] | None = None,
    ) -> SendResult:
        if not self.api_key:
            return SendResult(ok=False, id=None, to=to,
                              error="RESEND_API_KEY missing")
        payload: dict[str, Any] = {
            "from": from_email,
            "to": [to],
            "subject": subject,
            "html": html,
        }
        if text:
            payload["text"] = text
        if reply_to:
            payload["reply_to"] = reply_to
        if headers:
            payload["headers"] = headers
        if tags:
            payload["tags"] = [{"name": k, "value": v} for k, v in tags.items()]

        return self._post_with_retry("/emails", payload, to=to)

    def send_batch(
        self,
        items: Sequence[dict],
        *,
        from_email: str,
        reply_to: str | None = None,
    ) -> list[SendResult]:
        """
        Items: list of dicts with keys (to, subject, html, text?, headers?).
        Auto-chunks into batches of RESEND_BATCH_MAX. One API call per chunk.
        """
        if not self.api_key:
            return [SendResult(ok=False, id=None, to=i.get("to", ""),
                               error="RESEND_API_KEY missing") for i in items]

        out: list[SendResult] = []
        for chunk_start in range(0, len(items), RESEND_BATCH_MAX):
            chunk = items[chunk_start:chunk_start + RESEND_BATCH_MAX]
            batch_payload = []
            for it in chunk:
                msg: dict[str, Any] = {
                    "from": from_email,
                    "to": [it["to"]],
                    "subject": it["subject"],
                    "html": it["html"],
                }
                if it.get("text"):
                    msg["text"] = it["text"]
                if reply_to:
                    msg["reply_to"] = reply_to
                if it.get("headers"):
                    msg["headers"] = it["headers"]
                batch_payload.append(msg)

            try:
                resp = self._http().post("/emails/batch", json=batch_payload)
                resp.raise_for_status()
                data = resp.json()
                # Resend returns {"data": [{"id": ...}, ...]} in order
                ids = [d.get("id") for d in (data.get("data") or [])]
                for i, item in enumerate(chunk):
                    out.append(SendResult(
                        ok=True,
                        id=(ids[i] if i < len(ids) else None),
                        to=item["to"],
                    ))
            except httpx.HTTPStatusError as e:
                err = f"{e.response.status_code}: {e.response.text[:200]}"
                log.warning(f"[resend.batch] failed: {err}")
                for item in chunk:
                    out.append(SendResult(ok=False, id=None, to=item["to"], error=err))
            except Exception as e:
                log.warning(f"[resend.batch] exception: {e!r}")
                for item in chunk:
                    out.append(SendResult(ok=False, id=None, to=item["to"], error=repr(e)))
        return out

    # ── internals ───────────────────────────────────────────────────────────

    def _post_with_retry(self, path: str, payload: dict, *, to: str) -> SendResult:
        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._http().post(path, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return SendResult(ok=True, id=data.get("id"), to=to)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                last_err = f"{status}: {e.response.text[:200]}"
                # 4xx (other than 429) = client error, don't retry
                if 400 <= status < 500 and status != 429:
                    break
            except Exception as e:
                last_err = repr(e)
            time.sleep(min(2 ** attempt * 0.5, 8))
        return SendResult(ok=False, id=None, to=to, error=last_err or "unknown")


# ── module-level convenience functions ──────────────────────────────────────


def send_one(**kwargs) -> SendResult:
    return ResendClient().send_one(**kwargs)


def send_batch(items, *, from_email, reply_to=None) -> list[SendResult]:
    return ResendClient().send_batch(items, from_email=from_email, reply_to=reply_to)
