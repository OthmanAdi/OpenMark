"""
Publish orchestrator — single entry point that takes a composer output and
dispatches it to web + email + LinkedIn channels.

Channels (each opt-in via `channels` set):
    "web"      -> write site/src/content/newsletters/<slug>.mdx
    "email"    -> render via Maizzle + send via Resend to active subscribers
    "linkedin" -> post via Voyager API (LinkedInPost shape only)

`publish_issue` is idempotent at the file level (web write is overwriting).
Email sends are NOT idempotent — guard at the caller (mark issue 'sent').
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from openmark.publish.linkedin_post import LinkedInVoyagerClient, PostResult
from openmark.publish.maizzle_render import render_email
from openmark.publish.payload import composer_to_payload
from openmark.publish.resend_client import ResendClient, SendResult
from openmark.publish.subscribers import (
    Subscriber,
    list_active,
    mark_sent,
)


log = logging.getLogger("openmark.publish.orchestrator")


SITE_CONTENT_DIR = Path(__file__).resolve().parents[2] / "site" / "src" / "content" / "newsletters"


@dataclass
class PublishReport:
    slug: str
    channels_attempted: list[str] = field(default_factory=list)
    web_path: Path | None = None
    email_sends: list[SendResult] = field(default_factory=list)
    linkedin_post: PostResult | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ── slug + url helpers ──────────────────────────────────────────────────────


_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def slugify(text: str, *, max_len: int = 60) -> str:
    s = (text or "").lower().strip()
    s = _SLUG_RE.sub("-", s).strip("-")
    return s[:max_len] or "issue"


def issue_slug(composer_output) -> str:
    """Derive a stable slug for the newsletter issue."""
    title = getattr(composer_output, "title", "issue")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{date}-{slugify(title, max_len=50)}"


def web_url_for(slug: str, base_url: str) -> str:
    return f"{base_url.rstrip('/')}/issues/{slug}"


def unsubscribe_url_for(subscriber: Subscriber, base_url: str) -> str:
    return f"{base_url.rstrip('/')}/unsubscribe?t={subscriber.unsubscribe_token}"


# ── Web channel: write site MDX ─────────────────────────────────────────────


def _write_mdx(slug: str, payload: dict) -> Path:
    """
    Write site/src/content/newsletters/<slug>.mdx in Astro Content Collection
    frontmatter format. The Astro site reads this directly.
    """
    SITE_CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_CONTENT_DIR / f"{slug}.mdx"

    fm_keys = ("title", "subtitle", "kicker", "language", "preheader")
    frontmatter_lines = ["---"]
    for k in fm_keys:
        v = payload.get(k)
        if v is not None:
            frontmatter_lines.append(f'{k}: "{_escape_yaml(v)}"')
    frontmatter_lines.append(f'published_at: "{datetime.now(timezone.utc).isoformat()}"')
    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)

    body_parts = []
    if payload.get("hook"):
        body_parts.append(f"> {payload['hook']}")
    for section in (payload.get("sections") or []):
        if section.get("heading"):
            body_parts.append(f"## {section['heading']}")
        body_parts.append(section["body_html"])
    if payload.get("what_im_reading"):
        body_parts.append("## What I'm reading")
        for item in payload["what_im_reading"]:
            body_parts.append(
                f"- **[{item['title']}]({item['url']})** ({item['domain']}) — {item.get('so_what', '')}"
            )
    if payload.get("sources"):
        body_parts.append("## Sources cited")
        for i, s in enumerate(payload["sources"], 1):
            note = f" — {s['note']}" if s.get("note") else ""
            body_parts.append(f"{i}. [{s['title']}]({s['url']}){note}")

    body = "\n\n".join(body_parts)
    out.write_text(f"{frontmatter}\n\n{body}\n", encoding="utf-8")
    log.info(f"[web] wrote {out}")
    return out


def _escape_yaml(s) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.replace('"', '\\"')


# ── Email channel ───────────────────────────────────────────────────────────


def _send_to_subscribers(
    *,
    subject: str,
    html_template: str,
    text_fallback: str,
    payload_base: dict,
    base_url: str,
    from_email: str,
    reply_to: str | None,
    subscribers: Sequence[Subscriber],
) -> list[SendResult]:
    """
    Render once with a generic unsubscribe token, then patch the unsubscribe
    URL per-recipient via a string substitution. Resend doesn't support per-
    recipient template variables in /emails/batch, so we render N times if
    we want true per-recipient HTML. For now we use a marker string in the
    template and substitute it at send time.
    """
    client = ResendClient()
    results: list[SendResult] = []
    items = []
    for sub in subscribers:
        unsub_url = unsubscribe_url_for(sub, base_url)
        # Marker substitution — payload renders with a literal sentinel that
        # we then replace per-recipient. Keeps Resend batch endpoint usable.
        html = html_template.replace("__UNSUBSCRIBE_URL__", unsub_url)
        items.append({"to": sub.email, "subject": subject, "html": html, "text": text_fallback})

    if not items:
        return results

    log.info(f"[email] sending to {len(items)} subscribers via Resend")
    results = client.send_batch(items, from_email=from_email, reply_to=reply_to)
    for res, sub in zip(results, subscribers):
        if res.ok:
            mark_sent(sub.id)
    client.close()
    return results


# ── Public surface ──────────────────────────────────────────────────────────


def publish_issue(
    composer_output,
    *,
    channels: set[str] = frozenset({"web"}),
    base_url: str = "https://openmark.dev",
    from_email: str = "ahmad@openmark.dev",
    reply_to: str | None = None,
    publication_name: str = "OpenMark",
    subscribers: Sequence[Subscriber] | None = None,
    dry_run: bool = False,
) -> PublishReport:
    """
    Dispatch a composer Pydantic output across the chosen channels.

    `channels` is any subset of {"web", "email", "linkedin"}. The publish
    layer never auto-decides — the caller (UI or cron) picks.

    For "email" channel: subscribers defaults to list_active(); pass an
    explicit list to override (e.g. testing to just one address).

    `dry_run=True` builds the artifacts (writes MDX, renders email HTML)
    but does NOT call Resend or LinkedIn. Used for the UI preview.
    """
    slug = issue_slug(composer_output)
    web_url = web_url_for(slug, base_url)
    report = PublishReport(slug=slug, channels_attempted=sorted(channels))

    # Build the email-shaped payload once. We use a sentinel for per-sub URL.
    base_payload = composer_to_payload(
        composer_output,
        web_url=web_url,
        unsubscribe_url="__UNSUBSCRIBE_URL__",
        publication_name=publication_name,
    )

    # ── Web channel ─────────────────────────────────────────────────────────
    if "web" in channels:
        try:
            report.web_path = _write_mdx(slug, base_payload)
        except Exception as e:
            log.exception("[web] failed")
            report.errors.append(f"web: {e!r}")

    # ── Email channel ───────────────────────────────────────────────────────
    if "email" in channels:
        try:
            html = render_email("newsletter", base_payload)
            text_fallback = _build_text_fallback(base_payload)
            subs = list(subscribers) if subscribers is not None else list_active()
            if dry_run:
                log.info(f"[email] dry_run: would send to {len(subs)} subscribers")
                report.email_sends = [
                    SendResult(ok=True, id="dry-run", to=s.email) for s in subs
                ]
            else:
                report.email_sends = _send_to_subscribers(
                    subject=base_payload["title"],
                    html_template=html,
                    text_fallback=text_fallback,
                    payload_base=base_payload,
                    base_url=base_url,
                    from_email=from_email,
                    reply_to=reply_to,
                    subscribers=subs,
                )
        except Exception as e:
            log.exception("[email] failed")
            report.errors.append(f"email: {e!r}")

    # ── LinkedIn channel ────────────────────────────────────────────────────
    if "linkedin" in channels:
        try:
            text = _linkedin_text_from_composer(composer_output, web_url=web_url)
            if dry_run:
                log.info(f"[linkedin] dry_run text={text[:80]!r}")
                report.linkedin_post = PostResult(ok=True, urn="dry-run", permalink=None)
            else:
                client = LinkedInVoyagerClient()
                report.linkedin_post = client.post_text(text, anchor_url=web_url)
                client.close()
        except Exception as e:
            log.exception("[linkedin] failed")
            report.errors.append(f"linkedin: {e!r}")

    return report


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_text_fallback(payload: dict) -> str:
    """Plain-text version of the email for clients that strip HTML."""
    lines = []
    if payload.get("kicker"):
        lines.append(payload["kicker"])
    lines.append(payload["title"])
    if payload.get("subtitle"):
        lines.append("")
        lines.append(payload["subtitle"])
    if payload.get("hook"):
        lines.append("")
        lines.append(payload["hook"])
    for section in payload.get("sections", []):
        lines.append("")
        if section.get("heading"):
            lines.append(section["heading"].upper())
        # Strip HTML tags crudely for the plaintext version
        body = re.sub(r"<[^>]+>", "", section.get("body_html", ""))
        lines.append(body)
    if payload.get("what_im_reading"):
        lines.append("")
        lines.append("WHAT I'M READING")
        for item in payload["what_im_reading"]:
            lines.append(f"- {item['title']} ({item['url']})")
    lines.append("")
    lines.append(f"Web: {payload.get('web_url', '')}")
    lines.append(f"Unsubscribe: {payload.get('unsubscribe_url', '')}")
    return "\n".join(lines)


def _linkedin_text_from_composer(composer_output, *, web_url: str) -> str:
    """
    Convert a composer output to a LinkedIn post body.

    For LinkedInPost: use hook + body_paragraphs + closer + anchor_url verbatim.
    For other composer types: build a minimal post pointing at the web archive.
    """
    from openmark.agent.schemas import LinkedInPost
    if isinstance(composer_output, LinkedInPost):
        parts = [composer_output.hook, ""]
        parts.extend(composer_output.body_paragraphs)
        parts.append("")
        parts.append(composer_output.closer)
        parts.append("")
        parts.append(composer_output.anchor_url)
        return "\n".join(parts)
    title = getattr(composer_output, "title", "New issue")
    subtitle = getattr(composer_output, "hook", None) or getattr(composer_output, "thesis", "")
    return f"{title}\n\n{subtitle}\n\n{web_url}"
