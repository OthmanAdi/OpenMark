"""
Convert composer Pydantic outputs into the flat dict Maizzle expects.

Maizzle (and the Astro site) both consume a shared payload schema:

    {
      kicker:           str | None   "Issue #001", "Sunday Roundup", ...
      title:            str
      subtitle:         str | None
      preheader:        str
      publication_name: str          "OpenMark" default
      language:         str          "en" | "ar-msa" | ...
      web_url:          str          link to web archive of this issue
      unsubscribe_url:  str          per-subscriber token URL
      hook:             str | None
      sections:         [{heading, body_html}]
      what_im_reading:  [{title, url, domain, so_what}]
      sources:          [{title, url, note}]
    }

This module is the single bridge between composer Pydantic schemas and that
payload, so we only have one place to keep in sync when schemas change.
"""

from __future__ import annotations

from typing import Any

from openmark.agent.schemas import (
    LinkedInPost,
    NewsletterAnalytical,
    NewsletterComparison,
    NewsletterEssay,
    NewsletterRoundup,
)


DEFAULT_PUBLICATION = "OpenMark"


def _md_paragraphs_to_html(paragraphs: list[str]) -> str:
    """Wrap each paragraph in <p>; preserves order, no extra escaping."""
    return "".join(f"<p>{p}</p>" for p in paragraphs if p and p.strip())


def _md_to_html_basic(md: str) -> str:
    """
    Lightweight markdown → HTML for composer body_markdown fields.

    Intentionally minimal: paragraphs, bold, italic, inline code, links.
    Avoids adding a heavy markdown dep when our composer outputs are
    deliberately short prose blocks. If we ever need fenced code blocks /
    tables in email bodies, swap this for `markdown-it-py`.
    """
    import re
    md = md or ""
    # Bold **text**
    md = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", md)
    # Italic *text* (single asterisk; don't collide with bold which is already gone)
    md = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", md)
    # Inline code `text`
    md = re.sub(r"`([^`]+)`", r"<code>\1</code>", md)
    # Links [text](url)
    md = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', md)
    # Paragraphs split on blank line
    paragraphs = [p.strip() for p in md.split("\n\n") if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paragraphs)


# ── Per-format adapters ─────────────────────────────────────────────────────


def linkedin_to_payload(post: LinkedInPost, **ctx) -> dict[str, Any]:
    """LinkedIn shape -> email payload. Mostly previewable. Hook + paragraphs + closer."""
    body_html = (
        _md_paragraphs_to_html(post.body_paragraphs) +
        f"<p><strong>{post.closer}</strong></p>"
    )
    return _base_payload(
        title=post.hook,
        hook=None,
        sections=[{"heading": "", "body_html": body_html}],
        what_im_reading=[],
        sources=[
            {"title": s.title, "url": s.url, "note": s.note}
            for s in post.sources
        ],
        language=post.language,
        kicker="LinkedIn / short-form",
        **ctx,
    )


def essay_to_payload(essay: NewsletterEssay, **ctx) -> dict[str, Any]:
    sections = [
        {"heading": s.heading, "body_html": _md_to_html_basic(s.body_markdown)}
        for s in essay.sections
    ]
    sections.append({"heading": "The counter", "body_html": _md_to_html_basic(essay.counter)})
    sections.append({"heading": "", "body_html": _md_to_html_basic(essay.closing_paragraph)})
    return _base_payload(
        title=essay.title,
        subtitle=essay.thesis,
        hook=essay.opening_paragraph,
        sections=sections,
        what_im_reading=[],
        sources=[
            {"title": s.title, "url": s.url, "note": s.note}
            for s in essay.sources
        ],
        language=essay.language,
        kicker="Essay",
        **ctx,
    )


def roundup_to_payload(r: NewsletterRoundup, **ctx) -> dict[str, Any]:
    sections = []
    for bucket in r.buckets:
        items_html = "<ol>"
        for item in bucket.items:
            items_html += (
                f'<li><a href="{item.url}"><strong>{item.title}</strong></a> '
                f'<span style="color:#64748b">({item.domain})</span><br>'
                f'<span>{item.so_what}</span></li>'
            )
        items_html += "</ol>"
        sections.append({"heading": bucket.name, "body_html": items_html})

    return _base_payload(
        title=r.title,
        hook=r.pulse,
        sections=sections,
        what_im_reading=[],
        sources=[
            {"title": s.title, "url": s.url, "note": s.note}
            for s in r.sources
        ],
        language=r.language,
        kicker=r.window_label,
        **ctx,
    )


def comparison_to_payload(c: NewsletterComparison, **ctx) -> dict[str, Any]:
    # Render table as inline HTML in a single section
    header = "<tr><th>Dimension</th>" + "".join(f"<th>{i}</th>" for i in c.items) + "</tr>"
    rows = ""
    for row in c.rows:
        vals = row.values + [""] * max(0, len(c.items) - len(row.values))
        rows += f"<tr><td><strong>{row.dimension}</strong></td>" + \
                "".join(f"<td>{v}</td>" for v in vals[:len(c.items)]) + "</tr>"

    table_html = (
        '<table cellpadding="6" cellspacing="0" border="1" '
        'style="border-collapse:collapse;border-color:#e2e8f0;font-size:14px">'
        f"{header}{rows}</table>"
    )

    picks_html = "<ol>"
    for p in c.picks:
        picks_html += (
            f'<li><strong>{p.item_name}</strong> — if {p.condition}.<br>'
            f'<span>{p.rationale}</span></li>'
        )
    picks_html += "</ol>"

    return _base_payload(
        title=c.title,
        hook=c.recommendation,
        sections=[
            {"heading": "The table", "body_html": table_html},
            {"heading": "How to read this", "body_html": _md_to_html_basic(c.how_to_read)},
            {"heading": "When to pick each", "body_html": picks_html},
        ],
        what_im_reading=[],
        sources=[
            {"title": s.title, "url": s.url, "note": s.note}
            for s in c.sources
        ],
        language=c.language,
        kicker="Comparison",
        **ctx,
    )


def analytical_to_payload(a: NewsletterAnalytical, **ctx) -> dict[str, Any]:
    sections = [
        {"heading": "What happened", "body_html": _md_paragraphs_to_html(a.what_happened_paragraphs)},
        {"heading": "Why it matters", "body_html": _md_to_html_basic(a.why_it_matters)},
    ]
    if a.one_more_thing:
        sections.append(
            {"heading": "One more thing", "body_html": _md_to_html_basic(a.one_more_thing)}
        )

    return _base_payload(
        title=a.title,
        hook=a.hook,
        sections=sections,
        what_im_reading=[
            {"title": item.title, "url": item.url, "domain": item.domain,
             "so_what": item.so_what}
            for item in a.what_im_reading
        ],
        sources=[
            {"title": s.title, "url": s.url, "note": s.note}
            for s in a.sources
        ],
        language=a.language,
        kicker="Newsletter",
        **ctx,
    )


# ── Dispatch ─────────────────────────────────────────────────────────────────


_ADAPTERS = {
    LinkedInPost:        linkedin_to_payload,
    NewsletterEssay:     essay_to_payload,
    NewsletterRoundup:   roundup_to_payload,
    NewsletterComparison: comparison_to_payload,
    NewsletterAnalytical: analytical_to_payload,
}


def composer_to_payload(composer_output, **ctx) -> dict[str, Any]:
    """
    Convert a composer Pydantic instance to a Maizzle-ready payload dict.

    `ctx` must contain at minimum:
        web_url         absolute permalink to this issue on the site archive
        unsubscribe_url per-subscriber unsubscribe URL

    Optional:
        publication_name  defaults to 'OpenMark'
        preheader         derived from hook / subtitle if missing
    """
    adapter = _ADAPTERS.get(type(composer_output))
    if adapter is None:
        raise TypeError(
            f"No payload adapter for {type(composer_output).__name__}. "
            f"Add one to openmark/publish/payload.py."
        )
    return adapter(composer_output, **ctx)


# ── Internals ────────────────────────────────────────────────────────────────


def _base_payload(
    *,
    title: str,
    sections: list[dict],
    what_im_reading: list[dict],
    sources: list[dict],
    language: str,
    kicker: str | None = None,
    subtitle: str | None = None,
    hook: str | None = None,
    web_url: str,
    unsubscribe_url: str,
    publication_name: str = DEFAULT_PUBLICATION,
    preheader: str | None = None,
) -> dict[str, Any]:
    preheader = preheader or (hook or subtitle or title)[:140]
    return {
        "kicker": kicker,
        "title": title,
        "subtitle": subtitle,
        "preheader": preheader,
        "publication_name": publication_name,
        "language": language,
        "web_url": web_url,
        "unsubscribe_url": unsubscribe_url,
        "hook": hook,
        "sections": sections,
        "what_im_reading": what_im_reading,
        "sources": sources,
    }
