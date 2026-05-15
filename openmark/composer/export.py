"""
Render composer output (LinkedInPost / NewsletterEssay / ...) into the three
delivery formats the user asked for:

  - markdown                — what the chat UI auto-saves today
  - linkedin_plaintext      — paste straight into the LinkedIn editor
                              (no markdown, URL footnotes, line-break tuned)
  - linkedin_html           — paste-as-rich-text into LinkedIn's editor

The export layer has ZERO LLM calls. Pure stdlib + the Pydantic shapes from
openmark.agent.schemas. That makes it cheap to call, test, and reuse from
both the Gradio UI and the future public endpoint.
"""

from __future__ import annotations

import html
from typing import Union

from openmark.agent.schemas import (
    LinkedInPost,
    NewsletterAnalytical,
    NewsletterComparison,
    NewsletterEssay,
    NewsletterRoundup,
    PostSource,
)

ComposerOutput = Union[
    LinkedInPost,
    NewsletterEssay,
    NewsletterRoundup,
    NewsletterComparison,
    NewsletterAnalytical,
]


# ── Markdown renderers (used by the Gradio auto-save path) ───────────────────


def _md_sources(sources: list[PostSource]) -> str:
    lines = ["## Sources cited"]
    for i, s in enumerate(sources, 1):
        note = f" — {s.note}" if s.note else ""
        lines.append(f"{i}. [{s.title}]({s.url}){note}")
    return "\n".join(lines)


def linkedin_to_markdown(post: LinkedInPost) -> str:
    parts = [f"# {post.hook}", ""]
    for p in post.body_paragraphs:
        parts.append(p)
        parts.append("")
    parts.append(post.closer)
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(_md_sources(post.sources))
    parts.append("")
    parts.append(
        f"_~{post.word_count} words · thread style · 1 anchor source · language={post.language}_"
    )
    return "\n".join(parts)


def essay_to_markdown(essay: NewsletterEssay) -> str:
    parts = [
        f"# {essay.title}",
        "",
        f"> **{essay.thesis}**",
        "",
        essay.opening_paragraph,
        "",
    ]
    for section in essay.sections:
        parts.append(f"## {section.heading}")
        parts.append("")
        parts.append(section.body_markdown)
        parts.append("")
    parts.append("## The counter")
    parts.append("")
    parts.append(essay.counter)
    parts.append("")
    parts.append(essay.closing_paragraph)
    parts.append("")
    parts.append(_md_sources(essay.sources))
    parts.append("")
    parts.append(
        f"_{essay.word_count} words · {len(essay.sources)} sources · language={essay.language}_"
    )
    return "\n".join(parts)


def roundup_to_markdown(r: NewsletterRoundup) -> str:
    parts = [f"# {r.title}", "", f"_{r.pulse}_", ""]
    for bucket in r.buckets:
        parts.append(f"## {bucket.name}  ({len(bucket.items)})")
        parts.append("")
        for i, item in enumerate(bucket.items, 1):
            parts.append(f"{i}. **{item.title}** — [{item.domain}]({item.url})")
            parts.append("")
            parts.append(f"   {item.so_what}")
            parts.append("")
    parts.append(_md_sources(r.sources))
    parts.append("")
    parts.append(
        f"_{r.item_count} items · feed pulled {r.window_label} · language={r.language}_"
    )
    return "\n".join(parts)


def comparison_to_markdown(c: NewsletterComparison) -> str:
    parts = [
        f"# {c.title}",
        "",
        f"> **{c.recommendation}**",
        "",
        "## The table",
        "",
    ]
    header = "| Dimension | " + " | ".join(c.items) + " |"
    sep = "|---" * (len(c.items) + 1) + "|"
    parts.append(header)
    parts.append(sep)
    for row in c.rows:
        # Pad to match item count if model under-filled (defensive)
        values = list(row.values) + [""] * max(0, len(c.items) - len(row.values))
        parts.append("| **" + row.dimension + "** | " + " | ".join(values[: len(c.items)]) + " |")
    parts.append("")
    parts.append("## How to read this")
    parts.append("")
    parts.append(c.how_to_read)
    parts.append("")
    parts.append("## When to pick each")
    parts.append("")
    for p in c.picks:
        parts.append(f"### Pick **{p.item_name}** if {p.condition}.")
        parts.append("")
        parts.append(p.rationale)
        parts.append("")
    parts.append(_md_sources(c.sources))
    parts.append("")
    parts.append(
        f"_Comparing {len(c.items)} items across {len(c.rows)} dimensions · {len(c.sources)} sources · language={c.language}_"
    )
    return "\n".join(parts)


def analytical_to_markdown(a: NewsletterAnalytical) -> str:
    parts = [f"# {a.title}", "", a.hook, "", "## What happened", ""]
    for p in a.what_happened_paragraphs:
        parts.append(p)
        parts.append("")
    parts.append("## Why it matters")
    parts.append("")
    parts.append(a.why_it_matters)
    parts.append("")
    parts.append("## What I'm reading")
    parts.append("")
    for i, item in enumerate(a.what_im_reading, 1):
        parts.append(f"{i}. **{item.title}** — [{item.domain}]({item.url})")
        parts.append("")
        parts.append(f"   {item.so_what}")
        parts.append("")
    if a.one_more_thing:
        parts.append("## One more thing")
        parts.append("")
        parts.append(a.one_more_thing)
        parts.append("")
    parts.append(_md_sources(a.sources))
    parts.append("")
    parts.append(
        f"_{a.word_count} words · {len(a.sources)} OpenMark hits cited · language={a.language}_"
    )
    return "\n".join(parts)


# ── LinkedIn plaintext (the copy-paste win Ahmad asked for) ──────────────────
#
# LinkedIn's editor strips markdown. Hashes survive but render as plain text.
# Hyperlinks must be plain URLs on their own line for LinkedIn to auto-detect
# and preview. Bold via **...** doesn't survive either; Unicode bold is the
# trick most LinkedIn power-users use, but it hurts accessibility and we keep
# it as opt-in (kwarg unicode_bold=False default).


def _to_unicode_bold(s: str) -> str:
    """Map ASCII letters/digits to their Unicode 'mathematical bold' glyphs."""
    out = []
    for ch in s:
        cp = ord(ch)
        if 0x41 <= cp <= 0x5A:        # A-Z
            out.append(chr(0x1D400 + (cp - 0x41)))
        elif 0x61 <= cp <= 0x7A:      # a-z
            out.append(chr(0x1D41A + (cp - 0x61)))
        elif 0x30 <= cp <= 0x39:      # 0-9
            out.append(chr(0x1D7CE + (cp - 0x30)))
        else:
            out.append(ch)
    return "".join(out)


def linkedin_to_plaintext(post: LinkedInPost, *, unicode_bold: bool = False) -> str:
    """
    LinkedIn-editor-ready plain text. Paste with Ctrl-V; preserves line breaks.
    - No markdown headings, no asterisks
    - The single anchor URL stands on its own line so LinkedIn renders a preview
    - Sources block at the end is a flat list, one URL per line
    """
    hook = _to_unicode_bold(post.hook) if unicode_bold else post.hook
    lines: list[str] = [hook, ""]
    for p in post.body_paragraphs:
        lines.append(p)
        lines.append("")
    lines.append(post.closer)
    lines.append("")
    # Anchor URL on its own line so LinkedIn auto-previews it.
    lines.append(post.anchor_url)
    return "\n".join(lines)


def essay_to_plaintext(essay: NewsletterEssay, *, unicode_bold: bool = False) -> str:
    title = _to_unicode_bold(essay.title) if unicode_bold else essay.title
    lines: list[str] = [title, "", essay.thesis, "", essay.opening_paragraph, ""]
    for section in essay.sections:
        h = _to_unicode_bold(section.heading) if unicode_bold else section.heading
        lines.append(h)
        lines.append("")
        lines.append(section.body_markdown)
        lines.append("")
    lines.append("The counter")
    lines.append("")
    lines.append(essay.counter)
    lines.append("")
    lines.append(essay.closing_paragraph)
    lines.append("")
    lines.append("Sources:")
    for s in essay.sources:
        lines.append(s.url)
    return "\n".join(lines)


def roundup_to_plaintext(r: NewsletterRoundup) -> str:
    lines: list[str] = [r.title, "", r.pulse, ""]
    for bucket in r.buckets:
        lines.append(f"{bucket.name} ({len(bucket.items)})")
        lines.append("")
        for item in bucket.items:
            lines.append(f"- {item.title}")
            lines.append(f"  {item.so_what}")
            lines.append(f"  {item.url}")
            lines.append("")
    return "\n".join(lines)


# ── LinkedIn HTML (paste-as-rich-text) ───────────────────────────────────────
#
# When pasted into LinkedIn's editor, HTML survives partially: <p>, <strong>,
# <a> all render. Keep it minimal.


def _escape(s: str) -> str:
    return html.escape(s, quote=False)


def linkedin_to_html(post: LinkedInPost) -> str:
    body_paras = "\n".join(f"<p>{_escape(p)}</p>" for p in post.body_paragraphs)
    src = post.sources[0]
    return (
        "<div>"
        f"<p><strong>{_escape(post.hook)}</strong></p>"
        f"{body_paras}"
        f"<p>{_escape(post.closer)}</p>"
        f'<p><a href="{_escape(src.url)}">{_escape(src.title)}</a></p>'
        "</div>"
    )


def essay_to_html(essay: NewsletterEssay) -> str:
    parts: list[str] = [
        "<article>",
        f"<h1>{_escape(essay.title)}</h1>",
        f"<blockquote><strong>{_escape(essay.thesis)}</strong></blockquote>",
        f"<p>{_escape(essay.opening_paragraph)}</p>",
    ]
    for section in essay.sections:
        parts.append(f"<h2>{_escape(section.heading)}</h2>")
        parts.append(f"<p>{_escape(section.body_markdown)}</p>")
    parts.append("<h2>The counter</h2>")
    parts.append(f"<p>{_escape(essay.counter)}</p>")
    parts.append(f"<p>{_escape(essay.closing_paragraph)}</p>")
    parts.append("<h2>Sources cited</h2><ol>")
    for s in essay.sources:
        note = f" — {_escape(s.note)}" if s.note else ""
        parts.append(f'<li><a href="{_escape(s.url)}">{_escape(s.title)}</a>{note}</li>')
    parts.append("</ol></article>")
    return "".join(parts)


# ── Dispatch ──────────────────────────────────────────────────────────────────

# Per-format renderer tables. Lookup by isinstance, not str, so misuse blows
# up at call time instead of producing a wrong-shape output.
_MD_RENDERERS: dict[type, callable] = {
    LinkedInPost:          linkedin_to_markdown,
    NewsletterEssay:       essay_to_markdown,
    NewsletterRoundup:     roundup_to_markdown,
    NewsletterComparison:  comparison_to_markdown,
    NewsletterAnalytical:  analytical_to_markdown,
}

_PLAINTEXT_RENDERERS: dict[type, callable] = {
    LinkedInPost:      linkedin_to_plaintext,
    NewsletterEssay:   essay_to_plaintext,
    NewsletterRoundup: roundup_to_plaintext,
    # comparison + analytical fall back to markdown -> stripped, see below
}

_HTML_RENDERERS: dict[type, callable] = {
    LinkedInPost:    linkedin_to_html,
    NewsletterEssay: essay_to_html,
    # roundup, comparison, analytical fall back to a generic md->html, see below
}


def to_markdown(post: ComposerOutput) -> str:
    fn = _MD_RENDERERS.get(type(post))
    if fn is None:
        raise TypeError(f"No markdown renderer for {type(post).__name__}")
    return fn(post)


def to_linkedin_plaintext(post: ComposerOutput, *, unicode_bold: bool = False) -> str:
    fn = _PLAINTEXT_RENDERERS.get(type(post))
    if fn is None:
        # Fall back: render markdown, strip the common markdown bits.
        md = to_markdown(post)
        return _strip_markdown(md)
    if fn in (linkedin_to_plaintext, essay_to_plaintext):
        return fn(post, unicode_bold=unicode_bold)
    return fn(post)


def to_linkedin_html(post: ComposerOutput) -> str:
    fn = _HTML_RENDERERS.get(type(post))
    if fn is not None:
        return fn(post)
    # Fall back: minimal wrap of the plaintext.
    txt = to_linkedin_plaintext(post)
    paras = [f"<p>{_escape(line)}</p>" for line in txt.split("\n\n") if line.strip()]
    return "<div>" + "".join(paras) + "</div>"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _strip_markdown(md: str) -> str:
    """Cheap stripper. Drops headings, bold/italic markers, list bullets, blockquote arrows."""
    out_lines: list[str] = []
    for line in md.splitlines():
        s = line
        # Strip leading heading hashes
        while s.startswith("#"):
            s = s[1:]
        s = s.lstrip()
        # Drop blockquote markers
        if s.startswith(">"):
            s = s[1:].lstrip()
        # Strip bold/italic markers
        s = s.replace("**", "").replace("__", "")
        # Drop leading list bullets
        if s.startswith("- ") or s.startswith("* "):
            s = s[2:]
        out_lines.append(s)
    return "\n".join(out_lines).strip()
