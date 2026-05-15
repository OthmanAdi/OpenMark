"""Export-layer tests — pure rendering, no LLM."""

from __future__ import annotations

from openmark.agent.schemas import (
    ComparisonPick,
    ComparisonRow,
    EssaySection,
    LinkedInPost,
    NewsletterAnalytical,
    NewsletterComparison,
    NewsletterEssay,
    NewsletterRoundup,
    PostSource,
    RoundupBucket,
    RoundupItem,
)
from openmark.composer.export import (
    essay_to_html,
    essay_to_markdown,
    essay_to_plaintext,
    linkedin_to_html,
    linkedin_to_markdown,
    linkedin_to_plaintext,
    to_linkedin_html,
    to_linkedin_plaintext,
    to_markdown,
)


def _linkedin() -> LinkedInPost:
    return LinkedInPost(
        hook="Three weeks ago an agent shipped its own pull request.",
        body_paragraphs=[
            "First paragraph.",
            "Second paragraph.",
            "Third paragraph with the surprising number.",
            "Fourth paragraph with the so-what.",
        ],
        closer="Read the abstract then the footnotes.",
        anchor_url="https://example.com/a",
        sources=[PostSource(url="https://example.com/a", title="The anchor source")],
        word_count=200,
    )


def _essay() -> NewsletterEssay:
    return NewsletterEssay(
        title="Title here",
        thesis="This single thesis is what the essay defends to the very end.",
        opening_paragraph="Opening paragraph.",
        sections=[
            EssaySection(heading="First sub-claim", body_markdown="Section 1 body."),
            EssaySection(heading="Second sub-claim", body_markdown="Section 2 body."),
            EssaySection(heading="Third sub-claim", body_markdown="Section 3 body."),
        ],
        counter="The strongest objection is X, but Y holds because of Z. Sharpened thesis follows.",
        closing_paragraph="The closing line.",
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(5)],
        word_count=700,
    )


# ── Markdown ──────────────────────────────────────────────────────────────────


def test_linkedin_markdown_has_required_sections():
    md = linkedin_to_markdown(_linkedin())
    assert md.startswith("# Three weeks ago")
    assert "## Sources cited" in md
    assert "https://example.com/a" in md


def test_essay_markdown_has_required_sections():
    md = essay_to_markdown(_essay())
    assert md.startswith("# Title here")
    assert "> **" in md           # thesis blockquote
    assert "## The counter" in md
    assert "## Sources cited" in md


def test_to_markdown_dispatches_by_type():
    li_md = to_markdown(_linkedin())
    es_md = to_markdown(_essay())
    assert "Three weeks ago" in li_md
    assert "Title here" in es_md


# ── LinkedIn plaintext ────────────────────────────────────────────────────────


def test_plaintext_strips_markdown():
    txt = linkedin_to_plaintext(_linkedin())
    assert "#" not in txt.split("\n")[0]           # no leading hash
    assert "**" not in txt
    # Anchor URL must be on its own line for LinkedIn preview
    assert "\nhttps://example.com/a" in txt or txt.endswith("https://example.com/a")


def test_plaintext_unicode_bold_opt_in():
    txt_plain = linkedin_to_plaintext(_linkedin(), unicode_bold=False)
    txt_bold = linkedin_to_plaintext(_linkedin(), unicode_bold=True)
    assert txt_plain != txt_bold
    # bold variant should NOT contain the ASCII T from the hook's first word
    assert "Three" not in txt_bold.split("\n")[0]


def test_to_linkedin_plaintext_fallback_for_comparison():
    """Comparison has no dedicated plaintext renderer — falls back to stripped md."""
    c = NewsletterComparison(
        title="A vs B",
        recommendation="A wins for speed. B wins for flexibility. Pick your pain.",
        items=["A", "B"],
        rows=[ComparisonRow(dimension="License", values=["MIT", "Apache"])] * 4,
        how_to_read="Read by dimension; the table is the spine.",
        picks=[
            ComparisonPick(item_name="A", condition="you ship weekly", rationale="r1"),
            ComparisonPick(item_name="B", condition="you need flex", rationale="r2"),
        ],
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(3)],
    )
    txt = to_linkedin_plaintext(c)
    assert "A vs B" in txt
    assert "#" not in txt.split("\n")[0]  # heading stripped


# ── LinkedIn HTML ─────────────────────────────────────────────────────────────


def test_html_escapes_content():
    post = _linkedin()
    post.body_paragraphs = [
        "Paragraph with <script>alert('xss')</script> attempt.",
        "Second.",
        "Third.",
        "Fourth.",
    ]
    out = linkedin_to_html(post)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_html_renders_link():
    out = linkedin_to_html(_linkedin())
    assert '<a href="https://example.com/a">The anchor source</a>' in out


def test_essay_html_has_thesis_blockquote_and_counter():
    out = essay_to_html(_essay())
    assert "<blockquote><strong>" in out
    assert "The counter" in out


def test_to_linkedin_html_fallback_for_roundup():
    r = NewsletterRoundup(
        title="Roundup",
        pulse="A short pulse on the week that crosses twenty characters.",
        buckets=[
            RoundupBucket(
                name=f"Bucket {i}",
                items=[
                    RoundupItem(
                        title=f"Item {j}",
                        url=f"https://example.com/{i}/{j}",
                        domain="example.com",
                        so_what=f"matters because {j}.",
                    )
                    for j in range(1, 3)
                ],
            )
            for i in range(1, 4)
        ],
        sources=[PostSource(url=f"https://example.com/s/{i}", title=f"S{i}") for i in range(6)],
        item_count=6,
        window_label="last 7 days",
    )
    out = to_linkedin_html(r)
    assert "<div>" in out and "</div>" in out
    assert "Roundup" in out
