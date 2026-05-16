"""Composer Pydantic -> Maizzle payload adapter tests. Pure, no Maizzle build."""

from __future__ import annotations

import pytest

from openmark.agent.schemas import (
    EssaySection,
    LinkedInPost,
    NewsletterAnalytical,
    NewsletterComparison,
    NewsletterEssay,
    NewsletterRoundup,
    PostSource,
    RoundupBucket,
    RoundupItem,
    ComparisonRow,
    ComparisonPick,
)
from openmark.publish.payload import composer_to_payload, _md_to_html_basic


CTX = {
    "web_url": "https://openmark.dev/issues/test",
    "unsubscribe_url": "https://openmark.dev/u/abc123",
}


def _valid_linkedin():
    return LinkedInPost(
        hook="Three weeks ago an agent shipped its own pull request.",
        body_paragraphs=[
            "First paragraph with detail.",
            "Second paragraph with context.",
            "Third paragraph with the punch.",
            "Fourth paragraph with the so-what.",
        ],
        closer="Read the abstract then the footnotes.",
        anchor_url="https://example.com/a",
        sources=[PostSource(url="https://example.com/a", title="A")],
        word_count=200,
    )


def _valid_essay():
    return NewsletterEssay(
        title="Title",
        thesis="A clear one-sentence thesis the essay defends end to end.",
        opening_paragraph="Opening that sets the hook.",
        sections=[
            EssaySection(heading=f"Section {i}", body_markdown=f"Paragraph for section {i}.")
            for i in range(1, 4)
        ],
        counter=(
            "The strongest objection is that this scales poorly, but on examination "
            "the bottleneck disappears once embeddings are cached at query time."
        ),
        closing_paragraph="Closing line.",
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(5)],
        word_count=700,
    )


def _valid_roundup():
    items = [
        RoundupItem(
            title=f"Item {i}", url=f"https://example.com/{i}",
            domain="example.com", so_what=f"Why item {i} matters in 2026.",
        ) for i in range(1, 4)
    ]
    buckets = [RoundupBucket(name=f"Bucket {b}", items=items) for b in range(1, 4)]
    return NewsletterRoundup(
        title="Week roundup",
        pulse="One sentence pulse summarizing the week's most important shifts.",
        buckets=buckets,
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(6)],
        item_count=9, window_label="last 7 days",
    )


def _valid_comparison():
    rows = [ComparisonRow(dimension=f"Dim {i}", values=["x", "y"]) for i in range(1, 5)]
    return NewsletterComparison(
        title="A vs B",
        recommendation="A wins for speed. B wins for flexibility.",
        items=["A", "B"], rows=rows,
        how_to_read="Read by dimension.",
        picks=[
            ComparisonPick(item_name="A", condition="you ship weekly", rationale="A rationale"),
            ComparisonPick(item_name="B", condition="you need flexibility", rationale="B rationale"),
        ],
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(3)],
    )


def _valid_analytical():
    return NewsletterAnalytical(
        title="Why agent demos keep failing",
        hook="Three releases this week made one thing obvious.",
        what_happened_paragraphs=["Para 1.", "Para 2.", "Para 3."],
        why_it_matters=(
            "Two paragraphs of take connecting the dots: builders watch this, "
            "users skip it, the teams that move first own the pattern for a year."
        ),
        what_im_reading=[
            RoundupItem(title=f"Anchor {i}", url=f"https://example.com/{i}",
                        domain="example.com", so_what=f"Why anchor {i} matters in 2026.")
            for i in range(1, 6)
        ],
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(5)],
        word_count=700,
    )


# ── md helper ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "md, must_contain",
    [
        ("**bold** text",          "<strong>bold</strong>"),
        ("*italic* word",          "<em>italic</em>"),
        ("see `code` inline",      "<code>code</code>"),
        ("[link](https://x.com)",  '<a href="https://x.com">link</a>'),
        ("para one\n\npara two",   "<p>para one</p><p>para two</p>"),
    ],
)
def test_md_to_html_basic(md, must_contain):
    out = _md_to_html_basic(md)
    assert must_contain in out


# ── adapter dispatch ───────────────────────────────────────────────────────


def test_linkedin_payload_shape():
    p = composer_to_payload(_valid_linkedin(), **CTX)
    assert p["title"]
    assert p["sections"] and len(p["sections"]) == 1
    assert "<strong>" in p["sections"][0]["body_html"]
    assert p["sources"] and len(p["sources"]) == 1
    assert p["language"] == "en"
    assert p["web_url"] == CTX["web_url"]
    assert p["unsubscribe_url"] == CTX["unsubscribe_url"]


def test_essay_payload_includes_counter():
    p = composer_to_payload(_valid_essay(), **CTX)
    headings = [s["heading"] for s in p["sections"]]
    assert "The counter" in headings
    assert p["subtitle"]  # thesis fills subtitle
    assert len(p["sources"]) == 5


def test_roundup_payload_one_section_per_bucket():
    p = composer_to_payload(_valid_roundup(), **CTX)
    assert len(p["sections"]) == 3  # 3 buckets -> 3 sections
    for s in p["sections"]:
        assert "<ol>" in s["body_html"]


def test_comparison_payload_table_section():
    p = composer_to_payload(_valid_comparison(), **CTX)
    headings = [s["heading"] for s in p["sections"]]
    assert "The table" in headings
    assert "How to read this" in headings
    assert "When to pick each" in headings


def test_analytical_payload_what_im_reading_populated():
    p = composer_to_payload(_valid_analytical(), **CTX)
    assert len(p["what_im_reading"]) == 5
    for item in p["what_im_reading"]:
        assert item["url"].startswith("https://")
        assert item["domain"]


def test_unknown_type_raises():
    class Foo:
        pass
    with pytest.raises(TypeError):
        composer_to_payload(Foo(), **CTX)
