"""Schema round-trip tests — no LLM, no network."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openmark.agent.schemas import (
    ComparisonPick,
    ComparisonRow,
    COMPOSER_SCHEMAS,
    EssaySection,
    LinkedInPost,
    NewsletterAnalytical,
    NewsletterComparison,
    NewsletterEssay,
    NewsletterRoundup,
    PostSource,
    RoundupBucket,
    RoundupItem,
    VerificationReport,
)


# ── LinkedInPost ──────────────────────────────────────────────────────────────


def _valid_linkedin() -> dict:
    return dict(
        hook="Three weeks ago an agent shipped its own pull request.",
        body_paragraphs=[
            "First paragraph with one cite.",
            "Second paragraph with context.",
            "Third paragraph with the surprising number.",
            "Fourth paragraph with the so-what.",
        ],
        closer="Read the abstract then the footnotes.",
        anchor_url="https://example.com/a",
        sources=[PostSource(url="https://example.com/a", title="A")],
        word_count=200,
    )


def test_linkedin_happy_path():
    post = LinkedInPost(**_valid_linkedin())
    assert post.language == "en"
    assert post.humanizer_applied is False
    assert post.voice_check == "pass"


def test_linkedin_rejects_short_hook():
    bad = _valid_linkedin()
    bad["hook"] = "short"
    with pytest.raises(ValidationError):
        LinkedInPost(**bad)


def test_linkedin_rejects_few_paragraphs():
    bad = _valid_linkedin()
    bad["body_paragraphs"] = ["just one paragraph here"]
    with pytest.raises(ValidationError):
        LinkedInPost(**bad)


def test_linkedin_rejects_two_sources():
    bad = _valid_linkedin()
    bad["sources"] = [
        PostSource(url="https://a", title="A"),
        PostSource(url="https://b", title="B"),
    ]
    with pytest.raises(ValidationError):
        LinkedInPost(**bad)


def test_linkedin_rejects_word_count_under_180():
    bad = _valid_linkedin()
    bad["word_count"] = 100
    with pytest.raises(ValidationError):
        LinkedInPost(**bad)


def test_linkedin_arabic_language():
    body = _valid_linkedin()
    body["language"] = "ar-egt"
    post = LinkedInPost(**body)
    assert post.language == "ar-egt"


# ── NewsletterEssay ───────────────────────────────────────────────────────────


def _valid_essay() -> dict:
    sections = [
        EssaySection(heading=f"Section {i}", body_markdown=f"Paragraph {i}.")
        for i in range(1, 4)
    ]
    return dict(
        title="Title here",
        thesis="The whole essay defends this single sentence to the very end.",
        opening_paragraph="Opening paragraph that sets the hook.",
        sections=sections,
        counter=(
            "The strongest objection is X, but on examination Y holds because "
            "Z. The thesis sharpens, not breaks."
        ),
        closing_paragraph="The closing line that lands.",
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(5)],
        word_count=700,
    )


def test_essay_happy_path():
    e = NewsletterEssay(**_valid_essay())
    assert len(e.sections) == 3
    assert len(e.sources) == 5


def test_essay_rejects_short_counter():
    bad = _valid_essay()
    bad["counter"] = "too short"
    with pytest.raises(ValidationError):
        NewsletterEssay(**bad)


def test_essay_rejects_too_few_sources():
    bad = _valid_essay()
    bad["sources"] = bad["sources"][:2]
    with pytest.raises(ValidationError):
        NewsletterEssay(**bad)


# ── NewsletterRoundup ─────────────────────────────────────────────────────────


def _valid_roundup() -> dict:
    items = [
        RoundupItem(
            title=f"Item {i}",
            url=f"https://example.com/{i}",
            domain="example.com",
            so_what=f"This matters because reason {i}.",
        )
        for i in range(1, 4)
    ]
    buckets = [
        RoundupBucket(name=f"Bucket {i}", items=items)
        for i in range(1, 4)
    ]
    return dict(
        title="Week of 2026-05-15 — Punchy headline",
        pulse="One sentence pulse on the week.",
        buckets=buckets,
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(6)],
        item_count=9,
        window_label="last 7 days",
    )


def test_roundup_happy_path():
    r = NewsletterRoundup(**_valid_roundup())
    assert len(r.buckets) == 3
    assert r.item_count == 9


def test_roundup_rejects_one_bucket():
    bad = _valid_roundup()
    bad["buckets"] = bad["buckets"][:1]
    with pytest.raises(ValidationError):
        NewsletterRoundup(**bad)


# ── NewsletterComparison ──────────────────────────────────────────────────────


def _valid_comparison() -> dict:
    rows = [
        ComparisonRow(dimension=f"Dim {i}", values=["x", "y"])
        for i in range(1, 5)
    ]
    picks = [
        ComparisonPick(item_name="A", condition="you ship weekly", rationale="rationale A"),
        ComparisonPick(item_name="B", condition="you need flexibility", rationale="rationale B"),
    ]
    return dict(
        title="A vs B",
        recommendation="A wins for speed. B wins for flexibility.",
        items=["A", "B"],
        rows=rows,
        how_to_read="Read by dimension; the table is the spine.",
        picks=picks,
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(3)],
    )


def test_comparison_happy_path():
    c = NewsletterComparison(**_valid_comparison())
    assert len(c.items) == 2
    assert len(c.rows) == 4


# ── NewsletterAnalytical ──────────────────────────────────────────────────────


def _valid_analytical() -> dict:
    return dict(
        title="Why agent demos keep failing",
        hook="Three releases this week made one thing obvious.",
        what_happened_paragraphs=[
            "Paragraph one with a cite.",
            "Paragraph two with another cite.",
            "Paragraph three with the punch.",
        ],
        why_it_matters="Two paragraphs of take. The connect-the-dots. Builders watch this; users skip it.",
        what_im_reading=[
            RoundupItem(
                title=f"Anchor {i}",
                url=f"https://example.com/{i}",
                domain="example.com",
                so_what=f"Why it matters: reason {i}.",
            )
            for i in range(1, 6)
        ],
        sources=[PostSource(url=f"https://example.com/{i}", title=f"S{i}") for i in range(5)],
        word_count=700,
    )


def test_analytical_happy_path():
    a = NewsletterAnalytical(**_valid_analytical())
    assert len(a.what_im_reading) == 5
    assert a.one_more_thing is None


def test_analytical_rejects_few_reads():
    bad = _valid_analytical()
    bad["what_im_reading"] = bad["what_im_reading"][:3]
    with pytest.raises(ValidationError):
        NewsletterAnalytical(**bad)


# ── VerificationReport ────────────────────────────────────────────────────────


def test_verification_happy_path():
    v = VerificationReport(
        cite_check="pass",
        voice_check="pass",
        word_count_check="pass",
        schema_check="pass",
        overall_passed=True,
        score=1.0,
    )
    assert v.score == 1.0


def test_verification_score_bounds():
    with pytest.raises(ValidationError):
        VerificationReport(
            cite_check="pass",
            voice_check="pass",
            word_count_check="pass",
            schema_check="pass",
            overall_passed=True,
            score=1.5,
        )


# ── Registry ──────────────────────────────────────────────────────────────────


def test_composer_schemas_registry_complete():
    expected = {"linkedin", "thread", "essay", "roundup", "comparison", "analytical"}
    assert set(COMPOSER_SCHEMAS) == expected
    assert COMPOSER_SCHEMAS["linkedin"] is LinkedInPost
    assert COMPOSER_SCHEMAS["thread"] is LinkedInPost
    assert COMPOSER_SCHEMAS["essay"] is NewsletterEssay
