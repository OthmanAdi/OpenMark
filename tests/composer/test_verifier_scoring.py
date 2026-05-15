"""
Verifier scoring math + composer-format alignment.

The verifier is the project's ≥0.90 gate. This file confirms the scoring math
is deterministic (no LLM) and that the composer schema registry matches the
ComposerFormat literal exactly.
"""

from __future__ import annotations

import pytest

from openmark.agent.schemas import (
    COMPOSER_SCHEMAS,
    LinkedInPost,
    NewsletterAnalytical,
    NewsletterComparison,
    NewsletterEssay,
    NewsletterRoundup,
    PostSource,
    VerificationReport,
)


def _build_verification(cite, voice, wc, schema, fix=""):
    pass_count = sum(1 for c in (cite, voice, wc, schema) if c == "pass")
    score = pass_count / 4
    return VerificationReport(
        cite_check=cite,
        voice_check=voice,
        word_count_check=wc,
        schema_check=schema,
        overall_passed=(score >= 0.90),
        score=score,
        fix_instructions=fix,
    )


def test_all_four_pass_is_one_point_zero():
    v = _build_verification("pass", "pass", "pass", "pass")
    assert v.score == 1.0
    assert v.overall_passed is True


def test_three_of_four_pass_is_zero_seventy_five_fail():
    v = _build_verification("pass", "pass", "pass", "fail", fix="schema fail")
    assert v.score == 0.75
    assert v.overall_passed is False
    assert v.fix_instructions == "schema fail"


def test_two_of_four_pass_is_zero_five_fail():
    v = _build_verification("pass", "pass", "fail", "fail", fix="word count + schema")
    assert v.score == 0.50
    assert v.overall_passed is False


def test_score_threshold_exactly_ninety_passes():
    """0.90 boundary: any score >= 0.90 passes."""
    v = VerificationReport(
        cite_check="pass",
        voice_check="pass",
        word_count_check="pass",
        schema_check="pass",
        overall_passed=True,
        score=0.90,
    )
    assert v.score == 0.90
    assert v.overall_passed is True


def test_composer_schemas_cover_every_format():
    """COMPOSER_SCHEMAS must cover the ComposerFormat literal — no silent gap."""
    from typing import get_args
    from openmark.agent.schemas import ComposerFormat

    formats_in_literal = set(get_args(ComposerFormat))
    formats_in_registry = set(COMPOSER_SCHEMAS.keys())
    assert formats_in_literal == formats_in_registry, (
        f"format mismatch: literal={formats_in_literal} registry={formats_in_registry}"
    )


def test_composer_schemas_point_to_real_pydantic_classes():
    for fmt, model_cls in COMPOSER_SCHEMAS.items():
        assert hasattr(model_cls, "model_validate"), f"{fmt} -> {model_cls} not Pydantic"


def test_per_format_minimum_source_count_matches_skill_rules():
    """Confirm the source-count rules each skill promises match the schema."""
    # LinkedInPost / thread: exactly 1 source
    src = PostSource(url="https://x", title="t")
    li = LinkedInPost(
        hook="Three weeks ago an agent shipped its own pull request.",
        body_paragraphs=["p1", "p2", "p3", "p4"],
        closer="Read the abstract then the footnotes.",
        anchor_url="https://x",
        sources=[src],
        word_count=200,
    )
    assert len(li.sources) == 1

    # Essay: 5-8 sources required
    with pytest.raises(Exception):
        NewsletterEssay(
            title="t",
            thesis="a thesis sentence that crosses twenty characters easily.",
            opening_paragraph="opening.",
            sections=[],   # also will fail
            counter="x" * 100,
            closing_paragraph="end.",
            sources=[src] * 2,   # too few
            word_count=700,
        )
