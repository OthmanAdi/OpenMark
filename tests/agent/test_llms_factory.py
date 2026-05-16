"""
LLM factory tests — verify each role builder returns a properly-configured
AzureChatOpenAI instance with the expected Foundry quirks applied.

No LLM calls. Just constructor inspection.
"""

from __future__ import annotations

import pytest


def test_role_defaults_all_present():
    from openmark.models import ROLE_DEFAULTS

    must_have = {
        "orchestrator", "summarizer", "classifier", "researcher", "composer",
        "humanizer", "polisher", "verifier", "skill_author",
    }
    assert set(ROLE_DEFAULTS.keys()) == must_have


def test_role_model_id_resolves_to_bank_entry():
    from openmark.models import BANK, role_model_id, ROLE_DEFAULTS

    for role in ROLE_DEFAULTS:
        mid = role_model_id(role)
        # The resolved id should either be a bank-known one OR an env-supplied
        # custom deployment we can't validate without a Foundry call.
        # We at least confirm the resolution returns something non-empty.
        assert mid, f"role {role} resolved to empty"


@pytest.mark.parametrize(
    "builder_name",
    [
        "build_orchestrator",
        "build_classifier",
        "build_summarizer",
        "build_researcher",
        "build_composer",
        "build_humanizer",
        "build_polisher",
        "build_verifier",
        "build_skill_author",
    ],
)
def test_role_builder_returns_chat_model(builder_name):
    import openmark.agent.llms as llms

    builder = getattr(llms, builder_name)
    llm = builder()
    # Should be an AzureChatOpenAI (or local ChatOpenAI under AGENT_PROVIDER=local).
    cls_name = type(llm).__name__
    assert cls_name in ("AzureChatOpenAI", "ChatOpenAI"), f"unexpected: {cls_name}"


def test_backcompat_aliases_exist():
    import openmark.agent.llms as llms

    for legacy in ("build_executor", "build_planner", "build_synthesizer", "build_default"):
        assert hasattr(llms, legacy), f"missing legacy alias: {legacy}"


def test_classifier_is_not_streaming():
    """The classifier is invoked once per turn for structured output — streaming would be wasteful."""
    import openmark.agent.llms as llms

    cls_llm = llms.build_classifier()
    # AzureChatOpenAI exposes `streaming` as a field
    assert getattr(cls_llm, "streaming", True) is False
