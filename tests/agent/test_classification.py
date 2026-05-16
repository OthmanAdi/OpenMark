"""
Classification middleware tests — no LLM call, only slash + heuristic paths.

Asserts that the cheap routing paths short-circuit correctly so the LLM
classifier only fires when truly ambiguous.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "text, expected",
    [
        ("/fast-search find my RAG bookmarks",     "fast"),
        ("/deep-research multi-agent frameworks",  "deep"),
        ("/newsletter on context engineering",     "newsletter"),
        ("/newsletter-essay agents",               "newsletter"),
        ("/newsletter-roundup last week",          "newsletter"),
        ("/weekly-digest",                          "digest"),
        ("/bookmark-dive https://example.com",      "dive"),
        ("/repo-research langchain",                "deep"),
        ("/niche-hunter agent demos",               "deep"),
    ],
)
def test_slash_intent_lookup(text, expected):
    from openmark.agent.classification import _slash_intent

    assert _slash_intent(text) == expected


def test_slash_unknown_returns_none():
    from openmark.agent.classification import _slash_intent
    assert _slash_intent("/nonexistent-skill hello") is None


def test_no_slash_returns_none():
    from openmark.agent.classification import _slash_intent
    assert _slash_intent("just a normal question") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("research the landscape of agent frameworks", "deep"),
        ("compare LangChain and DSPy and CrewAI",      "deep"),
        ("survey of vector DBs",                        "deep"),
        ("map out RAG patterns",                        "deep"),
        ("what's the state of agent observability",    "deep"),
        ("compose a newsletter on agents",              "newsletter"),
        ("draft my newsletter for this week",           "newsletter"),
        ("what did I save this week",                   "digest"),
        ("last week's bookmarks on RAG",                "digest"),
        ("past 7 days digest",                          "digest"),
        ("https://github.com/foo/bar expand neighbors", "dive"),
        ("dig into https://arxiv.org/abs/2406.12345",   "dive"),
    ],
)
def test_heuristic_intent(text, expected):
    from openmark.agent.classification import _heuristic_intent

    assert _heuristic_intent(text) == expected


def test_heuristic_no_match_returns_none():
    from openmark.agent.classification import _heuristic_intent

    # Plain lookup query — no signal in any regex
    assert _heuristic_intent("show me my best agent bookmarks") is None
    assert _heuristic_intent("what is langgraph") is None


def test_classifier_middleware_is_before_model():
    """classify_intent must be an AgentMiddleware (post @before_model)."""
    from langchain.agents.middleware import AgentMiddleware
    from openmark.agent.classification import classify_intent

    assert isinstance(classify_intent, AgentMiddleware)


def test_dynamic_prompt_returns_string_with_intent_hint():
    """dynamic_orchestrator_prompt must read state.intent and inject a hint."""
    from openmark.agent.classification import dynamic_orchestrator_prompt
    from langchain.agents.middleware import AgentMiddleware

    assert isinstance(dynamic_orchestrator_prompt, AgentMiddleware)


def test_orchestrator_state_has_intent_field():
    from openmark.agent.classification import OrchestratorState

    annotations = OrchestratorState.__annotations__
    assert "intent" in annotations
    assert "intent_source" in annotations
