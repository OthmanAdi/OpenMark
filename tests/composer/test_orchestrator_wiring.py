"""
Orchestrator wiring tests — confirm the graph compiles AND all sub-agent
contracts are intact. NO real LLM call. NO Neo4j call.

For the live LLM end-to-end smoke test see `test_e2e_smoke.py` (skipped by
default, opt-in via OPENMARK_RUN_LIVE_E2E=1).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_subagent_registry_complete():
    from openmark.composer.subagents import build_all_subagents

    subs = build_all_subagents()
    names = {s["name"] for s in subs}
    expected = {
        "researcher",
        "composer-linkedin",
        "composer-thread",
        "composer-essay",
        "composer-roundup",
        "composer-comparison",
        "composer-analytical",
        "humanizer",
        "polisher",
        "verifier",
        "skill-author",
    }
    assert names == expected, f"missing or extra: {names ^ expected}"


def test_researcher_has_all_retrieval_tools():
    from openmark.composer.subagents import RESEARCHER_TOOLS, build_researcher

    sub = build_researcher()
    assert sub["tools"] is RESEARCHER_TOOLS
    tool_names = {getattr(t, "name", "") for t in RESEARCHER_TOOLS}
    # Sanity: every primary retrieval tool present
    must_have = {
        "search_semantic",
        "search_by_category",
        "search_by_community",
        "find_by_tag",
        "search_linkedin",
        "search_youtube",
        "get_bookmark_full",
        "web_search",
        "web_fetch",
        "github_repo_intel",
    }
    missing = must_have - tool_names
    assert not missing, f"researcher missing: {missing}"


def test_composers_have_response_format_and_no_tools():
    from openmark.composer.subagents import build_all_subagents

    for s in build_all_subagents():
        if s["name"].startswith("composer-"):
            assert "response_format" in s, f"{s['name']} missing response_format"
            assert s.get("tools") == [], f"{s['name']} must have empty tools list"


def test_verifier_has_verification_report_response_format():
    from openmark.agent.schemas import VerificationReport
    from openmark.composer.subagents import build_verifier

    v = build_verifier()
    rf = v.get("response_format")
    assert rf is not None
    # ToolStrategy wraps the schema — check by attribute
    assert getattr(rf, "schema", None) is VerificationReport or (
        getattr(rf, "schema_", None) is VerificationReport
    ) or "VerificationReport" in repr(rf)


def test_skill_author_has_only_write_skill():
    from openmark.agent.tools import write_skill
    from openmark.composer.subagents import build_skill_author

    sa = build_skill_author()
    assert sa["tools"] == [write_skill]


def test_humanizer_polisher_have_no_tools_and_no_schema():
    from openmark.composer.subagents import build_humanizer, build_polisher

    for sub in (build_humanizer(), build_polisher()):
        assert sub.get("tools") == []
        assert "response_format" not in sub


class _StubLLM:
    """Minimal stand-in for a BaseChatModel — just enough for create_deep_agent
    to bind tools without hitting the wire."""

    def bind_tools(self, *a, **kw):
        return self

    def with_structured_output(self, *a, **kw):
        return self

    def invoke(self, *a, **kw):
        return None


def _capture_create_deep_agent_kwargs():
    """Returns (capture_dict, side_effect_fn) — side_effect bypasses the real
    create_deep_agent so no LLM is needed and we can inspect the args."""
    captured: dict = {}

    class _Stub:
        def invoke(self, *a, **kw): pass
        def stream(self, *a, **kw): pass
        def get_state(self, *a, **kw): pass

    def _side(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Stub()

    return captured, _side


def test_orchestrator_compiles_via_create_deep_agent_path():
    """build_composer_orchestrator must call create_deep_agent and return its result."""
    from openmark.composer import orchestrator

    captured, side = _capture_create_deep_agent_kwargs()
    with patch.object(orchestrator, "create_deep_agent", side_effect=side):
        with patch.object(orchestrator, "build_executor", return_value=_StubLLM()):
            with patch.object(orchestrator, "_warm_up_tools"):
                agent = orchestrator.build_composer_orchestrator()
    assert hasattr(agent, "invoke")
    assert "kwargs" in captured, "create_deep_agent was not called"
    kw = captured["kwargs"]
    assert "model" in kw and "subagents" in kw and "middleware" in kw
    assert kw["model"].__class__.__name__ == "_StubLLM"


def test_orchestrator_uses_existing_event_bus_middleware():
    """Beautiful UI promise: the new orchestrator must wire the same
    `tool_event_middleware` the chat agent uses, so the Compose UI cards work."""
    from openmark.agent.middleware import (
        OpenMarkSkillMiddleware,
        slash_skill_loader,
        tool_event_middleware,
    )
    from openmark.composer import orchestrator

    captured, side = _capture_create_deep_agent_kwargs()
    with patch.object(orchestrator, "create_deep_agent", side_effect=side):
        with patch.object(orchestrator, "build_executor", return_value=_StubLLM()):
            with patch.object(orchestrator, "_warm_up_tools"):
                orchestrator.build_composer_orchestrator()

    mw = list(captured["kwargs"].get("middleware") or [])
    assert slash_skill_loader in mw, f"slash_skill_loader missing from middleware list ({len(mw)} items)"
    assert tool_event_middleware in mw, "tool_event_middleware missing — UI cards will be empty"
    assert any(isinstance(m, OpenMarkSkillMiddleware) for m in mw), (
        "OpenMarkSkillMiddleware missing — skill catalogue not exposed to orchestrator"
    )

    subs = list(captured["kwargs"].get("subagents") or [])
    names = {s["name"] for s in subs}
    assert "researcher" in names and "verifier" in names and "humanizer" in names


def test_orchestrator_is_llm_neutral_uses_build_executor():
    """Confirm we go through build_executor() — that's the LLM-neutrality contract."""
    from openmark.composer import orchestrator

    seen = {"calls": 0}

    def _spy():
        seen["calls"] += 1
        return _StubLLM()

    _, side = _capture_create_deep_agent_kwargs()
    with patch.object(orchestrator, "create_deep_agent", side_effect=side):
        with patch.object(orchestrator, "build_executor", side_effect=_spy):
            with patch.object(orchestrator, "_warm_up_tools"):
                orchestrator.build_composer_orchestrator()
    assert seen["calls"] == 1, "build_executor must be called exactly once"
