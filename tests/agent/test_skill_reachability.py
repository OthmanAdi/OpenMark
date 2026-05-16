"""
Skill reachability tests — every sub-agent gets `load_skill` so dropping a new
SKILL.md and restarting actually reaches the sub-agent's runtime.

The bug this guards against: prior to this fix, sub-agent prompts hardcoded
'Call load_skill('X') on your first turn' but the sub-agents had ZERO load_skill
tool registered. Composers ran with tools=[]; the researcher had 21 retrieval
tools but not load_skill. Models improvised. Recipe fidelity was incidental.

These tests pin down the contract:
  1. Every sub-agent ships `load_skill` in its tool surface.
  2. OpenMarkSkillMiddleware is in every sub-agent's middleware stack so the
     skill catalogue appears in the system prompt.
  3. Scan + cache invalidation works — drop a SKILL.md, reload, count goes up.
  4. The orchestrator already has load_skill (regression guard).
"""

from __future__ import annotations

import os
import shutil

import pytest


# Module-level patch so warm_up doesn't load the pplx model in tests
@pytest.fixture(autouse=True)
def _no_warmup(monkeypatch):
    import openmark.agent.tools as _t
    monkeypatch.setattr(_t, "warm_up", lambda: None)


def _tools_of(graph) -> list[str]:
    """Pull the tool names from a compiled sub-agent graph."""
    tn = graph.nodes["tools"]
    inner = tn.bound
    tools = getattr(inner, "tools_by_name", None) or getattr(inner, "tools", None)
    if isinstance(tools, dict):
        return sorted(tools.keys())
    if isinstance(tools, list):
        return sorted(getattr(t, "name", repr(t)) for t in tools)
    return []


SUBAGENTS = [
    ("researcher",         "openmark.agent.subagents.researcher"),
    ("composer-linkedin",  "openmark.agent.subagents.composer_linkedin"),
    ("composer-essay",     "openmark.agent.subagents.composer_essay"),
    ("composer-roundup",   "openmark.agent.subagents.composer_roundup"),
    ("composer-comparison", "openmark.agent.subagents.composer_comparison"),
    ("composer-analytical", "openmark.agent.subagents.composer_analytical"),
    ("humanizer",          "openmark.agent.subagents.humanizer"),
    ("polisher",           "openmark.agent.subagents.polisher"),
    ("verifier",           "openmark.agent.subagents.verifier"),
    ("skill-author",       "openmark.agent.subagents.skill_author"),
]


@pytest.mark.parametrize("name,mod_path", SUBAGENTS)
def test_every_subagent_has_load_skill_tool(name, mod_path):
    """The fix the user demanded: every sub-agent gets load_skill."""
    mod = __import__(mod_path, fromlist=["_get_graph"])
    graph = mod._get_graph()
    tools = _tools_of(graph)
    assert "load_skill" in tools, (
        f"sub-agent '{name}' is missing load_skill — its prompt instructs the "
        f"model to call load_skill but the tool is not registered. Tools: {tools}"
    )


def test_researcher_keeps_all_retrieval_tools():
    """Regression: the fix must not strip researcher's 21 retrieval tools."""
    import openmark.agent.subagents.researcher as r
    tools = _tools_of(r._get_graph())
    must_have = {
        "search_semantic", "search_by_category", "search_by_community",
        "find_by_tag", "explore_tag_cluster", "graph_expand",
        "find_by_domain", "find_by_source", "search_linkedin", "search_youtube",
        "find_recent", "search_by_date_range", "get_bookmark_full",
        "get_stats", "run_cypher", "web_search", "web_fetch", "web_extract",
        "web_crawl", "github_repo_intel", "reddit_search",
    }
    missing = must_have - set(tools)
    assert not missing, f"researcher lost retrieval tools: {missing}"
    # Plus load_skill from the fix
    assert "load_skill" in tools


def test_skill_author_keeps_write_skill():
    """Regression: skill-author still has write_skill alongside load_skill."""
    import openmark.agent.subagents.skill_author as sa
    tools = _tools_of(sa._get_graph())
    assert "write_skill" in tools
    assert "load_skill" in tools


def test_composers_only_have_load_skill():
    """Composer sub-agents have no retrieval tools (compose-only contract).
    Adding load_skill is the entire intended tool surface."""
    for mod_path in (
        "openmark.agent.subagents.composer_linkedin",
        "openmark.agent.subagents.composer_essay",
        "openmark.agent.subagents.composer_roundup",
        "openmark.agent.subagents.composer_comparison",
        "openmark.agent.subagents.composer_analytical",
    ):
        mod = __import__(mod_path, fromlist=["_get_graph"])
        tools = _tools_of(mod._get_graph())
        assert tools == ["load_skill"], (
            f"{mod_path} expected exactly ['load_skill'], got {tools}"
        )


def test_humanizer_polisher_verifier_only_have_load_skill():
    for mod_path in (
        "openmark.agent.subagents.humanizer",
        "openmark.agent.subagents.polisher",
        "openmark.agent.subagents.verifier",
    ):
        mod = __import__(mod_path, fromlist=["_get_graph"])
        tools = _tools_of(mod._get_graph())
        assert tools == ["load_skill"], (
            f"{mod_path} expected exactly ['load_skill'], got {tools}"
        )


def test_orchestrator_still_has_load_skill():
    """Regression: don't break the orchestrator path while wiring sub-agents."""
    import openmark.agent.graph as graph_mod
    captured = {}
    from langchain.agents import create_agent as _orig
    def _spy(*args, **kw):
        captured["kwargs"] = kw
        class _Stub:
            def invoke(self, *a, **k): pass
            def stream(self, *a, **k): pass
            def get_state(self, *a, **k): pass
        return _Stub()
    real = graph_mod.create_agent
    try:
        graph_mod.create_agent = _spy
        graph_mod.build_agent()
    finally:
        graph_mod.create_agent = real
    mw_list = captured["kwargs"]["middleware"]
    from openmark.agent.middleware import OpenMarkSkillMiddleware
    has_skill_mw = any(isinstance(m, OpenMarkSkillMiddleware) for m in mw_list)
    assert has_skill_mw, "OpenMarkSkillMiddleware missing from orchestrator stack"


def test_dropping_new_skill_md_is_discovered_after_reload():
    """End-to-end: drop a file under .claude/skills/openmark-<name>/, call
    reload_skills(), and the new short_name is in list_skills()."""
    from openmark.agent.skills import (
        SKILLS_DIR, list_skills, reload_skills,
    )
    test_name = "openmark-test-skill-reachability-fix"
    test_dir = os.path.join(SKILLS_DIR, test_name)
    test_md = os.path.join(test_dir, "SKILL.md")
    os.makedirs(test_dir, exist_ok=True)
    try:
        with open(test_md, "w", encoding="utf-8") as f:
            f.write(
                "---\n"
                f"name: {test_name}\n"
                "description: smoke test that fresh skill is discovered\n"
                "metadata:\n"
                "  type: test\n"
                "---\n\n"
                "Test body.\n"
            )
        reload_skills()
        names = {s["short_name"] for s in list_skills()}
        assert "test-skill-reachability-fix" in names, (
            f"new SKILL.md not discovered. Available: {sorted(names)}"
        )
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        reload_skills()


def test_subagent_skill_middleware_attached():
    """Verify the OpenMarkSkillMiddleware instance is in each sub-agent's
    middleware stack. The middleware injects the skill catalogue addendum
    into the sub-agent's system prompt every turn — keeps the sub-agent
    aware of skills added since boot (after reload_skills)."""
    from openmark.agent.middleware import OpenMarkSkillMiddleware
    from langchain.agents import create_agent as _orig_create

    captured = {}
    def _spy(*args, **kw):
        captured["kwargs"] = kw
        class _Stub: pass
        return _Stub()

    # We re-call make_subagent_graph directly to capture its kwargs
    import openmark.agent.subagents._common as common
    real = common.create_agent
    try:
        common.create_agent = _spy
        from openmark.agent.llms import build_polisher
        common.make_subagent_graph(
            model=build_polisher(),
            tools=[],
            system_prompt="test",
        )
    finally:
        common.create_agent = real

    mw_list = captured["kwargs"]["middleware"]
    has_skill_mw = any(isinstance(m, OpenMarkSkillMiddleware) for m in mw_list)
    assert has_skill_mw, "OpenMarkSkillMiddleware missing from sub-agent stack"


def test_include_skills_opt_out():
    """A sub-agent can opt out via include_skills=False (escape hatch)."""
    import openmark.agent.subagents._common as common
    captured = {}
    def _spy(*args, **kw):
        captured["kwargs"] = kw
        class _Stub: pass
        return _Stub()
    real = common.create_agent
    try:
        common.create_agent = _spy
        from openmark.agent.llms import build_polisher
        common.make_subagent_graph(
            model=build_polisher(),
            tools=[],
            system_prompt="test",
            include_skills=False,
        )
    finally:
        common.create_agent = real

    tools = captured["kwargs"]["tools"]
    tool_names = [getattr(t, "name", repr(t)) for t in tools]
    assert "load_skill" not in tool_names
    from openmark.agent.middleware import OpenMarkSkillMiddleware
    mw_list = captured["kwargs"]["middleware"]
    assert not any(isinstance(m, OpenMarkSkillMiddleware) for m in mw_list)
