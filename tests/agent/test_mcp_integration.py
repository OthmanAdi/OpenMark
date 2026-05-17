"""
MCP integration tests — registry + client wrapper + scope wiring.

The full live spawn-and-load path is gated on OPENMARK_MCP_TRENDRADAR=1
(same convention as the live Foundry tests). Pure-unit tests run by default
and use no subprocess.
"""

from __future__ import annotations

import os

import pytest


# ── Pure-unit (no subprocess) ───────────────────────────────────────────────


def test_registry_lists_trendradar():
    from openmark.agent.mcp.registry import SERVER_REGISTRY

    assert "trendradar" in SERVER_REGISTRY
    spec = SERVER_REGISTRY["trendradar"]
    assert spec["transport"] == "stdio"
    assert "researcher" in spec.get("scopes", [])
    # Launcher path — we MUST use our wrapper, not bare mcp_server.server
    assert any("trendradar_stdio_launcher" in a for a in spec["args"]), (
        "stdio command must use the launcher that redirects banner prints to stderr"
    )


def test_trendradar_excludes_broken_get_current_config():
    """TrendRadar v6.7.0's get_current_config crashes with
    'Object of type Pattern is not JSON serializable' — must be denylisted."""
    from openmark.agent.mcp.registry import SERVER_REGISTRY
    exclude = SERVER_REGISTRY["trendradar"].get("exclude_tool_suffixes", [])
    assert "get_current_config" in exclude, (
        "get_current_config is upstream-broken; keep it on the denylist"
    )


def test_default_disabled(monkeypatch):
    """Without OPENMARK_MCP_TRENDRADAR, no servers should be enabled."""
    monkeypatch.delenv("OPENMARK_MCP_TRENDRADAR", raising=False)
    from openmark.agent.mcp import list_enabled_servers
    assert list_enabled_servers() == []


def test_enabled_via_env(monkeypatch):
    monkeypatch.setenv("OPENMARK_MCP_TRENDRADAR", "1")
    from openmark.agent.mcp import list_enabled_servers
    assert "trendradar" in list_enabled_servers()


@pytest.mark.parametrize("val,expected_enabled", [
    ("1", True),
    ("true", True),
    ("TRUE", True),
    ("yes", True),
    ("on", True),
    ("0", False),
    ("false", False),
    ("", False),
    ("anything-else", False),
])
def test_env_truthy_parsing(monkeypatch, val, expected_enabled):
    monkeypatch.setenv("OPENMARK_MCP_TRENDRADAR", val)
    from openmark.agent.mcp.registry import is_enabled
    assert is_enabled("trendradar") is expected_enabled


def test_load_tools_returns_empty_when_no_servers_for_scope(monkeypatch):
    monkeypatch.delenv("OPENMARK_MCP_TRENDRADAR", raising=False)
    from openmark.agent.mcp import load_tools_for
    from openmark.agent.mcp.client import reset_client_cache
    reset_client_cache()
    # No servers configured for orchestrator scope right now — must return []
    tools = load_tools_for("orchestrator")
    assert tools == []


def test_scope_routing_only_researcher(monkeypatch):
    """TrendRadar scope is ['researcher'] — composer/orchestrator must skip it."""
    monkeypatch.setenv("OPENMARK_MCP_TRENDRADAR", "1")
    from openmark.agent.mcp.registry import servers_for_scope
    assert "trendradar" in servers_for_scope("researcher")
    assert "trendradar" not in servers_for_scope("orchestrator")
    assert "trendradar" not in servers_for_scope("composer")
    assert "trendradar" not in servers_for_scope("polisher")


def test_subagent_graph_accepts_mcp_scope_kwarg():
    """make_subagent_graph signature must include mcp_scope (without it the
    sub-agents wouldn't be able to opt into MCP tool merging at all)."""
    import inspect
    from openmark.agent.subagents._common import make_subagent_graph

    sig = inspect.signature(make_subagent_graph)
    assert "mcp_scope" in sig.parameters
    assert sig.parameters["mcp_scope"].default is None


def test_researcher_module_declares_researcher_scope():
    """Researcher must wire mcp_scope='researcher' so TrendRadar tools merge in."""
    src = open(
        os.path.join(os.path.dirname(__file__), "..", "..",
                     "openmark", "agent", "subagents", "researcher.py"),
        encoding="utf-8",
    ).read()
    assert 'mcp_scope="researcher"' in src or "mcp_scope='researcher'" in src


# ── Live (opt-in) ──────────────────────────────────────────────────────────


_LIVE = os.environ.get("OPENMARK_MCP_TRENDRADAR") == "1"
pytestmark_live = pytest.mark.skipif(
    not _LIVE,
    reason="Set OPENMARK_MCP_TRENDRADAR=1 to enable live MCP spawn tests.",
)


@pytestmark_live
def test_live_trendradar_spawn_and_load_tools():
    """Actually spawn TrendRadar MCP server via stdio + load tools through
    the langchain-mcp-adapters client. Verifies the launcher script silences
    stdout banners so JSON-RPC parses cleanly.

    Requires:
      - TrendRadar installed at TRENDRADAR_HOME (default sibling of repo)
      - `uv` on PATH
    """
    from openmark.agent.mcp import load_tools_for
    from openmark.agent.mcp.client import reset_client_cache

    reset_client_cache()
    tools = load_tools_for("researcher")
    assert len(tools) >= 20, f"expected 20+ TrendRadar MCP tools, got {len(tools)}"
    names = {getattr(t, "name", "") for t in tools}
    # Spot-check a few high-value ones (after the trendradar_ prefix)
    must_have_suffixes = {
        "get_latest_news",
        "get_trending_topics",
        "search_news",
        "analyze_sentiment",
    }
    matched = {s for s in must_have_suffixes if any(s in n for n in names)}
    assert matched == must_have_suffixes, f"missing: {must_have_suffixes - matched}"


@pytestmark_live
def test_live_researcher_graph_contains_trendradar_tools():
    """The compiled researcher graph must include trendradar_* tools when
    OPENMARK_MCP_TRENDRADAR=1, alongside its existing 21 retrieval + load_skill."""
    import openmark.agent.tools as _t
    _t.warm_up = lambda: None
    from openmark.agent.mcp.client import reset_client_cache
    import openmark.agent.subagents.researcher as r
    reset_client_cache()
    # Force a fresh graph build
    r._RESEARCHER_GRAPH = None
    g = r._get_graph()
    inner = g.nodes["tools"].bound
    tools = getattr(inner, "tools_by_name", None) or getattr(inner, "tools", None)
    names = (
        sorted(tools.keys()) if isinstance(tools, dict)
        else sorted(getattr(t, "name", repr(t)) for t in tools)
    )
    # Existing tools still present
    assert "search_semantic" in names
    assert "load_skill" in names
    # AND TrendRadar tools merged in via mcp_scope='researcher'
    trendradar_count = sum(1 for n in names if n.startswith("trendradar_"))
    assert trendradar_count >= 20, (
        f"researcher must have 20+ trendradar_* tools, got {trendradar_count}"
    )
