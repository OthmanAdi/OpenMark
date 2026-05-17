"""
MCP (Model Context Protocol) integration for the OpenMark v3 stack.

External MCP servers expose tools to the orchestrator + sub-agents through
the same `BaseTool` surface every other tool uses. There is no separate
"MCP middleware" — `langchain-mcp-adapters` converts MCP server tools into
LangChain `BaseTool` instances at boot, and they merge into the agent's
tool list like anything else.

Public surface:
    SERVER_REGISTRY            — dict of MCP server connection configs
    load_tools_for(scope)      — sync wrapper, returns list[BaseTool] for a scope
                                 (e.g. "researcher" -> [trendradar tools])
    list_enabled_servers()     — quick diagnostic
"""

from openmark.agent.mcp.registry import (
    SERVER_REGISTRY,
    Scope,
    is_enabled,
    list_enabled_servers,
)
from openmark.agent.mcp.client import (
    load_tools_for,
    reset_client_cache,
)

__all__ = [
    "SERVER_REGISTRY",
    "Scope",
    "is_enabled",
    "list_enabled_servers",
    "load_tools_for",
    "reset_client_cache",
]
