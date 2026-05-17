"""
Singleton MCP client that loads tools from every enabled server.

Why singleton: each MCP server is a long-lived subprocess (stdio) or
TCP connection (http). Opening a fresh client per agent build would
re-spawn TrendRadar (~3-5s startup), leak file descriptors, and burn
RAM. One client, one cached tool list keyed by scope, lifetime = parent
process.

The langchain-mcp-adapters API is async (anyio under the hood). We
bridge to sync at boot via `asyncio.run()` — `build_agent()` is itself
sync and called once per process startup, so no event-loop nesting risk.

Public:
    load_tools_for(scope)    sync, returns list[BaseTool] for the given scope.
                             Empty list when no servers configured for the scope.
    reset_client_cache()     test helper; clears singleton + cached tools.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any


# Absolute path to the OpenMark project root, so subprocess children can
# `import openmark...` even when they run in a different venv (TrendRadar's,
# for example). We inject this into PYTHONPATH for every stdio connection.
_OPENMARK_ROOT = str(Path(__file__).resolve().parents[3])

from openmark.agent.mcp.registry import (
    SERVER_REGISTRY,
    Scope,
    is_enabled,
    servers_for_scope,
)


log = logging.getLogger("openmark.agent.mcp")


# Module-level singleton state.
_CLIENT: Any | None = None
_TOOLS_BY_SCOPE: dict[Scope, list[Any]] = {}
_LOAD_FAILED: dict[str, str] = {}


def _connection_for(name: str) -> dict[str, Any] | None:
    """
    Build a langchain-mcp-adapters connection dict for `name`. Returns None
    when the server is not enabled or the config is incomplete (so the
    caller can skip it without raising).
    """
    spec = SERVER_REGISTRY.get(name)
    if not spec or not is_enabled(name):
        return None

    transport = spec.get("transport", "stdio")

    if transport == "stdio":
        command = spec.get("command")
        args = list(spec.get("args") or [])
        if not command:
            log.warning(f"[mcp] {name}: stdio transport missing 'command'; skipping")
            return None
        # IMPORTANT: stdio servers run as a subprocess with a CLEAN env (the
        # MCP SDK does NOT inherit os.environ by default). We must explicitly
        # forward anything the child needs.
        env: dict[str, str] = {}
        # 1. Ensure the child can `import openmark...` — needed for the
        #    trendradar_stdio_launcher and any future launcher we ship.
        existing_pp = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{_OPENMARK_ROOT}{os.pathsep}{existing_pp}"
            if existing_pp else _OPENMARK_ROOT
        )
        # 2. Standard Windows env vars some subprocesses (npm, uv) need to
        #    locate their data and executables. Without these uv may fail to
        #    find a Python interpreter.
        for var in (
            "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "USERPROFILE", "APPDATA",
            "LOCALAPPDATA", "TEMP", "TMP", "HOMEDRIVE", "HOMEPATH",
            "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA",
            "PATHEXT", "COMSPEC", "WINDIR",
        ):
            val = os.environ.get(var)
            if val is not None:
                env[var] = val
        # 3. Anything the spec asked us to pass through (API keys, model
        #    overrides, TZ — whatever the user marked in env_passthrough).
        for var in spec.get("env_passthrough", []) or []:
            val = os.environ.get(var)
            if val is not None:
                env[var] = val
        conn: dict[str, Any] = {
            "transport": "stdio",
            "command": command,
            "args": args,
        }
        if env:
            conn["env"] = env
        cwd = spec.get("cwd")
        if cwd:
            conn["cwd"] = cwd
        return conn

    if transport == "http":
        url = spec.get("url")
        if not url:
            log.warning(f"[mcp] {name}: http transport missing 'url'; skipping")
            return None
        # langchain-mcp-adapters uses 'streamable_http' as the transport
        # literal for HTTP server connections.
        return {"transport": "streamable_http", "url": url}

    log.warning(f"[mcp] {name}: unknown transport {transport!r}; skipping")
    return None


def _get_client():
    """Lazily build the MultiServerMCPClient with every enabled server."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        log.warning(f"[mcp] langchain-mcp-adapters not installed: {e}")
        _LOAD_FAILED["__import__"] = str(e)
        return None

    connections: dict[str, dict[str, Any]] = {}
    for name in SERVER_REGISTRY:
        conn = _connection_for(name)
        if conn is None:
            continue
        connections[name] = conn

    if not connections:
        log.info("[mcp] no MCP servers enabled; client will be a no-op")

    # tool_name_prefix=True so the server name is prepended to every tool
    # name. Prevents collisions when two servers both expose, say, a
    # `search` tool.
    _CLIENT = MultiServerMCPClient(connections=connections, tool_name_prefix=True)
    log.info(
        f"[mcp] built MultiServerMCPClient with {len(connections)} server(s): "
        f"{sorted(connections)}"
    )
    return _CLIENT


def _gather_tools_sync(server_names: list[str]) -> list[Any]:
    """Sync wrapper around the async get_tools call. Returns empty list on
    any error so an MCP failure never crashes agent boot."""
    client = _get_client()
    if client is None or not server_names:
        return []

    async def _async_load() -> list[Any]:
        # MultiServerMCPClient.get_tools(server_name=...) loads from one
        # server at a time; we concatenate. Concurrent gather keeps boot fast.
        gathers = [
            client.get_tools(server_name=name) for name in server_names
        ]
        results: list[list[Any]] = await asyncio.gather(*gathers, return_exceptions=False)
        flat: list[Any] = []
        for r in results:
            flat.extend(r)
        return flat

    try:
        # Use asyncio.run() — build_agent runs in sync context once at boot.
        return asyncio.run(_async_load())
    except RuntimeError as e:
        # Most common: "asyncio.run() cannot be called from a running event loop".
        # Fall back to nest_asyncio-style execution via a fresh loop.
        log.info(f"[mcp] asyncio.run failed ({e!r}); retrying with new loop")
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_async_load())
        finally:
            loop.close()
    except Exception as e:
        for name in server_names:
            _LOAD_FAILED[name] = f"{type(e).__name__}: {e}"
        log.warning(f"[mcp] tool load failed for {server_names}: {e!r}")
        return []


def load_tools_for(scope: Scope) -> list[Any]:
    """
    Return MCP tools for the given scope. Cached per-scope after first call.

    Failure modes (return empty list, never raise):
      - langchain-mcp-adapters not installed
      - no servers map to `scope`
      - subprocess for any enabled server fails to start
      - get_tools() raises mid-load

    The orchestrator + sub-agents call this at build time and merge the
    returned tools into their static tool lists.
    """
    if scope in _TOOLS_BY_SCOPE:
        return _TOOLS_BY_SCOPE[scope]

    server_names = servers_for_scope(scope)
    if not server_names:
        _TOOLS_BY_SCOPE[scope] = []
        return []

    tools = _gather_tools_sync(server_names)
    _TOOLS_BY_SCOPE[scope] = tools
    log.info(
        f"[mcp] loaded {len(tools)} tools for scope={scope!r} "
        f"from servers={server_names}"
    )
    return tools


def reset_client_cache() -> None:
    """Test helper. Clears singleton + per-scope cache + failure log."""
    global _CLIENT
    _CLIENT = None
    _TOOLS_BY_SCOPE.clear()
    _LOAD_FAILED.clear()
