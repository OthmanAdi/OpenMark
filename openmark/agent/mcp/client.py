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
import threading
from pathlib import Path
from typing import Any


# Absolute path to the OpenMark project root, so subprocess children can
# `import openmark...` even when they run in a different venv (TrendRadar's,
# for example). We inject this into PYTHONPATH for every stdio connection.
_OPENMARK_ROOT = str(Path(__file__).resolve().parents[3])


# ── Persistent background asyncio loop ──────────────────────────────────────
#
# MCP servers are long-lived async resources (stdio subprocess + reader/writer
# streams). If we spin up a fresh `asyncio.run()` per call, every tool
# invocation would re-spawn the server. With TrendRadar that's a 4s tax per
# call — fatal for an agent that wants to chain 5+ tool calls.
#
# Solution: a dedicated background thread runs ONE event loop for the process
# lifetime. Tool loading + per-call invocation both submit coroutines to it
# via `asyncio.run_coroutine_threadsafe`. The MCP stdio subprocess stays open
# on that loop's transport, reused across calls.
#
# Crucial side effect: our agent runs sync (`agent.stream(stream_mode='updates')`)
# but the MCP tool the adapter returns has only `coroutine=` set (no `func=`).
# `StructuredTool._run()` then raises NotImplementedError. The bridge below
# rebuilds each tool with `func=` set to a sync shim that schedules the
# coroutine on the background loop and blocks for the result.

_BG_LOOP: asyncio.AbstractEventLoop | None = None
_BG_THREAD: threading.Thread | None = None
_BG_LOOP_LOCK = threading.Lock()


def _ensure_bg_loop() -> asyncio.AbstractEventLoop:
    """Start the background asyncio thread on first call. Idempotent."""
    global _BG_LOOP, _BG_THREAD
    with _BG_LOOP_LOCK:
        if _BG_LOOP is not None and _BG_LOOP.is_running():
            return _BG_LOOP
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            daemon=True,
            name="openmark-mcp-bg-loop",
        )
        thread.start()
        _BG_LOOP = loop
        _BG_THREAD = thread
        log.info("[mcp] background asyncio loop started")
        return loop


def _run_coro_sync(coro):
    """Submit a coroutine to the background loop and block for its result."""
    loop = _ensure_bg_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()

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


def _wrap_async_only_tool(tool: Any) -> Any:
    """
    Two transformations on each MCP-adapted tool:

    1. Sync invocation bridge.
       langchain-mcp-adapters returns StructuredTool with only `coroutine=`
       set, so `.invoke()` raises `NotImplementedError: StructuredTool does
       not support sync invocation.` under our sync agent. We install a
       sync `func` that schedules the coroutine on the persistent background
       asyncio loop.

    2. TOON post-processing.
       Wrap the underlying coroutine so its return value is run through
       openmark.agent.mcp.toon_codec.toonify_tool_result before it lands in
       the agent's context. MCP tool results are usually nested JSON; TOON
       saves 30-60% on uniform/tabular shapes (TrendRadar returns a lot of
       these). Lossless either way — if the text isn't JSON or TOON isn't
       materially smaller, the original text passes through untouched.

    Implementation note: `tool.model_copy(update={'coroutine': ...})` does NOT
    reliably swap the `coroutine` field on StructuredTool under Pydantic v2
    (the copy machinery treats it as a default-having field and discards the
    update). We use object.__setattr__ to bypass the model setter — safe
    because Pydantic models are plain Python classes underneath.

    Returns the original instance when the tool is already sync-callable AND
    has no async coroutine to wrap, so this is safe to call on every tool.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        return tool
    if not isinstance(tool, StructuredTool):
        return tool
    coroutine = getattr(tool, "coroutine", None)
    if coroutine is None:
        return tool

    from openmark.agent.mcp.toon_codec import toonify_tool_result

    async def _toon_coro(*args, **kwargs):
        raw = await coroutine(*args, **kwargs)
        try:
            return toonify_tool_result(raw)
        except Exception as e:  # noqa: BLE001
            log.debug(f"[toon] post-process skipped: {type(e).__name__}: {e}")
            return raw

    def _sync_run(*args, **kwargs):
        return _run_coro_sync(_toon_coro(*args, **kwargs))

    _sync_run.__name__ = getattr(coroutine, "__name__", "mcp_sync_run")

    wrapped = tool.model_copy()
    object.__setattr__(wrapped, "coroutine", _toon_coro)
    if getattr(wrapped, "func", None) is None:
        object.__setattr__(wrapped, "func", _sync_run)
    return wrapped


def _gather_tools_sync(server_names: list[str]) -> list[Any]:
    """Load tools from every named server on the background asyncio loop,
    drop any names blocked by the per-server `exclude_tool_suffixes` denylist,
    return survivors with sync `func=` wrappers installed.

    Empty list on failure — never raises."""
    client = _get_client()
    if client is None or not server_names:
        return []

    async def _async_load() -> dict[str, list[Any]]:
        # Load per-server so we can match denylist suffixes back to the
        # owning spec (the tool name carries the server prefix, but the
        # exclude list lives on the SPEC, not the prefixed name).
        results: dict[str, list[Any]] = {}
        async def _load_one(name: str):
            results[name] = await client.get_tools(server_name=name)
        await asyncio.gather(*[_load_one(n) for n in server_names])
        return results

    try:
        per_server = _run_coro_sync(_async_load())
    except Exception as e:
        for name in server_names:
            _LOAD_FAILED[name] = f"{type(e).__name__}: {e}"
        log.warning(f"[mcp] tool load failed for {server_names}: {e!r}")
        return []

    flat: list[Any] = []
    dropped: list[str] = []
    for server_name, tools in per_server.items():
        spec = SERVER_REGISTRY.get(server_name, {})
        exclude = set(spec.get("exclude_tool_suffixes", []) or [])
        prefix = spec.get("name_prefix") or server_name
        for tool in tools:
            tool_name = getattr(tool, "name", "")
            # langchain-mcp-adapters with tool_name_prefix=True names tools
            # "<server>_<orig>"; the denylist uses the bare original suffix.
            unprefixed = tool_name
            for p in (f"{prefix}_", f"{server_name}_"):
                if unprefixed.startswith(p):
                    unprefixed = unprefixed[len(p):]
                    break
            if unprefixed in exclude:
                dropped.append(tool_name)
                continue
            flat.append(tool)

    if dropped:
        log.info(f"[mcp] dropped {len(dropped)} tool(s) via exclude_tool_suffixes: {dropped}")

    wrapped = [_wrap_async_only_tool(t) for t in flat]
    log.info(
        f"[mcp] wrapped {len(wrapped)} tool(s) with sync->async bridge "
        f"(persistent background loop)"
    )
    return wrapped


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
