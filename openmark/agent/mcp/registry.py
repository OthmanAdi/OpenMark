"""
Declarative registry of MCP servers.

Adding a new MCP server = one entry below. Each entry declares:

    enabled_env    str    The .env flag that gates this server.
                          When the value is "0" / "false" / unset (and no
                          default_enabled=True), the server is skipped.
    scopes         set[Scope]  Which agents should receive these tools.
                          "orchestrator" = the top-level chat agent.
                          Per-sub-agent values: "researcher", "composer",
                          "humanizer", "polisher", "verifier",
                          "skill_author". Use the literal "all" sentinel
                          to mean every scope.
    name_prefix    str    A short tag prepended to tool names so two MCP
                          servers can't collide on names. Recommended:
                          short and lowercase (e.g. "trendradar").
    transport      "stdio" | "http"   Subprocess (stdio) or remote (http).
    command        str    Subprocess executable. Required for stdio.
    args           list[str]  Args to that subprocess.
    env_passthrough  list[str]  Names of OUR env vars whose values to
                          forward into the subprocess.
    url            str    Remote URL. Required for http transport.
    description    str    Free-text — explains what this MCP server does
                          and which sub-agent benefits.

The registry is intentionally a plain dict so it's diff-friendly + readable
in commit reviews. No YAML, no config files, no plugin loader.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal, TypedDict


# Scopes that determine which agent receives a server's tools.
Scope = Literal[
    "orchestrator",
    "researcher",
    "composer",
    "humanizer",
    "polisher",
    "verifier",
    "skill_author",
    "all",
]


class MCPServerSpec(TypedDict, total=False):
    enabled_env: str
    default_enabled: bool
    scopes: list[Scope]
    name_prefix: str
    transport: Literal["stdio", "http"]

    # stdio
    command: str
    args: list[str]
    env_passthrough: list[str]
    cwd: str | None

    # http
    url: str

    # docs
    description: str


# Where TrendRadar lives. Defaults to a sibling directory of OpenMark
# (e.g. Documents/TrendRadar next to Documents/OpenMark); can be overridden
# by setting TRENDRADAR_HOME in .env. Used as the cwd= for the stdio
# subprocess so TrendRadar finds its config/ + output/ folders.
#
# Path math from openmark/agent/mcp/registry.py:
#   parents[0] -> openmark/agent/mcp
#   parents[1] -> openmark/agent
#   parents[2] -> openmark
#   parents[3] -> <project root: Documents/OpenMark>
#   parents[4] -> Documents
_DEFAULT_TRENDRADAR_HOME = str(
    (Path(__file__).resolve().parents[4] / "TrendRadar").resolve()
)
TRENDRADAR_HOME = os.environ.get("TRENDRADAR_HOME", _DEFAULT_TRENDRADAR_HOME)


def _resolve_uv() -> str:
    """Locate the `uv` executable on PATH. Used to spawn TrendRadar in its
    own pinned virtualenv via `uv run`. Falls back to a bare 'uv' string so
    the subprocess error is informative if uv is missing."""
    for cand in ("uv.cmd", "uv"):
        path = shutil.which(cand)
        if path:
            return path
    return "uv"


SERVER_REGISTRY: dict[str, MCPServerSpec] = {
    # ── TrendRadar ──────────────────────────────────────────────────────
    # Hot-topic + RSS trend monitor. Exposes 26+ tools across data_query,
    # analytics, search, config_mgmt, system, storage_sync, article_reader,
    # notification. We hand it to the researcher sub-agent so retrieval
    # missions can pull "what's trending today on platform X" alongside
    # the existing Neo4j + web tools.
    "trendradar": {
        "enabled_env": "OPENMARK_MCP_TRENDRADAR",
        "default_enabled": False,
        "scopes": ["researcher"],
        "name_prefix": "trendradar",
        "transport": "stdio",
        # We launch via openmark.agent.mcp.trendradar_stdio_launcher (NOT
        # `python -m mcp_server.server`) because TrendRadar v6.7.0 prints a
        # banner to stdout before mcp.run() — that pollutes the JSON-RPC
        # stream the MCP client reads. The launcher monkey-patches print
        # to stderr so mcp.run()'s stdout stays clean.
        "command": _resolve_uv(),
        "args": [
            "run",
            "--directory",
            TRENDRADAR_HOME,
            "python",
            "-m",
            "openmark.agent.mcp.trendradar_stdio_launcher",
        ],
        "env_passthrough": [
            "AI_API_KEY",
            "AI_BASE_URL",
            "AI_MODEL",
            "TZ",
            "PYTHONPATH",
        ],
        "cwd": TRENDRADAR_HOME,
        "description": (
            "Trend monitor over Chinese hot platforms + RSS sources + AI "
            "analysis briefs. Researcher uses it for 'what's trending in "
            "<topic>' queries."
        ),
    },
}


# ── Helpers ─────────────────────────────────────────────────────────────


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_enabled(server_name: str) -> bool:
    """Whether `enabled_env` is truthy (or default_enabled fallback)."""
    spec = SERVER_REGISTRY.get(server_name)
    if not spec:
        return False
    return _env_truthy(spec["enabled_env"], default=bool(spec.get("default_enabled", False)))


def list_enabled_servers() -> list[str]:
    return [name for name in SERVER_REGISTRY if is_enabled(name)]


def servers_for_scope(scope: Scope) -> list[str]:
    """Servers whose scopes include `scope` (or contain 'all')."""
    out: list[str] = []
    for name, spec in SERVER_REGISTRY.items():
        if not is_enabled(name):
            continue
        scopes = set(spec.get("scopes", []))
        if scope in scopes or "all" in scopes:
            out.append(name)
    return out
