"""
Stdio launcher for TrendRadar's MCP server.

TrendRadar's `mcp_server.server.run_server()` (v6.7.0) writes a startup banner
+ tool list to stdout BEFORE calling `mcp.run(transport='stdio')`. The MCP
stdio protocol requires JSON-RPC ONLY on stdout — banners break the parser.

Fix: monkey-patch `builtins.print` to route to stderr at the very start of
this module. Then run TrendRadar's run_server() unchanged. mcp.run() does
NOT use print() under the hood (writes JSON-RPC directly via the streams
the SDK opens), so JSON-RPC still flows on real stdout.

This launcher is invoked by openmark.agent.mcp.registry as the stdio command:
    uv run --directory <TRENDRADAR_HOME> python -m \
        openmark.agent.mcp.trendradar_stdio_launcher

WHY THIS PATTERN: zero modification to the upstream TrendRadar repo. No
fork, no patch file to keep in sync. If TrendRadar ever fixes their
stdout pollution upstream, we delete this launcher.
"""

from __future__ import annotations

import builtins
import sys


# Redirect every print() to stderr BEFORE importing TrendRadar — so even
# import-time banner prints get caught.
_orig_print = builtins.print


def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    return _orig_print(*args, **kwargs)


builtins.print = _stderr_print


def main() -> None:
    """Launch TrendRadar's MCP server over stdio."""
    # Import AFTER the print patch so banner prints already route to stderr.
    from mcp_server.server import run_server

    run_server(transport="stdio")


if __name__ == "__main__":
    main()
