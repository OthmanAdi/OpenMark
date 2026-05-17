"""
TOON (Token-Oriented Object Notation) codec for MCP tool results.

Why: MCP tools (especially TrendRadar's analytics + search) return nested
JSON that an LLM has to read verbatim out of its context. JSON's quote /
brace / repeated-key overhead is dead weight. TOON is a lossless alternative
designed for LLM prompts — YAML-style indentation for nested objects, CSV-
style tabular layout for uniform arrays. Token savings: 30-60% on uniform
data, 10-20% on mixed structure.

How: post-process the result returned by each MCP tool. The MCP adapter
emits content as either a string of JSON or a list of `{type: "text",
text: "..."}` blocks. We try to parse the text as JSON; if it parses and
TOON encodes smaller, we swap. If it doesn't parse or TOON isn't smaller,
we pass through unchanged.

Public:
    toonify_tool_result(result, *, min_chars=120) -> same shape, TOON-encoded text
    toonify_text(text) -> str (returns text unchanged if not worth it)
    is_toon_compact_for(text) -> bool  (introspection only)
    OVERHEAD_NOTE: short string we prepend so the LLM knows what TOON looks like

The OVERHEAD_NOTE goes ONCE per tool result. Compact (~40 chars). Pays for
itself the first time the tool returns more than ~100 bytes.
"""

from __future__ import annotations

import json
import logging
from typing import Any


log = logging.getLogger("openmark.agent.mcp.toon")


# Prepended to TOON-encoded tool outputs so the model can identify the format
# even if it hasn't seen TOON before. Designed to be short (cheap) and
# self-explanatory.
OVERHEAD_NOTE = (
    "<!-- TOON format: YAML-style nesting, tabular arrays. Same data as JSON, "
    "~40% fewer tokens. -->\n"
)


def toonify_text(text: str, *, min_chars: int = 120) -> str:
    """
    Try to parse `text` as JSON and re-encode as TOON. Returns the original
    text if (a) it doesn't parse, (b) it's too short to be worth it, or
    (c) TOON encoding isn't actually smaller.

    Failure modes return the input unchanged — never raise.
    """
    if not text or len(text) < min_chars:
        return text
    try:
        import toon
    except ImportError:
        return text

    stripped = text.lstrip()
    if not stripped or stripped[0] not in "{[":
        return text  # not JSON

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text

    try:
        encoded = toon.encode(parsed)
    except Exception as e:  # noqa: BLE001
        log.debug(f"[toon] encode failed ({type(e).__name__}); passing through")
        return text

    # Only swap if TOON is materially smaller (account for OVERHEAD_NOTE)
    with_note = OVERHEAD_NOTE + encoded
    if len(with_note) >= len(text):
        return text
    return with_note


def toonify_tool_result(result: Any, *, min_chars: int = 120) -> Any:
    """
    Walk a typical MCP tool return value and TOON-ify any JSON-shaped text
    blocks inside it.

    Common shapes handled:
        list[dict] where each dict has {type: "text", text: "..."}
        plain str (already serialized)
        anything else: returned unchanged

    Returns a NEW list/dict where applicable; never mutates the input.
    """
    if isinstance(result, list):
        out: list[Any] = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                original = block.get("text", "")
                if isinstance(original, str):
                    new_text = toonify_text(original, min_chars=min_chars)
                    if new_text is not original:
                        out.append({**block, "text": new_text})
                        continue
            out.append(block)
        return out

    if isinstance(result, str):
        return toonify_text(result, min_chars=min_chars)

    return result


def is_toon_compact_for(text: str) -> bool:
    """Diagnostic: whether `text` would benefit from TOON encoding."""
    if not text or len(text) < 120:
        return False
    new = toonify_text(text, min_chars=120)
    return new is not text and len(new) < len(text)
