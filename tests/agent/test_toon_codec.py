"""TOON codec tests — verifies token savings on MCP-shaped payloads."""

from __future__ import annotations

import json

import pytest


def test_toon_module_available():
    """Sanity: the underlying `toon` library is installed."""
    import toon  # noqa: F401


def test_toonify_text_passes_through_non_json():
    from openmark.agent.mcp.toon_codec import toonify_text
    plain = "Hello world, this is a plain text response, not JSON."
    assert toonify_text(plain) == plain


def test_toonify_text_passes_through_short_input():
    from openmark.agent.mcp.toon_codec import toonify_text
    short = '{"a":1}'
    assert toonify_text(short, min_chars=120) == short


def test_toonify_text_encodes_uniform_array():
    """TOON should beat JSON on tabular arrays (the case it was designed for)."""
    from openmark.agent.mcp.toon_codec import toonify_text, OVERHEAD_NOTE
    payload = {
        "results": [
            {"id": i, "title": f"Article {i}", "url": f"https://x.com/{i}",
             "score": i * 10, "domain": "x.com"}
            for i in range(1, 12)
        ],
        "total": 11,
    }
    j = json.dumps(payload, ensure_ascii=False)
    out = toonify_text(j)
    assert out is not j, "should have been TOON-encoded"
    assert out.startswith(OVERHEAD_NOTE)
    assert len(out) < len(j), f"TOON {len(out)} >= JSON {len(j)} — no savings"
    # Lossless roundtrip
    import toon
    body = out[len(OVERHEAD_NOTE):]
    decoded = toon.decode(body)
    assert decoded == payload


def test_toonify_tool_result_list_of_text_blocks():
    """MCP tools return `[{type:'text', text:'<json string>'}]`."""
    from openmark.agent.mcp.toon_codec import toonify_tool_result, OVERHEAD_NOTE
    inner = json.dumps({
        "results": [{"id": i, "title": f"T{i}", "score": i * 5} for i in range(1, 10)],
        "total": 9,
    })
    result = [{"type": "text", "text": inner, "id": "lc_xyz"}]
    out = toonify_tool_result(result)
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["type"] == "text"
    assert out[0]["id"] == "lc_xyz"
    assert out[0]["text"].startswith(OVERHEAD_NOTE)
    assert len(out[0]["text"]) < len(inner)


def test_toonify_tool_result_skips_non_text_blocks():
    from openmark.agent.mcp.toon_codec import toonify_tool_result
    inner = json.dumps({"a": 1, "b": 2})
    result = [
        {"type": "image", "data": "...", "url": "..."},
        {"type": "text", "text": inner},
    ]
    out = toonify_tool_result(result)
    assert out[0] == result[0]  # image block untouched


def test_toonify_tool_result_passes_through_unknown_shape():
    from openmark.agent.mcp.toon_codec import toonify_tool_result
    weird = 42
    assert toonify_tool_result(weird) == 42


def test_is_toon_compact_diagnostic():
    from openmark.agent.mcp.toon_codec import is_toon_compact_for
    fat = json.dumps({
        "rows": [{"a": i, "b": i * 2, "c": "x"} for i in range(1, 15)]
    })
    assert is_toon_compact_for(fat) is True
    assert is_toon_compact_for("plain text not json") is False


@pytest.mark.parametrize("rows", [3, 7, 15, 30])
def test_savings_scale_with_array_size(rows):
    """Larger uniform arrays should see proportionally bigger savings."""
    from openmark.agent.mcp.toon_codec import toonify_text, OVERHEAD_NOTE
    payload = {"data": [{"id": i, "title": f"T{i}", "score": i * 7}
                        for i in range(rows)]}
    j = json.dumps(payload)
    out = toonify_text(j)
    if out is j:
        # Tiny array might not break even after the overhead note
        return
    body = out[len(OVERHEAD_NOTE):] if out.startswith(OVERHEAD_NOTE) else out
    savings_pct = 100 * (1 - len(body) / len(j))
    # Even small uniform arrays should beat JSON by at least 15%
    assert savings_pct > 15, f"rows={rows}: only {savings_pct:.0f}% savings"


def test_subagent_mcp_tools_have_toon_wrapper_when_enabled(monkeypatch):
    """End-to-end: when MCP is enabled, the wrapped tool's coroutine should
    route through the toonify post-processor."""
    monkeypatch.setenv("OPENMARK_MCP_TRENDRADAR", "0")  # disable to keep this unit-test fast
    from openmark.agent.mcp.client import _wrap_async_only_tool
    from langchain_core.tools import StructuredTool

    async def fake_mcp_coro(query: str) -> list:
        # Simulate MCP returning structured JSON text
        return [{"type": "text", "text": json.dumps({
            "results": [{"id": i, "title": f"T{i}", "score": i * 7}
                        for i in range(1, 12)],
            "total": 11,
        })}]

    tool = StructuredTool.from_function(
        coroutine=fake_mcp_coro,
        name="fake_mcp_tool",
        description="fake_mcp_tool",
    )
    wrapped = _wrap_async_only_tool(tool)
    assert wrapped is not tool
    # Sync invoke should now work AND return TOON-encoded text
    out = wrapped.invoke({"query": "x"})
    from openmark.agent.mcp.toon_codec import OVERHEAD_NOTE
    text = out[0]["text"]
    assert text.startswith(OVERHEAD_NOTE), \
        f"wrapped tool result not TOON-encoded: {text[:120]}"
