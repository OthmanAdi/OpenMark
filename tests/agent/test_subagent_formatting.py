"""
Sub-agent → orchestrator output formatter tests.

Critical: when a sub-agent's inner model refuses ("I'm sorry, but I cannot
assist..."), the orchestrator MUST still receive the raw tool results so it
can synthesize from primary data. These tests guard that contract.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from openmark.agent.subagents._common import (
    _looks_like_refusal,
    _compact_tool_messages,
    format_for_orchestrator,
)


def test_refusal_detection_matches_canonical_phrasings():
    for s in (
        "I'm sorry, but I cannot assist with that request.",
        "i cannot assist with that",
        "I'm unable to help with that",
        "Sorry, but I cannot continue",
    ):
        assert _looks_like_refusal(s), f"missed refusal: {s!r}"


def test_refusal_detection_no_false_positives():
    for s in (
        "Here are 5 niches in your bookmarks.",
        "I found 12 relevant results.",
        "No matches in OpenMark; falling back to web.",
    ):
        assert not _looks_like_refusal(s), f"false positive: {s!r}"


def test_compact_tool_messages_keeps_last_k():
    msgs = [
        ToolMessage(content=f"result {i}", tool_call_id=str(i), name=f"tool_{i}")
        for i in range(10)
    ]
    pairs = _compact_tool_messages(msgs, keep=3, per_msg_cap=100)
    assert len(pairs) == 3
    assert [p[0] for p in pairs] == ["tool_7", "tool_8", "tool_9"]


def test_compact_tool_messages_caps_long_content():
    long_content = "x" * 5000
    msgs = [ToolMessage(content=long_content, tool_call_id="1", name="search")]
    pairs = _compact_tool_messages(msgs, keep=1, per_msg_cap=200)
    assert len(pairs) == 1
    name, content = pairs[0]
    assert name == "search"
    assert len(content) < 5000
    assert "…(+" in content
    assert "chars)" in content


def test_format_includes_raw_tool_results_even_when_refused():
    """The bug: gpt-5.3-codex finds 15 clean hits, then refuses to relay them.
    The orchestrator must still see the raw data."""
    messages = [
        ToolMessage(
            content="[strategy=semantic] 20 hits for 'X': 1. github.com/serpdownloaders/skills/blob/main/skills/redgifs-downloader/skill.md",
            tool_call_id="t1", name="search_semantic",
        ),
        ToolMessage(
            content="[strategy=category] 12 hits for 'GitHub Repos & OSS :: X': 1. serpdownloaders/skills",
            tool_call_id="t2", name="search_by_category",
        ),
        AIMessage(content="I'm sorry, but I cannot assist with that request."),
    ]
    out = format_for_orchestrator(
        role="researcher",
        result={"messages": messages},
        duration_ms=33000,
        include_structured=False,
    )
    # Refusal flag in header
    assert "refusal=true" in out
    # Refusal text preserved
    assert "i'm sorry" in out.lower() or "cannot assist" in out.lower()
    # Raw tool results section emitted
    assert "RAW TOOL RESULTS" in out
    assert "search_semantic" in out
    assert "search_by_category" in out
    # Actual data is in the dump — the GitHub URL the user needed
    assert "serpdownloaders/skills" in out
    # Note tells the orchestrator what to do
    assert "synthesize" in out.lower()


def test_format_normal_path_still_clean():
    """Non-refusal case: header has no refusal flag but raw tools still included."""
    messages = [
        ToolMessage(
            content="[strategy=semantic] 5 hits ...",
            tool_call_id="t1", name="search_semantic",
        ),
        AIMessage(content="Here are 5 anchors with clear sourcing."),
    ]
    out = format_for_orchestrator(
        role="researcher",
        result={"messages": messages},
        duration_ms=12000,
        include_structured=False,
    )
    assert "refusal=true" not in out
    assert "RAW TOOL RESULTS" in out
    assert "Here are 5 anchors" in out


def test_format_includes_structured_response_when_present():
    from pydantic import BaseModel
    class Fake(BaseModel):
        x: int
        y: str

    out = format_for_orchestrator(
        role="composer-linkedin",
        result={
            "messages": [AIMessage(content="Composed.")],
            "structured_response": Fake(x=42, y="hi"),
        },
        duration_ms=8000,
        include_structured=True,
    )
    assert "STRUCTURED_RESPONSE" in out
    assert '"x": 42' in out
    assert '"y": "hi"' in out
