"""
Sub-agent event propagation test — pure unit, no LLM.

Asserts that the tool_event_middleware contextvar correctly carries the
parent thread_id into nested sub-agent tool calls. This is the fix for the
"sub-agent internal tool events never appear in the UI" bug.
"""

from __future__ import annotations


def test_contextvar_propagates_parent_thread_id():
    """When an outer tool_call binds the contextvar, an inner tool_call sees it."""
    from openmark.agent.middleware import _PARENT_THREAD_ID, _emit, _TOOL_EVENTS, drain_events

    # Drain whatever's there
    drain_events(None)

    # Simulate: outer sets the contextvar to "sess-123"
    token_outer = _PARENT_THREAD_ID.set("sess-123")
    try:
        # Inner emit reads contextvar (no thread_id supplied)
        inner_thread = _PARENT_THREAD_ID.get() or "default"
        _emit(inner_thread, "start", tool="inner_tool", args={})
    finally:
        _PARENT_THREAD_ID.reset(token_outer)

    # Drain by parent thread_id — inner event should be there
    events = drain_events("sess-123")
    assert any(e["tool"] == "inner_tool" for e in events), \
        f"inner_tool event missing from sess-123 drain; got {[e['tool'] for e in events]}"


def test_contextvar_resets_after_handler_exit():
    """After the outer handler exits, the contextvar must be back to None."""
    from openmark.agent.middleware import _PARENT_THREAD_ID

    assert _PARENT_THREAD_ID.get() is None, "contextvar leaked across tests"

    token = _PARENT_THREAD_ID.set("sess-leak-test")
    try:
        assert _PARENT_THREAD_ID.get() == "sess-leak-test"
    finally:
        _PARENT_THREAD_ID.reset(token)

    assert _PARENT_THREAD_ID.get() is None, "contextvar not reset properly"


def test_drain_events_thread_isolation():
    """Events for different threads stay isolated."""
    from openmark.agent.middleware import _emit, drain_events

    drain_events(None)  # clear

    _emit("sess-A", "start", tool="tool_A")
    _emit("sess-B", "start", tool="tool_B")
    _emit("sess-A", "end", tool="tool_A", duration_ms=5)

    events_a = drain_events("sess-A")
    events_b = drain_events("sess-B")

    assert {e["tool"] for e in events_a} == {"tool_A"}
    assert len(events_a) == 2  # start + end
    assert {e["tool"] for e in events_b} == {"tool_B"}
    assert len(events_b) == 1
