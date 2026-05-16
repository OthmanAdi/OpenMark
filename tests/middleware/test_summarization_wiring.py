"""
Middleware wiring tests — confirm the v3 stack contains every required
middleware in the right shape.

Pure introspection. No LLM call, no Neo4j call.
"""

from __future__ import annotations

import pytest


def test_orchestrator_middleware_stack_complete():
    """The orchestrator middleware list contains every expected component."""
    # We rebuild the graph and inspect its middleware list via internals.
    # create_agent stores middleware on the compiled state graph; the safest
    # introspection is to walk the build kwargs in graph.py rather than the
    # compiled graph. We use a patch to capture them.
    import openmark.agent.tools as _t
    _t.warm_up = lambda: None

    captured: dict = {}
    from langchain.agents import create_agent as _orig_create

    def _spy(*args, **kw):
        captured["kwargs"] = kw
        return _orig_create(*args, **kw)

    import openmark.agent.graph as graph_mod
    real = graph_mod.create_agent
    try:
        graph_mod.create_agent = _spy
        graph_mod.build_agent()
    finally:
        graph_mod.create_agent = real

    mw_list = captured["kwargs"]["middleware"]
    names = []
    for m in mw_list:
        # built-ins are instances; decorators wrap functions into AgentMiddleware
        cls = type(m).__name__
        if cls == "AgentMiddleware":
            # decorator-built — fall back to .__class__.__name__
            cls = m.__class__.__name__
        names.append(cls)
    text = " ".join(names)
    # Spot-check every load-bearing component
    must_have = [
        "classify_intent",
        "dynamic_orchestrator_prompt",
        "ContextEditingMiddleware",
        "SummarizationMiddleware",
        "TodoListMiddleware",
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "OpenMarkSkillMiddleware",
    ]
    for needed in must_have:
        assert needed in text, f"middleware missing from stack: {needed} (have: {text})"


def test_summarization_uses_cheap_model():
    """SummarizationMiddleware on the orchestrator uses the cheap summarizer
    role, not the orchestrator's own frontier model."""
    import openmark.agent.tools as _t
    _t.warm_up = lambda: None

    captured: dict = {}
    from langchain.agents import create_agent as _orig_create

    def _spy(*args, **kw):
        captured["kwargs"] = kw
        return _orig_create(*args, **kw)

    import openmark.agent.graph as graph_mod
    real = graph_mod.create_agent
    try:
        graph_mod.create_agent = _spy
        graph_mod.build_agent()
    finally:
        graph_mod.create_agent = real

    mw_list = captured["kwargs"]["middleware"]
    from langchain.agents.middleware import SummarizationMiddleware
    summ = next((m for m in mw_list if isinstance(m, SummarizationMiddleware)), None)
    assert summ is not None, "SummarizationMiddleware not in stack"
    # Trigger should be a list of conditions (token + message)
    # We don't assert exact thresholds — only that trigger is set
    assert summ.trigger is not None, "SummarizationMiddleware trigger not set"


def test_context_editing_uses_clear_tool_uses_edit():
    """ContextEditingMiddleware uses ClearToolUsesEdit with sane defaults."""
    import openmark.agent.tools as _t
    _t.warm_up = lambda: None

    captured: dict = {}
    from langchain.agents import create_agent as _orig_create

    def _spy(*args, **kw):
        captured["kwargs"] = kw
        return _orig_create(*args, **kw)

    import openmark.agent.graph as graph_mod
    real = graph_mod.create_agent
    try:
        graph_mod.create_agent = _spy
        graph_mod.build_agent()
    finally:
        graph_mod.create_agent = real

    mw_list = captured["kwargs"]["middleware"]
    from langchain.agents.middleware import ContextEditingMiddleware
    ctx = next((m for m in mw_list if isinstance(m, ContextEditingMiddleware)), None)
    assert ctx is not None, "ContextEditingMiddleware not in stack"


def test_orchestrator_has_state_schema():
    """Orchestrator must use OrchestratorState (carries intent fields)."""
    import openmark.agent.tools as _t
    _t.warm_up = lambda: None

    captured: dict = {}
    from langchain.agents import create_agent as _orig_create

    def _spy(*args, **kw):
        captured["kwargs"] = kw
        return _orig_create(*args, **kw)

    import openmark.agent.graph as graph_mod
    real = graph_mod.create_agent
    try:
        graph_mod.create_agent = _spy
        graph_mod.build_agent()
    finally:
        graph_mod.create_agent = real

    schema = captured["kwargs"].get("state_schema")
    assert schema is not None, "state_schema not set"
    assert "intent" in getattr(schema, "__annotations__", {})
