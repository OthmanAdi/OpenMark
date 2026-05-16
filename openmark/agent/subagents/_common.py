"""
Shared infrastructure for sub-agent modules.

A sub-agent is a compiled `langchain.agents.create_agent` graph exposed to the
orchestrator as a `@tool`. The tool's contract:
  - Input: a `brief` string from the orchestrator (the mission for this turn).
  - Output: a single string the orchestrator can read, containing:
        * the sub-agent's textual final answer
        * any `structured_response` JSON if response_format= was set
        * a compact telemetry tail (duration, tool_call count)
  - Side effects: every internal tool call surfaces in the UI via the shared
    `tool_event_middleware` event bus.

We cache compiled sub-agents at module level so we pay graph compilation
exactly once per process.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool as _tool_decorator
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from openmark.agent.llms import build_summarizer
from openmark.agent.middleware import tool_event_middleware


def make_subagent_graph(
    *,
    model: BaseChatModel,
    tools: Sequence[Any] | None,
    system_prompt: str,
    response_schema: type[BaseModel] | None = None,
    summarization_trigger: tuple[str, int] = ("tokens", 40_000),
    summarization_keep: tuple[str, int] = ("messages", 12),
    context_edit_trigger: int = 80_000,
    context_edit_keep: int = 5,
    run_limit: int = 8,
    extra_middleware: Sequence[AgentMiddleware] | None = None,
):
    """
    Compile a sub-agent graph with our standard middleware stack.

    Middleware order (verified):
        1. ContextEditingMiddleware  (deterministic prune of big tool outputs)
        2. SummarizationMiddleware    (fallback history compaction)
        3. ModelCallLimitMiddleware   (per-turn cap, sub-agents are bounded)
        4. ToolRetryMiddleware        (transient tool flakes)
        5. tool_event_middleware      (UI event bus — sub-agent tool calls
                                       appear as cards just like orchestrator tools)
        6. ...extra_middleware (caller-supplied additions go LAST)
    """
    mw: list[AgentMiddleware] = [
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=context_edit_trigger, keep=context_edit_keep)],
        ),
        SummarizationMiddleware(
            model=build_summarizer(),
            trigger=summarization_trigger,
            keep=summarization_keep,
        ),
        ModelCallLimitMiddleware(run_limit=run_limit, exit_behavior="end"),
        ToolRetryMiddleware(max_retries=2),
        tool_event_middleware,
    ]
    if extra_middleware:
        mw.extend(extra_middleware)

    kwargs: dict[str, Any] = dict(
        model=model,
        tools=list(tools or []),
        system_prompt=system_prompt,
        middleware=mw,
    )
    if response_schema is not None:
        kwargs["response_format"] = ToolStrategy(response_schema)
    return create_agent(**kwargs)


def _final_text(messages: list) -> str:
    """Extract clean text of the last AIMessage, ignoring reasoning blocks."""
    if not messages:
        return ""
    last = messages[-1]
    c = getattr(last, "content", "") or ""
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict):
                btype = b.get("type", "")
                if btype in ("text", "output_text") or btype == "":
                    t = (b.get("text", "") or "").strip()
                    if t:
                        parts.append(t)
            elif isinstance(b, str) and b.strip():
                parts.append(b.strip())
        return "\n\n".join(parts)
    return str(c)


def _count_tool_calls(messages: list) -> int:
    n = 0
    for m in messages:
        tc = getattr(m, "tool_calls", None) or []
        n += len(tc)
    return n


def format_for_orchestrator(
    *,
    role: str,
    result: dict,
    duration_ms: float,
    include_structured: bool = True,
    text_char_cap: int = 6_000,
) -> str:
    """
    Render a sub-agent result for the orchestrator's view.

    Long enough to act on, short enough that ContextEditingMiddleware can
    keep the last few without blowing the context. The orchestrator sees:

        [subagent=<role> tool_calls=N duration=Xms]
        <textual final answer, trimmed to text_char_cap>

        STRUCTURED_RESPONSE:
        <JSON dump if present>
    """
    messages = (result or {}).get("messages", []) or []
    text = _final_text(messages) or "(no textual response)"
    if len(text) > text_char_cap:
        text = text[:text_char_cap].rstrip() + f"\n\n…(truncated; full length {len(text)} chars)"

    n_calls = _count_tool_calls(messages)
    header = f"[subagent={role} tool_calls={n_calls} duration={int(duration_ms)}ms]"

    body = [header, text]
    if include_structured:
        sr = (result or {}).get("structured_response")
        if sr is not None:
            if isinstance(sr, BaseModel):
                payload = sr.model_dump_json(indent=2)
            else:
                try:
                    payload = json.dumps(sr, ensure_ascii=False, indent=2)
                except Exception:
                    payload = str(sr)
            body.append("\nSTRUCTURED_RESPONSE:\n" + payload)
    return "\n".join(body).strip()


def invoke_subagent(graph, brief: str) -> tuple[dict, float]:
    """Invoke a compiled sub-agent graph with a brief, return (result, duration_ms)."""
    t0 = time.time()
    result = graph.invoke({"messages": [{"role": "user", "content": brief}]})
    return result, (time.time() - t0) * 1000.0


def task_tool(role: str, description: str) -> Callable:
    """
    Decorator factory: turn a function `fn(brief: str) -> str` into a
    LangChain `@tool` named `task_<role>` with the supplied description.

    The wrapped function MUST take exactly one positional arg `brief: str`
    and return a string. We pass `name=` and `description=` directly to
    @tool so the function doesn't need its own docstring.
    """
    desc = description.strip()
    name = f"task_{role}"

    def _wrap(fn: Callable[[str], str]) -> Any:
        # langchain_core.tools.tool insists on a docstring; we fill it from
        # the supplied description so the function doesn't need one inline.
        if not fn.__doc__:
            fn.__doc__ = desc
        return _tool_decorator(name, description=desc)(fn)

    return _wrap
