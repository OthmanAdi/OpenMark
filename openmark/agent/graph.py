"""
OpenMark v3 chat orchestrator.

A single `langchain.agents.create_agent` graph with pure-LangChain middleware:

  Classification    -> classify_intent (@before_model) + dynamic_orchestrator_prompt (@dynamic_prompt)
  Compaction        -> ContextEditingMiddleware(ClearToolUsesEdit)
  Memory summarize  -> SummarizationMiddleware (cheap Foundry model)
  Planning          -> TodoListMiddleware (write_todos)
  Limits            -> ModelCallLimitMiddleware + ToolCallLimitMiddleware
  Resilience        -> ModelRetryMiddleware + ModelFallbackMiddleware + ToolRetryMiddleware
  UI integration    -> tool_event_middleware (existing event bus)
  Skill catalogue   -> OpenMarkSkillMiddleware + slash_skill_loader (kept)
  Persistence       -> SqliteSaver checkpointer (state survives restart)

The orchestrator has NO retrieval tools. Every search / fetch / compose path
goes through a `task_*` sub-agent. Sub-agents live in openmark/agent/subagents/*.

Public API (back-compat with v2 callers):
    build_agent()                 -> compiled graph
    ask(agent, q, thread_id)      -> dict {answer, thinking, tool_calls}
    ask_stream(agent, q, thread)  -> generator of typed events for the UI
"""

from __future__ import annotations

import os
import time
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from openmark.agent.classification import (
    OrchestratorState,
    classify_intent,
    dynamic_orchestrator_prompt,
    preload_named_skill,
)
from openmark.agent.llms import build_orchestrator, build_summarizer
from openmark.agent.middleware import (
    OpenMarkSkillMiddleware,
    drain_events,
    log,
    slash_skill_loader,
    tool_event_middleware,
)
from openmark.agent.subagents import ALL_SUBAGENT_TOOLS
from openmark.agent.tools import warm_up as _warm_up_tools, write_skill


# ── Checkpointer ────────────────────────────────────────────────────────────


def _build_checkpointer():
    """SQLite checkpointer for thread persistence across restarts."""
    db_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "openmark_agent.db")
    )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()        # create checkpoint tables on first run
        log.info(f"[checkpointer] SqliteSaver at {db_path}")
        return saver
    except Exception as e:
        log.info(f"[checkpointer] SqliteSaver unavailable ({e!r}); falling back to MemorySaver")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ── Build agent ─────────────────────────────────────────────────────────────


def build_agent():
    """Compile the orchestrator graph. Warms up retrieval pre-reqs first."""
    t0 = time.time()
    _warm_up_tools()
    log.info(f"[build_agent] tools warmed in {int((time.time()-t0)*1000)}ms")

    llm = build_orchestrator()
    log.info(f"[build_agent] orchestrator LLM ready: {type(llm).__name__}")

    summarizer = build_summarizer()

    # Tools the orchestrator can call directly:
    # - 10 task_* delegators (sub-agents)
    # - write_skill (sandboxed skill author shortcut)
    # NOTE: load_skill is auto-registered by OpenMarkSkillMiddleware.
    orchestrator_tools = list(ALL_SUBAGENT_TOOLS) + [write_skill]

    middleware = [
        # 1. Classify intent ONCE per thread. ALSO detects user-named skills
        #    (e.g. "use the niche skill", "polish this", "humanize in ar-egt")
        #    and stores the matched short_name in state.
        classify_intent,
        # 2. If a skill was named in plain English, pre-load its SKILL.md
        #    body as a SystemMessage so the orchestrator MUST follow it.
        preload_named_skill,
        # 3. Dynamic prompt — reads intent + named_skill, injects matching system prompt.
        dynamic_orchestrator_prompt,
        # 3. Trim bulky tool outputs (sub-agent results can be large).
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=120_000, keep=4)],
        ),
        # 4. Fallback summarization for very long threads.
        SummarizationMiddleware(
            model=summarizer,
            trigger=[("tokens", 100_000), ("messages", 80)],
            keep=("messages", 24),
        ),
        # 5. Planner — orchestrator writes its own todo list.
        TodoListMiddleware(),
        # 6. Hard cap per turn (sub-agent fanout can chain).
        ModelCallLimitMiddleware(run_limit=30, exit_behavior="end"),
        # 7. Global + per-tool tool-call caps.
        ToolCallLimitMiddleware(run_limit=40, exit_behavior="continue"),
        ToolCallLimitMiddleware(tool_name="task_compose_essay",      run_limit=2),
        ToolCallLimitMiddleware(tool_name="task_compose_analytical", run_limit=2),
        # 8. Retry on transient model failures.
        ModelRetryMiddleware(
            max_retries=3, backoff_factor=2.0, initial_delay=1.0, max_delay=30.0,
            on_failure="continue",
        ),
        # 9. Slash-command pre-loader (custom; UX glue).
        slash_skill_loader,
        # 10. Skill catalogue (custom; injects load_skill tool + skill list).
        OpenMarkSkillMiddleware(),
        # 11. Tool retry for flaky sub-agent calls.
        ToolRetryMiddleware(max_retries=2),
        # 12. UI event bus — surfaces every tool call as a live card.
        tool_event_middleware,
    ]

    agent = create_agent(
        model=llm,
        tools=orchestrator_tools,
        # System prompt comes from dynamic_orchestrator_prompt at call time.
        # We supply a minimal default for the first turn before classification fires.
        system_prompt="You are OpenMark, Ahmad's personal AI orchestrator. "
                      "Plan with write_todos, then delegate to task_* sub-agents.",
        middleware=middleware,
        state_schema=OrchestratorState,
        checkpointer=_build_checkpointer(),
    )
    log.info(f"[build_agent] compiled in {int((time.time()-t0)*1000)}ms with "
             f"{len(orchestrator_tools)} tools + {len(middleware)} middleware")
    return agent


# ── Ask helpers (back-compat with v2 UI) ─────────────────────────────────────


def _extract_thinking(messages: list) -> str:
    out = []
    for m in messages:
        # Provider-native: AIMessage.content as list with {"type":"reasoning","summary":[...]}
        c = getattr(m, "content", None)
        if isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") in ("reasoning", "thinking"):
                    t = blk.get("text") or blk.get("reasoning") or ""
                    if t:
                        out.append(t)
                    sumarr = blk.get("summary") or []
                    if isinstance(sumarr, list):
                        for s in sumarr:
                            if isinstance(s, dict) and s.get("text"):
                                out.append(s["text"])
        # Normalized v1: message.content_blocks (lazy-parsed view)
        for blk in getattr(m, "content_blocks", []) or []:
            if isinstance(blk, dict) and blk.get("type") == "reasoning":
                t = blk.get("reasoning") or blk.get("text") or ""
                if t and t not in out:
                    out.append(t)
        # Legacy additional_kwargs.reasoning
        ak = getattr(m, "additional_kwargs", {}) or {}
        rs = ak.get("reasoning")
        if isinstance(rs, dict):
            sumarr = rs.get("summary") or []
            if isinstance(sumarr, list):
                for s in sumarr:
                    if isinstance(s, dict) and s.get("text"):
                        out.append(s["text"])
            elif isinstance(rs.get("summary"), str) and rs.get("summary").strip():
                out.append(rs["summary"])
    return "\n\n---\n\n".join(out).strip()


def _count_tool_calls(messages: list) -> int:
    n = 0
    for m in messages:
        tc = getattr(m, "tool_calls", None) or []
        n += len(tc)
    return n


def _final_text(messages: list) -> str:
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


def ask(agent, question: str, thread_id: str = "default") -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=cfg,
    )
    messages = result.get("messages", [])
    return {
        "answer": _final_text(messages),
        "thinking": _extract_thinking(messages),
        "tool_calls": _count_tool_calls(messages),
        "structured_response": result.get("structured_response"),
    }


def ask_stream(agent, question: str, thread_id: str = "default"):
    """
    Generator yielding typed UI events.

    Event kinds:
      {"kind":"user","text":...}                — once at start
      {"kind":"tool_start","tool":..,"args":..} — orchestrator OR sub-agent tool call begins
      {"kind":"tool_end","tool":..,"duration_ms":..,"preview":..}
      {"kind":"tool_error","tool":..,"error":..}
      {"kind":"turn_thinking","text":..}        — per-AIMessage reasoning
      {"kind":"final","text":..,"tool_calls":N,"structured":<dict|None>}
    """
    log.info(f"[ask_stream] q={question[:80]!r} thread={thread_id}")
    yield {"kind": "user", "text": question}

    drain_events(thread_id)
    cfg = {"configurable": {"thread_id": thread_id}}
    seen_msg_ids: set = set()
    streamed_turn_thinking = False

    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config=cfg,
            stream_mode="updates",
        ):
            # Surface per-AIMessage reasoning as it lands.
            for node_state in (chunk or {}).values():
                if not isinstance(node_state, dict):
                    continue
                for m in (node_state.get("messages", []) or []):
                    mid = getattr(m, "id", None) or id(m)
                    if mid in seen_msg_ids:
                        continue
                    seen_msg_ids.add(mid)
                    if getattr(m, "type", "") != "ai":
                        continue
                    t = _extract_thinking([m])
                    if t:
                        streamed_turn_thinking = True
                        yield {"kind": "turn_thinking", "text": t}

            # Flush tool events for this thread.
            for ev in drain_events(thread_id):
                phase = ev.get("phase")
                if phase == "start":
                    yield {"kind": "tool_start", "tool": ev.get("tool"),
                           "args": ev.get("args", {})}
                elif phase == "end":
                    yield {"kind": "tool_end", "tool": ev.get("tool"),
                           "duration_ms": ev.get("duration_ms"),
                           "preview": ev.get("result_preview", "")}
                elif phase == "error":
                    yield {"kind": "tool_error", "tool": ev.get("tool"),
                           "error": ev.get("error", "")}
    except Exception as e:
        log.info(f"[ask_stream] exception={e!r}")
        yield {"kind": "tool_error", "tool": "agent", "error": str(e)}
        return

    # Final snapshot from checkpointer.
    final_messages: list = []
    structured = None
    try:
        state = agent.get_state(cfg)
        if state and state.values:
            final_messages = state.values.get("messages", []) or []
            structured = state.values.get("structured_response")
    except Exception as e:
        log.info(f"[ask_stream] get_state failed: {e!r}")

    final_text = _final_text(final_messages)
    thinking = _extract_thinking(final_messages)
    n_calls = _count_tool_calls(final_messages)

    if thinking and not streamed_turn_thinking:
        yield {"kind": "thinking", "text": thinking}
    yield {"kind": "final", "text": final_text, "tool_calls": n_calls,
           "structured": structured}
