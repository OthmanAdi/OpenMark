"""
Custom middleware for the OpenMark in-app agent.

Three concerns wired into LangChain v1 `create_agent`:

1. OpenMarkSkillMiddleware
   - Subclasses `AgentMiddleware` per the canonical LangChain skills pattern
     (see docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant).
   - Exposes a `load_skill(skill_name)` tool the agent can call mid-conversation
     when it realises a recipe like "newsletter compose" or "deep research" fits.
   - Injects an "Available skills" addendum into the system prompt via
     `wrap_model_call` so the agent SEES the catalogue without paying for the
     body of every skill on every turn (progressive disclosure).

2. Slash-command pre-load (`slash_skill_loader`)
   - `@before_model` hook. If the first human message starts with
     `/<skill-name>`, eagerly load that SKILL.md body and inject it as a
     SystemMessage so the agent has the recipe in context without spending a
     round-trip on `load_skill`.

3. Tool event bus (`tool_event_middleware`)
   - `@wrap_tool_call` hook. Captures each tool's name + args + duration into a
     module-level deque. The Gradio UI drains this between graph stream chunks
     to render tool calls as live cards in the chat.

All hook signatures verified against
docs.langchain.com/oss/python/langchain/middleware/* and the skills-sql-assistant
walkthrough.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from typing import Any, Callable

# Dedicated logger that ALSO prints to stdout so the terminal running the UI shows it.
log = logging.getLogger("openmark.agent")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [%(name)s] %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    before_model,
    wrap_tool_call,
)
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from openmark.agent import skills as skill_loader


# ── Tool event bus ────────────────────────────────────────────────────────────
# Each entry: {ts, thread_id, phase, tool, args, duration_ms, result_preview, error}
_TOOL_EVENTS: "deque[dict]" = deque(maxlen=400)


def _emit(thread_id: str, phase: str, **fields: Any) -> None:
    _TOOL_EVENTS.append({"ts": time.time(), "thread_id": thread_id, "phase": phase, **fields})


def drain_events(thread_id: str | None = None) -> list[dict]:
    """Pop matching events. UI calls this between stream chunks."""
    keep: list[dict] = []
    out: list[dict] = []
    while _TOOL_EVENTS:
        e = _TOOL_EVENTS.popleft()
        if thread_id is None or e.get("thread_id") == thread_id:
            out.append(e)
        else:
            keep.append(e)
    for e in keep:
        _TOOL_EVENTS.append(e)
    return out


# ── load_skill tool ───────────────────────────────────────────────────────────
@tool
def load_skill(skill_name: str) -> str:
    """
    Load a curated OpenMark skill recipe into your context. Skills bundle
    proven workflows for: deep research, newsletter composition, weekly digest,
    one-shot fast search, and single-URL deep dive.

    CALL THIS when the user's request looks like one of:
      - "research X across my saves"        → load 'openmark-deep-research'
      - "draft a newsletter on X"           → load 'openmark-newsletter'
      - "what did I save this week"         → load 'openmark-weekly-digest'
      - "find my bookmarks on X" (one-shot) → load 'openmark-fast-search'
      - "expand <URL>"                      → load 'openmark-bookmark-dive'

    Then follow the recipe in the loaded skill verbatim.

    Args:
        skill_name: Full name like 'openmark-newsletter' or short suffix like 'newsletter'.
    """
    skill = skill_loader.load_skill(skill_name)
    if not skill:
        available = ", ".join(s["short_name"] for s in skill_loader.list_skills())
        log.info(f"[load_skill] MISS name={skill_name!r} available={available}")
        return f"Skill '{skill_name}' not found. Available: {available}"
    log.info(f"[load_skill] HIT name={skill['name']} body_len={len(skill['body'])}")
    return f"Loaded skill: {skill['name']}\n\n{skill['body']}"


# ── OpenMarkSkillMiddleware ───────────────────────────────────────────────────
class OpenMarkSkillMiddleware(AgentMiddleware):
    """
    Progressive disclosure: agent sees skill descriptions (cheap) and only loads
    the full SKILL.md body when it decides to use one.
    """

    # Register load_skill as a class variable — create_agent picks it up.
    tools = [load_skill]

    def __init__(self) -> None:
        # Build the skill list addendum once at construction. If new skills land,
        # call skill_loader.reload_skills() and rebuild the agent.
        items = []
        for s in skill_loader.list_skills():
            items.append(f"- **{s['short_name']}** ({s['type']}): {s['description']}")
        self.skills_prompt = "\n".join(items) or "(no skills installed)"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Append the skill catalogue to the system prompt on every model call."""
        addendum = (
            f"\n\n## OpenMark skills (progressive disclosure)\n\n"
            f"{self.skills_prompt}\n\n"
            "When the user's request matches one of these patterns, CALL "
            "`load_skill(skill_name='<short-name>')` FIRST to load the recipe, "
            "THEN follow it. Do not invent your own workflow when a skill exists.\n"
        )

        sys_msg = request.system_message
        if sys_msg is None:
            new_system_message = SystemMessage(content=addendum)
        else:
            existing = sys_msg.content
            if isinstance(existing, str):
                new_system_message = SystemMessage(content=existing + addendum)
            else:
                # content_blocks API
                blocks = list(existing or []) + [{"type": "text", "text": addendum}]
                new_system_message = SystemMessage(content=blocks)
        return handler(request.override(system_message=new_system_message))


# ── Slash-command pre-loader (before_model) ───────────────────────────────────
@before_model
def slash_skill_loader(state: Any, runtime: Any) -> dict | None:
    """
    On the first turn: if the user typed `/<skill-name> <query>`, load the
    skill body eagerly as a SystemMessage. Strip the slash command from the
    user's message so the LLM sees only the actual question.
    """
    messages = state.get("messages") or []
    if not messages:
        return None
    if any(getattr(m, "type", "") in ("ai", "tool") for m in messages):
        return None

    first_human_idx = None
    first_human = None
    for i, m in enumerate(messages):
        if getattr(m, "type", "") == "human":
            first_human_idx = i
            first_human = m
            break
    if first_human is None:
        return None

    text = first_human.content
    if isinstance(text, list):
        text = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    if not isinstance(text, str) or not text.strip().startswith("/"):
        return None

    skill_name, remainder = skill_loader.parse_slash(text)
    if not skill_name:
        return None
    skill = skill_loader.load_skill(skill_name)
    if not skill:
        log.info(f"[slash] UNKNOWN /{skill_name} — passed through literal")
        return None

    log.info(f"[slash] PRELOAD skill={skill['name']} body_len={len(skill['body'])} remainder={remainder[:80]!r}")
    eager = SystemMessage(content=(
        f"# Active skill (pre-loaded by user via `/{skill['short_name']}`)\n\n"
        "Follow this recipe verbatim for the user's request below. The full "
        "skill body is here — you do NOT need to call load_skill again.\n\n"
        f"---\n\n{skill['body']}"
    ))
    new_messages = list(messages)
    new_messages[first_human_idx] = HumanMessage(content=remainder or text)
    return {"messages": [eager] + new_messages}


# ── Tool event capture (wrap_tool_call) ───────────────────────────────────────
@wrap_tool_call
def tool_event_middleware(request: Any, handler: Callable[[Any], Any]) -> Any:
    """
    Capture every tool call so the UI can render it live.

    Verified signature: request.tool_call is a dict with keys
    {name, args, id, type}. Source:
    docs.langchain.com/oss/python/langchain/agents (wrap_tool_call examples).
    """
    tc = getattr(request, "tool_call", {}) or {}
    tool_name = tc.get("name", "?") if isinstance(tc, dict) else "?"
    tool_args = tc.get("args", {}) if isinstance(tc, dict) else {}
    tool_id = tc.get("id", "") if isinstance(tc, dict) else ""

    thread_id = "default"
    try:
        runtime = getattr(request, "runtime", None)
        cfg = getattr(runtime, "config", None) or getattr(request, "config", None) or {}
        if isinstance(cfg, dict):
            thread_id = cfg.get("configurable", {}).get("thread_id", "default")
    except Exception:
        pass

    log.info(f"[tool→] {tool_name} args={tool_args!r}")
    _emit(thread_id, "start", tool=tool_name, args=tool_args, tool_id=tool_id)
    t0 = time.time()
    try:
        result = handler(request)
        dur = round((time.time() - t0) * 1000, 1)
        # Result can be a ToolMessage or Command — surface a FULL preview so the
        # UI card actually shows the data, not a 240-char crumb.
        preview = ""
        if isinstance(result, ToolMessage):
            preview = str(result.content)
        elif hasattr(result, "update") and isinstance(result.update, dict):
            msgs = result.update.get("messages", [])
            if msgs:
                preview = str(getattr(msgs[-1], "content", ""))
        else:
            preview = str(result)
        preview_short_for_log = preview[:160].replace("\n", " ")
        log.info(f"[tool✓] {tool_name} {dur}ms len={len(preview)} preview={preview_short_for_log!r}")
        _emit(thread_id, "end", tool=tool_name, args=tool_args, tool_id=tool_id,
              duration_ms=dur, result_preview=preview)
        return result
    except Exception as e:
        dur = round((time.time() - t0) * 1000, 1)
        log.info(f"[tool✗] {tool_name} {dur}ms error={e!r}")
        _emit(thread_id, "error", tool=tool_name, args=tool_args, tool_id=tool_id,
              duration_ms=dur, error=str(e)[:600])
        raise
