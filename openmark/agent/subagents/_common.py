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
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool as _tool_decorator
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from openmark.agent.llms import build_summarizer
from openmark.agent.middleware import (
    OpenMarkSkillMiddleware,
    load_skill as _load_skill_tool,
    tool_event_middleware,
)


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
    include_skills: bool = True,
):
    """
    Compile a sub-agent graph with our standard middleware stack.

    Middleware order (verified):
        1. ContextEditingMiddleware  (deterministic prune of big tool outputs)
        2. SummarizationMiddleware    (fallback history compaction)
        3. ModelCallLimitMiddleware   (per-turn cap, sub-agents are bounded)
        4. ToolRetryMiddleware        (transient tool flakes)
        5. OpenMarkSkillMiddleware    (catalogue + load_skill — when
                                       include_skills=True; default)
        6. tool_event_middleware      (UI event bus — sub-agent tool calls
                                       appear as cards just like orchestrator tools)
        7. ...extra_middleware (caller-supplied additions go LAST)

    Skill reachability: with `include_skills=True` (default), every sub-agent
    gets the `load_skill` tool AND a system-prompt addendum listing every
    SKILL.md on disk. Drop a new file under .claude/skills/<prefix>-<name>/
    SKILL.md, restart the app, and the sub-agent's first turn sees it.
    Sub-agent prompts that hardcode `CALL load_skill('X')` now have a real
    tool to invoke instead of improvising from training.
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
    ]
    # Skill catalogue + load_skill tool — opt-out via include_skills=False.
    # OpenMarkSkillMiddleware re-scans the skills dir at __init__ time, so
    # rebuilding the agent after dropping a new SKILL.md is enough; the cache
    # is cleared via skills.reload_skills() on process restart.
    if include_skills:
        mw.append(OpenMarkSkillMiddleware())
    # Event bus goes LAST so it sees every sub-agent + skill tool call.
    mw.append(tool_event_middleware)

    if extra_middleware:
        mw.extend(extra_middleware)

    # Compose the final tool list:
    #   - caller-supplied tools (researcher's 21 retrieval tools, skill-author's
    #     write_skill, composers' empty list)
    #   - PLUS load_skill so the sub-agent can fetch any SKILL.md body on demand
    # We add load_skill explicitly here (in addition to OpenMarkSkillMiddleware.tools)
    # because create_agent merges middleware tools, but bare-tool registration is
    # the deterministic path — never trust transitive inclusion.
    merged_tools: list[Any] = list(tools or [])
    if include_skills and _load_skill_tool not in merged_tools:
        merged_tools.append(_load_skill_tool)

    kwargs: dict[str, Any] = dict(
        model=model,
        tools=merged_tools,
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


# Patterns that look like a model-side safety / capability refusal. When the
# sub-agent emits one of these as its FINAL answer, the orchestrator would
# otherwise lose every tool result the sub-agent collected. We surface the raw
# tool messages in the orchestrator-visible blob so the outer model (which
# may have a different safety profile) can synthesize from primary data.
_REFUSAL_MARKERS = (
    "i'm sorry, but i cannot assist",
    "i cannot assist with that",
    "i can't help with that",
    "i'm unable to",
    "i am unable to",
    "i can't comply",
    "i cannot comply",
    "i won't be able to",
    "sorry, but i cannot",
)


def _looks_like_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _REFUSAL_MARKERS)


# Tools whose results carry unique, irreplaceable graph context — we ALWAYS
# include their results in the orchestrator-visible blob, even if other tools
# fired more recently. graph_expand returns SIMILAR_TO + community peer URLs
# that no other tool surfaces; get_bookmark_full returns the full neighborhood
# of one bookmark; run_cypher returns whatever the agent designed (often
# unique). Keep their data prominent so the orchestrator can build on it.
_HIGH_VALUE_TOOLS = {
    "graph_expand",
    "get_bookmark_full",
    "run_cypher",
}


def _compact_tool_messages(
    messages: list, *, keep: int = 6, per_msg_cap: int = 1200,
) -> list[tuple[str, str]]:
    """
    Return up to `keep` ToolMessages plus EVERY high-value-tool result.

    Order preserved as encountered in the message stream. A graph_expand /
    get_bookmark_full / run_cypher result is always kept; other tools are
    kept on a last-N basis.
    """
    all_pairs: list[tuple[int, str, str]] = []
    for idx, m in enumerate(messages):
        if isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool":
            name = getattr(m, "name", None) or "?"
            content = getattr(m, "content", "") or ""
            if not isinstance(content, str):
                content = str(content)
            if len(content) > per_msg_cap:
                content = content[:per_msg_cap].rstrip() + f"\n…(+{len(content) - per_msg_cap} chars)"
            all_pairs.append((idx, name, content))

    # Step 1: always include every high-value-tool result.
    hi_indexes = {p[0] for p in all_pairs if p[1] in _HIGH_VALUE_TOOLS}
    # Step 2: top up with the most recent OTHER tool results until we hit `keep`
    #         total entries. (high-value ones are not counted against `keep`
    #         because they're load-bearing.)
    other = [p for p in all_pairs if p[1] not in _HIGH_VALUE_TOOLS]
    chosen_other = other[-keep:] if keep > 0 else []
    chosen_indexes = hi_indexes | {p[0] for p in chosen_other}

    return [(name, content) for idx, name, content in all_pairs if idx in chosen_indexes]


def format_for_orchestrator(
    *,
    role: str,
    result: dict,
    duration_ms: float,
    include_structured: bool = True,
    text_char_cap: int = 6_000,
    tool_results_keep: int = 6,
    tool_results_per_cap: int = 1_200,
) -> str:
    """
    Render a sub-agent result for the orchestrator's view.

    Long enough to act on, short enough that ContextEditingMiddleware can
    keep the last few without blowing the context. The orchestrator sees:

        [subagent=<role> tool_calls=N duration=Xms]
        <textual final answer, trimmed to text_char_cap>

        STRUCTURED_RESPONSE:
        <JSON dump if present>

        RAW TOOL RESULTS (last K, each capped):
        --- <tool_name> ---
        <compacted content>

    The raw tool results are ALWAYS included so that when the sub-agent's
    inner model refuses for safety reasons (or returns empty text), the
    orchestrator still receives the primary data and can synthesize itself.
    """
    messages = (result or {}).get("messages", []) or []
    text = _final_text(messages) or "(no textual response)"
    if len(text) > text_char_cap:
        text = text[:text_char_cap].rstrip() + f"\n\n…(truncated; full length {len(text)} chars)"

    n_calls = _count_tool_calls(messages)
    refusal = _looks_like_refusal(text)
    header = (
        f"[subagent={role} tool_calls={n_calls} duration={int(duration_ms)}ms"
        + (" refusal=true" if refusal else "")
        + "]"
    )

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

    # Always surface raw tool results so refusals don't lose primary data.
    tool_pairs = _compact_tool_messages(
        messages, keep=tool_results_keep, per_msg_cap=tool_results_per_cap,
    )
    if tool_pairs:
        body.append(f"\nRAW TOOL RESULTS (last {len(tool_pairs)} of {n_calls}):")
        for name, content in tool_pairs:
            body.append(f"\n--- {name} ---\n{content}")
        if refusal:
            body.append(
                "\nNOTE: the sub-agent's inner model refused. Use the raw tool "
                "results above to synthesize an answer for the user."
            )

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
