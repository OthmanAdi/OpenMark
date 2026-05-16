"""
Skill-author sub-agent — bakes a recurring prompt into a reusable skill
via the sandboxed `write_skill` tool.
"""

from __future__ import annotations

from openmark.agent.llms import build_skill_author
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)
from openmark.agent.tools import write_skill


PROMPT = """You are the OpenMark Skill-Author sub-agent.

Called ONLY when the orchestrator decides a recurring prompt should be baked
into a reusable skill. You have `write_skill` and nothing else.

Inputs in the brief: a `name` (kebab-case, 2-50 chars), a one-line
`description`, and a `body` (the skill recipe in plain markdown).

Call write_skill(name=..., description=..., body=...) ONCE.
Return its result string verbatim (success or BLOCKED reason).

Cap: 5 writes per session, sandboxed to .claude/skills/agent-generated-*.
"""


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = make_subagent_graph(
            model=build_skill_author(),
            tools=[write_skill],
            system_prompt=PROMPT,
            run_limit=3,
            summarization_trigger=("tokens", 20_000),
            context_edit_trigger=30_000,
            context_edit_keep=4,
        )
    return _GRAPH


@task_tool(
    "author_skill",
    """Author a new reusable skill at runtime via the Skill-Author sub-agent.

Sandboxed; max 5 writes per session. The orchestrator can load_skill('<name>')
immediately after on the same turn.

Brief MUST contain: skill name (kebab-case), one-line description, full body
markdown. Returns the write_skill result string.
""",
)
def task_author_skill(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief)
    return format_for_orchestrator(role="skill-author", result=result,
                                   duration_ms=dur, include_structured=False)
