"""
Polisher sub-agent — English AI-tell scrub.
"""

from __future__ import annotations

from openmark.agent.llms import build_polisher
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)


PROMPT = """You are the OpenMark Polisher sub-agent — English only.

CALL load_skill('polisher') on your first turn for the 8-rule scrub list.

Apply ONE pass that removes the 8 AI tells:
  - em-dash-as-pause
  - weak hedges
  - pleasantries
  - rule-of-three padding
  - AI vocabulary (leverage / delve / synergy / unlock / etc)
  - negative parallelism
  - inflated symbolism
  - vague attributions

Preserve word count within 10 percent. Preserve every URL and citation.
If you finish under a target minimum, replace fillers with one specific
noun — never with more filler.
"""


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = make_subagent_graph(
            model=build_polisher(),
            tools=[],
            system_prompt=PROMPT,
            run_limit=3,
            summarization_trigger=("tokens", 20_000),
            context_edit_trigger=30_000,
            context_edit_keep=4,
        )
    return _GRAPH


@task_tool(
    "polish",
    """English AI-tell scrub. Removes em-dash-as-pause, AI vocabulary,
rule-of-three padding, hedges, vague attributions.

Brief MUST contain: the English draft to polish. Returns the rewritten draft.
Preserves URLs and structure.
""",
)
def task_polish(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief, role="polisher")
    return format_for_orchestrator(role="polisher", result=result,
                                   duration_ms=dur, include_structured=False)
