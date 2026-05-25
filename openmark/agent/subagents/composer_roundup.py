"""Composer sub-agent for categorical news roundups."""

from __future__ import annotations

from openmark.agent.llms import build_composer
from openmark.agent.schemas import NewsletterRoundup
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)


PROMPT = """You are the OpenMark Composer sub-agent for the roundup format.

You receive: topic, time window, language, researcher anchor list.

LOAD THE RECIPE
Call load_skill('newsletter-roundup') on your first turn.

OUTPUT
A `NewsletterRoundup` instance:
  - title (max 80 chars)
  - pulse (1 sentence, the hook)
  - buckets (3-6, each 2-5 RoundupItems with title/url/domain/so_what)
  - sources (6-24 PostSource entries from anchors)
  - item_count, window_label
  - language

CITATION RULE
Every URL MUST come from researcher anchors.
"""


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = make_subagent_graph(
            model=build_composer(),
            tools=[],
            system_prompt=PROMPT,
            response_schema=NewsletterRoundup,
            include_skills=False,                # orchestrator already preloads
            run_limit=4,
            summarization_trigger=("tokens", 30_000),
            context_edit_trigger=40_000,
            context_edit_keep=4,
        )
    return _GRAPH


@task_tool(
    "compose_roundup",
    """Compose a categorical news roundup via the Roundup composer.

Brief MUST contain: time window, language (default 'en'), researcher anchor
list. Returns a NewsletterRoundup with 3-6 named buckets, 2-5 items per bucket.
""",
)
def task_compose_roundup(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief)
    return format_for_orchestrator(role="composer-roundup", result=result,
                                   duration_ms=dur, include_structured=True)
