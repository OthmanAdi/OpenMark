"""Composer sub-agent for analytical newsletters (the default 'newsletter' shape)."""

from __future__ import annotations

from openmark.agent.llms import build_composer
from openmark.agent.schemas import NewsletterAnalytical
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)


PROMPT = """You are the OpenMark Composer sub-agent for the analytical newsletter format.

You receive: topic, angle, language, researcher anchor list.

LOAD THE RECIPE
Call load_skill('newsletter') on your first turn — this is the primary
newsletter recipe encoding Ahmad's voice and structure.

OUTPUT
A `NewsletterAnalytical` instance, 600-900 words:
  - title (max 80)
  - hook (1-2 sentences, thesis)
  - what_happened_paragraphs (3-5, each cites at least one URL inline)
  - why_it_matters (1-2 paragraphs, the take)
  - what_im_reading (5-7 RoundupItems from anchors)
  - one_more_thing (optional weird/funny bookmark)
  - sources (5-15 PostSource)
  - word_count (550-950)

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
            response_schema=NewsletterAnalytical,
            run_limit=4,
            summarization_trigger=("tokens", 30_000),
            context_edit_trigger=40_000,
            context_edit_keep=4,
        )
    return _GRAPH


@task_tool(
    "compose_analytical",
    """Compose an analytical newsletter (600-900 words) — the default
"newsletter on X" format.

Brief MUST contain: topic, angle, language (default 'en'), researcher anchor
list. Returns a NewsletterAnalytical with hook + what-happened + why-it-matters
+ what-I'm-reading sections.
""",
)
def task_compose_analytical(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief, role="composer-analytical")
    return format_for_orchestrator(role="composer-analytical", result=result,
                                   duration_ms=dur, include_structured=True)
