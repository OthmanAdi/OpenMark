"""Composer sub-agent for side-by-side comparison newsletters."""

from __future__ import annotations

from openmark.agent.llms import build_composer
from openmark.agent.schemas import NewsletterComparison
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)


PROMPT = """You are the OpenMark Composer sub-agent for the comparison format.

You receive: items to compare, dimensions hint, language, researcher anchor list.

LOAD THE RECIPE
Call load_skill('newsletter-comparison') on your first turn.

OUTPUT
A `NewsletterComparison` instance:
  - title (max 80)
  - recommendation (1-2 sentences, who wins for what)
  - items (2-5 strings)
  - rows (4-8 ComparisonRow with dimension + values matching items)
  - how_to_read
  - picks (2-5 ComparisonPick: item_name, condition, rationale)
  - sources (3-12 PostSource)
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
            response_schema=NewsletterComparison,
            run_limit=4,
            summarization_trigger=("tokens", 30_000),
            context_edit_trigger=40_000,
            context_edit_keep=4,
        )
    return _GRAPH


@task_tool(
    "compose_comparison",
    """Compose a side-by-side comparison newsletter via the Comparison composer.

Brief MUST contain: items being compared (2-5), language (default 'en'),
researcher anchor list. Returns a NewsletterComparison with table rows +
how-to-read + when-to-pick blocks.
""",
)
def task_compose_comparison(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief)
    return format_for_orchestrator(role="composer-comparison", result=result,
                                   duration_ms=dur, include_structured=True)
