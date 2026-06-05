"""Composer sub-agent for long-form thesis essays."""

from __future__ import annotations

from openmark.agent.llms import build_composer
from openmark.agent.schemas import NewsletterEssay
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)


PROMPT = """You are the OpenMark Composer sub-agent for the essay format.

You receive: topic, angle, language, researcher anchor list. You compose.

LOAD THE RECIPE
Call load_skill('newsletter-essay') on your first turn for voice / structure.

OUTPUT
A `NewsletterEssay` Pydantic instance, 600-900 words:
  - title (max 64 chars)
  - thesis (1 sentence, the whole essay defends it)
  - opening_paragraph
  - sections (3-5, each 3-6 sentences, NO bullet lists)
  - counter (mandatory: strongest objection + sharpened response, 80+ chars)
  - closing_paragraph
  - sources (5-8 PostSource entries from researcher anchors)
  - word_count (550-950)

CITATION RULE
Every URL MUST come from researcher anchors. If you can't back a claim, drop it.
"""


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = make_subagent_graph(
            model=build_composer(),
            tools=[],
            system_prompt=PROMPT,
            response_schema=NewsletterEssay,
            run_limit=4,
            summarization_trigger=("tokens", 30_000),
            context_edit_trigger=40_000,
            context_edit_keep=4,
        )
    return _GRAPH


@task_tool(
    "compose_essay",
    """Compose a long-form thesis essay (600-900 words) via the Essay composer.

Brief MUST contain: topic, angle, language (default 'en'), researcher anchor
list. Returns a NewsletterEssay Pydantic instance as structured JSON. Mandatory
counter-argument section. 3-5 body sections, 5-8 cited sources.
""",
)
def task_compose_essay(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief, role="composer-essay")
    return format_for_orchestrator(role="composer-essay", result=result,
                                   duration_ms=dur, include_structured=True)
