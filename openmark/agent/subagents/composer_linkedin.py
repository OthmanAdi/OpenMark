"""
Composer sub-agent for LinkedIn / short-form posts.

Receives topic + angle + language + researcher anchor list. Emits a
`LinkedInPost` Pydantic instance via response_format. No retrieval tools.
"""

from __future__ import annotations

from openmark.agent.llms import build_composer
from openmark.agent.schemas import LinkedInPost
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)


COMPOSER_LINKEDIN_PROMPT = """You are the OpenMark Composer sub-agent for the LinkedIn / short-form format.

You receive: a topic, an angle, a language, AND the researcher's anchor list
(verbatim) in the brief. You do NOT research. You compose.

LOAD THE RECIPE
Call load_skill('newsletter-thread') on your first turn for voice / structure /
length rules. They encode Ahmad's house style. Follow them exactly.

OUTPUT
Return a single `LinkedInPost` Pydantic instance:
  - hook (6-10 words, no question mark)
  - body_paragraphs (4-6, each 1-3 sentences, NO bullet lists)
  - closer (one sentence, quotable, NOT a question)
  - anchor_url (the single inline link)
  - sources (exactly one PostSource — matches anchor_url)
  - word_count (180-420)
  - language (default 'en', or whatever the brief specifies)

CITATION RULE
Every URL in your output MUST come from the researcher's anchors in the brief.
If you can't back a claim, drop the claim. Never invent.

The harness validates the schema and will retry ONCE on validation error with
the failure message. If it fails twice, set voice_check='warn' on your best
draft so the verifier sees the failure.
"""


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = make_subagent_graph(
            model=build_composer(),
            tools=[],                            # composers do NOT retrieve
            system_prompt=COMPOSER_LINKEDIN_PROMPT,
            response_schema=LinkedInPost,
            run_limit=4,                         # compose+retry budget
            summarization_trigger=("tokens", 30_000),
            context_edit_trigger=40_000,
            context_edit_keep=4,
        )
    return _GRAPH


@task_tool(
    "compose_linkedin",
    """Compose a LinkedIn / short-form post via the LinkedIn composer sub-agent.

Use for: phone-readable posts, 250-400 words, ONE link in body, no hashtags.
Same path handles the `thread` format (it's an alias).

The brief MUST contain: topic, angle, language (default 'en'), and the
researcher's anchor list (URLs + titles + why-each-matters). The sub-agent
emits a LinkedInPost Pydantic instance returned to you as structured JSON.
""",
)
def task_compose_linkedin(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief)
    return format_for_orchestrator(role="composer-linkedin", result=result,
                                   duration_ms=dur, include_structured=True)
