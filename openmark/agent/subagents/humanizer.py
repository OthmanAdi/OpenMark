"""
Humanizer sub-agent — rewrites Arabic / Hebrew composer drafts to remove
AI tells. Loads the matching humanizer-* skill body by language.
"""

from __future__ import annotations

from openmark.agent.llms import build_humanizer
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)


PROMPT = """You are the OpenMark Humanizer sub-agent — Arabic + Hebrew only.

You receive: a composer draft + a target language (one of ar-msa, ar-egt,
ar-shami, he). Run ONE pass:

  1. Pick the matching skill: humanizer-ar-msa, humanizer-ar-egt,
     humanizer-ar-shami, or humanizer-he. CALL load_skill('<short_name>').
  2. Apply the skill's three stages (identify, rewrite, audit) on the draft.
  3. In your final answer, return the rewritten body text verbatim and set
     `humanizer_applied = True` if you also re-emit the Pydantic shape.

DO NOT change `sources`, URLs, structure, or word counts beyond ~10 percent.
ONLY rewrite text. English drafts are NOT your job — the polisher handles those.
"""


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = make_subagent_graph(
            model=build_humanizer(),
            tools=[],
            system_prompt=PROMPT,
            # KEEP include_skills=True: the humanizer's PROMPT instructs the
            # model to CALL load_skill('<lang>') to get the per-language
            # humanizer-* rules (humanizer-ar-msa etc.). Disabling it would
            # leave the humanizer with no rules to apply. The skill body IS
            # the humanizer's job description.
            run_limit=3,
            summarization_trigger=("tokens", 20_000),
            context_edit_trigger=30_000,
            context_edit_keep=4,
        )
    return _GRAPH


@task_tool(
    "humanize",
    """Rewrite an Arabic (MSA / Egyptian / Levantine) or Hebrew composer
draft to remove AI tells. Loads the matching humanizer-* skill.

Brief MUST contain: target language (ar-msa, ar-egt, ar-shami, or he) and
the draft body to rewrite. Returns the rewritten draft as text — preserves
structure, word count, and citations.
""",
)
def task_humanize(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief, role="humanizer")
    return format_for_orchestrator(role="humanizer", result=result,
                                   duration_ms=dur, include_structured=False)
