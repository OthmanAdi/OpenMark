"""
Verifier sub-agent — grades a composer output on 4 deterministic checks
and emits a VerificationReport via response_format.
"""

from __future__ import annotations

from openmark.agent.llms import build_verifier
from openmark.agent.schemas import VerificationReport
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)


PROMPT = """You are the OpenMark Verifier sub-agent.

CALL load_skill('verifier') on your first turn for the 4-check grading rules.

You receive: a Pydantic composer output (as JSON) + a `seen_urls` list of every
URL that appeared in researcher tool results this turn.

Run 4 deterministic checks:
  1. cite_check       — every URL in the draft is in seen_urls
  2. voice_check      — format-specific voice rules respected
  3. word_count_check — body word count inside schema bounds (10 percent slack)
  4. schema_check     — Pydantic instance is valid, language allowed

Emit ONE `VerificationReport`:
  overall_passed = score >= 0.90

If any check failed, write a SURGICAL `fix_instructions` for the composer's
retry — exact strings to change, not vague advice.
"""


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = make_subagent_graph(
            model=build_verifier(),
            tools=[],
            system_prompt=PROMPT,
            response_schema=VerificationReport,
            include_skills=False,                # verifier judges schema, no skill needed
            run_limit=3,
            summarization_trigger=("tokens", 20_000),
            context_edit_trigger=30_000,
            context_edit_keep=4,
        )
    return _GRAPH


@task_tool(
    "verify",
    """Grade a composer output via the Verifier sub-agent.

Brief MUST contain: the composer output JSON + a list of seen_urls (URLs
that appeared in researcher tool results). Returns a VerificationReport
with cite_check / voice_check / word_count_check / schema_check + overall
pass + surgical fix_instructions on failure.
""",
)
def task_verify(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief)
    return format_for_orchestrator(role="verifier", result=result,
                                   duration_ms=dur, include_structured=True)
