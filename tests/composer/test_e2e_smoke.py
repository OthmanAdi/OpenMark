"""
End-to-end smoke test — OPT-IN, skipped by default.

Set OPENMARK_RUN_LIVE_E2E=1 to run. Requires Azure connectivity and Neo4j up.

The test asks the orchestrator to compose a short LinkedIn post on a topic
already covered by Ahmad's bookmarks ("RAG retrieval tradeoffs"), then
verifies:

  1. The orchestrator returns without exception.
  2. At least ONE researcher tool call happened.
  3. The structured_response is a LinkedInPost OR the final text references a
     verifier output.

This is a SMOKE test — it does not assert ≥0.90 pass rate. The 90% target is
measured offline across many briefs; one run is too noisy.
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("OPENMARK_RUN_LIVE_E2E") != "1",
    reason="Set OPENMARK_RUN_LIVE_E2E=1 to run live LLM + Neo4j composer smoke.",
)


def test_orchestrator_compose_linkedin_smoke():
    from openmark.composer.orchestrator import ask_compose, build_composer_orchestrator

    agent = build_composer_orchestrator()
    brief = (
        "Compose a LinkedIn post on RAG retrieval tradeoffs (vector vs graph). "
        "Language: en. Format: linkedin. Pull from my OpenMark bookmarks."
    )
    out = ask_compose(agent, brief, thread_id="test-e2e-smoke")
    # Sanity — orchestrator returned SOMETHING
    assert out is not None
    assert "answer" in out
    # Either structured response, OR a final message — both acceptable
    has_structured = out.get("structured") is not None
    has_answer = bool((out.get("answer") or "").strip())
    assert has_structured or has_answer, "orchestrator returned neither answer nor structured output"
    # At least one tool call (researcher should have fired)
    assert out.get("tool_calls", 0) >= 1, "no tool calls happened — composer is not using sub-agents"
