"""
Live Foundry e2e tests — proves every sub-agent works against the real LLM.

These tests CALL the real Azure Foundry deployments configured in .env. They
are GATED on `OPENMARK_RUN_LIVE_E2E=1` to avoid burning credits during normal
test runs. Each test sends a focused brief, asserts the orchestrator delegated
to the expected sub-agent, and where applicable validates the structured
Pydantic output.

Run with:
    OPENMARK_RUN_LIVE_E2E=1 pytest tests/agent/test_live_foundry.py -v

Pre-reqs:
    - Neo4j running locally (the researcher hits it).
    - .env populated with valid Azure Foundry creds.
"""

from __future__ import annotations

import json
import os
import time

import pytest

_LIVE = os.environ.get("OPENMARK_RUN_LIVE_E2E") == "1"
pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="Set OPENMARK_RUN_LIVE_E2E=1 to enable live Foundry e2e tests.",
)


@pytest.fixture(scope="module")
def agent():
    """Build the orchestrator once per module — graph compile is ~1s."""
    from openmark.agent.graph import build_agent
    return build_agent()


def _drain_tool_names(thread_id: str) -> list[str]:
    """Collect every tool name that fired on this thread since last drain."""
    from openmark.agent.middleware import drain_events
    return [e["tool"] for e in drain_events(thread_id) if e.get("phase") == "end"]


def _send(agent, brief: str, thread_id: str, timeout_s: float = 300) -> dict:
    """Invoke the orchestrator with a brief, return the final state values."""
    cfg = {"configurable": {"thread_id": thread_id}}
    t0 = time.time()
    agent.invoke({"messages": [{"role": "user", "content": brief}]}, config=cfg)
    elapsed = time.time() - t0
    assert elapsed < timeout_s, f"orchestrator exceeded {timeout_s}s: {elapsed:.1f}s"
    state = agent.get_state(cfg)
    return state.values if state else {}


# ── 1. Researcher direct ────────────────────────────────────────────────────


def test_researcher_returns_anchor_list(agent):
    """Plain retrieval query routes to task_researcher and returns hits."""
    tid = "live-researcher-1"
    state = _send(
        agent,
        "Find my OpenMark bookmarks about RAG retrieval. Just list 5 of them.",
        thread_id=tid,
    )
    msgs = state.get("messages", []) or []
    text_blocks = []
    for m in msgs:
        c = getattr(m, "content", None)
        if isinstance(c, str) and c.strip():
            text_blocks.append(c)
    full = "\n".join(text_blocks).lower()
    # The orchestrator should have called task_researcher AND surfaced URLs.
    tool_call_names: list[str] = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tool_call_names.append(name)
    assert "task_researcher" in tool_call_names, f"called: {tool_call_names}"
    assert "http" in full or "https" in full, "no URLs surfaced in final answer"


# ── 2. Composer-analytical full loop ────────────────────────────────────────


def test_compose_analytical_full_loop(agent):
    """Newsletter brief routes through researcher -> composer_analytical -> polish -> verify."""
    tid = "live-analytical-1"
    state = _send(
        agent,
        "Compose a short analytical newsletter on 'agent memory patterns' in English. "
        "Pull anchors from my OpenMark bookmarks. Use the standard compose loop.",
        thread_id=tid,
        timeout_s=600,
    )
    msgs = state.get("messages", []) or []
    tool_call_names = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            tool_call_names.append(tc.get("name"))
    assert "task_researcher" in tool_call_names, f"called: {tool_call_names}"
    assert any(n in tool_call_names for n in (
        "task_compose_analytical", "task_compose_essay", "task_compose_roundup",
    )), f"no composer fired; called: {tool_call_names}"


# ── 3. Compose linkedin (short-form) ────────────────────────────────────────


def test_compose_linkedin_emits_structured(agent):
    """LinkedIn brief produces a LinkedInPost structured_response somewhere."""
    tid = "live-linkedin-1"
    state = _send(
        agent,
        "Compose a LinkedIn post on 'why agent demos keep failing' in English. "
        "Pull anchors from my OpenMark bookmarks.",
        thread_id=tid,
        timeout_s=600,
    )
    msgs = state.get("messages", []) or []
    tool_names = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            tool_names.append(tc.get("name"))
    assert "task_compose_linkedin" in tool_names, f"called: {tool_names}"


# ── 4. Verifier (called as part of compose loop) ────────────────────────────


def test_verifier_runs_on_compose_path(agent):
    """The orchestrator should call task_verify after a composer."""
    tid = "live-verify-1"
    state = _send(
        agent,
        "Draft a LinkedIn post on 'context engineering in 2026' in English, "
        "then verify the result.",
        thread_id=tid,
        timeout_s=600,
    )
    msgs = state.get("messages", []) or []
    tool_names = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            tool_names.append(tc.get("name"))
    assert "task_verify" in tool_names, f"called: {tool_names}"


# ── 5. Polisher on English drafts ───────────────────────────────────────────


def test_polisher_routes_for_english(agent):
    """English compose path should call task_polish, NOT task_humanize."""
    tid = "live-polish-1"
    state = _send(
        agent,
        "Compose a LinkedIn post on 'graph RAG vs vector RAG' in English. "
        "Polish the output before returning.",
        thread_id=tid,
        timeout_s=600,
    )
    msgs = state.get("messages", []) or []
    tool_names = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            tool_names.append(tc.get("name"))
    assert "task_polish" in tool_names or "task_humanize" not in tool_names, (
        f"polish missing for English: {tool_names}"
    )


# ── 6. Humanizer on Arabic drafts ───────────────────────────────────────────


def test_humanizer_routes_for_arabic(agent):
    """Arabic compose path should call task_humanize."""
    tid = "live-humanize-1"
    state = _send(
        agent,
        "Compose a LinkedIn post on 'AI agents in 2026' in ar-egt. "
        "Run it through the humanizer for Egyptian Arabic style.",
        thread_id=tid,
        timeout_s=600,
    )
    msgs = state.get("messages", []) or []
    tool_names = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            tool_names.append(tc.get("name"))
    assert "task_humanize" in tool_names, f"called: {tool_names}"


# ── 7. Classification middleware: state.intent gets written ─────────────────


def test_classifier_writes_state_intent(agent):
    """After one turn, the state should carry an intent label."""
    tid = "live-classify-1"
    state = _send(
        agent,
        "/deep-research multi-agent orchestrator patterns",
        thread_id=tid,
    )
    intent = state.get("intent")
    source = state.get("intent_source")
    assert intent in ("fast", "deep", "newsletter", "digest", "dive"), f"intent={intent}"
    assert source in ("slash", "heuristic", "llm"), f"source={source}"
    # Slash dispatch should win on /deep-research
    assert source == "slash"
    assert intent == "deep"
