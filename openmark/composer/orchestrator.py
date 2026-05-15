"""
Newsletter / LinkedIn composer orchestrator.

Built on `deepagents.create_deep_agent`. LLM-neutral (uses `build_executor()`
which already honours AGENT_PROVIDER + AZURE_DEPLOYMENT_EXECUTOR — Foundry
codex / Grok / local Hermes all work without code change).

Architecture per `research/newsletter_mission/MISSION.md` v2 §5:

    Orchestrator (write_todos, task, load_skill, write_skill, files)
        ├── researcher                 (heavy tools, no schema)
        ├── composer-linkedin          (response_format=LinkedInPost)
        ├── composer-thread            (alias of linkedin)
        ├── composer-essay             (response_format=NewsletterEssay)
        ├── composer-roundup           (response_format=NewsletterRoundup)
        ├── composer-comparison        (response_format=NewsletterComparison)
        ├── composer-analytical        (response_format=NewsletterAnalytical)
        ├── humanizer                  (Arabic / Hebrew rewrite)
        ├── polisher                   (English AI-tell scrub)
        ├── verifier                   (returns VerificationReport)
        └── skill-author               (calls write_skill)

UI integration: we pass the existing `tool_event_middleware` and
`OpenMarkSkillMiddleware` through `middleware=[]` so the Gradio chat UI's
live tool-card stream keeps working. Calls to `task(subagent_type=...)` show
up as cards just like any other tool call.
"""

from __future__ import annotations

import time
from typing import Any

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from openmark import config
from openmark.agent.llms import build_executor
from openmark.agent.middleware import (
    OpenMarkSkillMiddleware,
    log,
    slash_skill_loader,
    tool_event_middleware,
)
from openmark.agent.tools import write_skill, warm_up as _warm_up_tools
from openmark.composer.subagents import build_all_subagents


_ORCHESTRATOR_PROMPT = """You are the OpenMark Composer Orchestrator — Ahmad's personal newsletter / LinkedIn writer.

Your job is to plan a compose run, delegate work to specialist sub-agents, then return
a single grounded output (LinkedInPost / NewsletterEssay / NewsletterRoundup / NewsletterComparison / NewsletterAnalytical).

You have these primitives:
  - `write_todos`     — plan first. Always.
  - `task`            — delegate to a sub-agent by name. See list below.
  - `load_skill`      — load an OpenMark / humanizer / agent-generated skill body
  - `write_skill`     — sandboxed skill author (max 5 per session)
  - filesystem        — `read_file`, `write_file`, `ls`, `glob`, `grep` if helpful

## Sub-agents (delegate via `task(subagent_type="<name>", description="<brief>")`)

  researcher                 — gather anchors + citations
  composer-linkedin          — LinkedIn / short post (also: composer-thread)
  composer-essay             — long-form thesis essay
  composer-roundup           — categorical news recap
  composer-comparison        — A vs B table-driven
  composer-analytical        — analytical newsletter (the default for "newsletter on X")
  humanizer                  — Arabic / Hebrew rewrite (ar-msa / ar-egt / ar-shami / he)
  polisher                   — English AI-tell scrub
  verifier                   — grade output, returns VerificationReport
  skill-author               — bake a recurring prompt into a reusable skill

## The compose loop (follow exactly)

1. PLAN. Call `write_todos` with the steps below tailored to the request.
2. RESEARCH. `task("researcher", "Find anchors for <topic>. Pull from OpenMark first, web second if thin. Return JSON anchor list.")`
3. COMPOSE. Pick the right composer sub-agent for the requested format. Pass the topic, angle, language, AND the researcher's anchor list verbatim.
4. STYLE PASS.
   - If language == "en": `task("polisher", "<draft>")`.
   - Else if language in (ar-msa / ar-egt / ar-shami / he): `task("humanizer", "Language: <lang>. Draft: <draft>")`.
5. VERIFY. `task("verifier", "Grade this output against seen_urls=<list>. Output a VerificationReport.")`.
6. If `verifier.overall_passed == False`, retry the composer ONCE with `verifier.fix_instructions`. Then re-verify.
7. RETURN. Emit the final composer output verbatim. The export layer renders it for the user.

## Citation discipline
- Every URL in the final output MUST appear in a researcher tool result this turn. Verifier enforces.
- If the researcher came back thin, ASK the user for a narrower topic. Do NOT invent.

## Language
- Default: English. The user names another language explicitly via `language=` in the brief.
- Supported: en, ar-msa, ar-egt, ar-shami, he.

## Output
- Return the final composer Pydantic instance. No extra prose. The harness will render it.
- If a verifier check failed twice in a row, return the best draft AND a one-line note explaining which check failed.
"""


def build_composer_orchestrator(
    *,
    model: Any = None,
    checkpointer: Any | None = None,
) -> Any:
    """
    Build the composer orchestrator. LLM-neutral via build_executor().

    Args:
        model: Optional pre-built BaseChatModel. Defaults to build_executor()
               so AGENT_PROVIDER / AZURE_DEPLOYMENT_EXECUTOR stays in charge.
        checkpointer: Optional checkpointer. Defaults to MemorySaver() so
                      each compose run has its own thread_id state.

    Returns:
        A compiled deep-agent graph. Call `.invoke({"messages": [...]}, config=...)`
        or `.stream(...)` exactly like a langchain create_agent.
    """
    t0 = time.time()
    _warm_up_tools()
    log.info(f"[composer] warm_up done in {round((time.time()-t0)*1000)}ms")

    llm = model if model is not None else build_executor()
    provider = (getattr(config, "AGENT_PROVIDER", "azure") or "azure").lower()
    log.info(
        f"[composer] LLM-neutral build: provider={provider} "
        f"deployment={getattr(config, 'AZURE_DEPLOYMENT_EXECUTOR', 'n/a')}"
    )

    subagents = build_all_subagents()
    log.info(f"[composer] registering {len(subagents)} sub-agents: "
             + ", ".join(s["name"] for s in subagents))

    agent = create_deep_agent(
        model=llm,
        # The orchestrator's own tools — deepagents ALSO ships write_todos,
        # task, ls, read_file, write_file, edit_file, glob, grep, execute.
        # We add write_skill so the skill-author sub-agent has it, and so the
        # orchestrator can directly bake a recipe without a sub-agent hop.
        tools=[write_skill],
        system_prompt=_ORCHESTRATOR_PROMPT,
        subagents=subagents,
        middleware=[
            # Beautiful UI: reuse the existing slash pre-loader so /<skill>
            # commands work in the Compose tab too.
            slash_skill_loader,
            # Skill catalogue + load_skill tool — same progressive disclosure
            # the chat agent uses.
            OpenMarkSkillMiddleware(),
            # Tool event bus → live UI cards. THIS is what keeps the Compose
            # tab visually consistent with the Chat tab.
            tool_event_middleware,
        ],
        checkpointer=checkpointer if checkpointer is not None else MemorySaver(),
    )
    log.info(f"[composer] orchestrator compiled in {round((time.time()-t0)*1000)}ms")
    return agent


def ask_compose(agent: Any, brief: str, *, thread_id: str = "compose-default") -> dict:
    """
    Convenience helper for one-shot composer runs.

    Args:
        agent: from build_composer_orchestrator()
        brief: free text with topic, format hint, language hint.
               e.g. "Compose a LinkedIn post on RAG retrieval tradeoffs. Language: en."
        thread_id: checkpointer thread id; default is shared.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": brief}]}, config=cfg)
    messages = result.get("messages", [])
    final_text = ""
    if messages:
        c = getattr(messages[-1], "content", "") or ""
        if isinstance(c, list):
            final_text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in c
            )
        else:
            final_text = c
    structured = result.get("structured_response")
    return {
        "answer": final_text,
        "structured": structured,
        "tool_calls": sum(
            len(getattr(m, "tool_calls", []) or []) for m in messages
        ),
    }
