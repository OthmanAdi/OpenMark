# LangChain `deepagents` — sub-agent runtime

**Source:** docs.langchain.com/oss/python/deepagents/* (queried via context7 2026-05-15).

## What it is
A pre-built framework on top of LangGraph + LangChain v1 `create_agent`. Bundles:
- `create_deep_agent()` — orchestrator factory
- `write_todos` tool — the planning surface the user explicitly wants
- `task()` tool — main agent delegates to a registered sub-agent
- Skills module — auto-loads SKILL.md files and exposes them to the system prompt
- Subagents middleware — exposes the task tool, manages sub-agent registry
- Summarization middleware
- Backends — InMemoryStore, StoreBackend, CompositeBackend, file system, Daytona sandbox
- `async_subagents` — remote Agent Protocol servers (for scale)

## Install
```bash
pip install deepagents
```

## Minimal pattern
```python
from deepagents import create_deep_agent

# Sub-agents
researcher = create_agent(
    model="<any-foundry-model>",
    tools=[search_semantic, find_by_tag, search_linkedin, web_fetch],
    system_prompt="You are a research specialist...",
)

composer = create_agent(
    model="<any-foundry-model>",
    tools=[],  # no search — researcher already gathered
    response_format=ToolStrategy(LinkedInPost),
    system_prompt="You compose; you do not research...",
)

humanizer = create_agent(
    model="<any-foundry-model>",
    tools=[],
    system_prompt="<humanizer skill body loaded here>",
)

verifier = create_agent(
    model="<any-foundry-model>",
    tools=[],
    response_format=ToolStrategy(VerificationReport),
    system_prompt="You only check; you do not write...",
)

# Orchestrator
agent = create_deep_agent(
    model="<any-foundry-model>",
    subagents={
        "researcher": researcher,
        "composer":   composer,
        "humanizer":  humanizer,
        "verifier":   verifier,
    },
    skills=["/skills/"],  # path or store namespace
)
```

## Delegation principles (per docs)
- **Default: 1 sub-agent.** Don't pre-decompose. "Research X" → one researcher, not three.
- **Parallelize only for explicit comparisons.** "Compare A vs B vs C" → 3 parallel sub-agents.
- **Bias toward token efficiency.** One broad sub-agent beats three narrow ones.

## Workflow loop
1. Orchestrator runs `write_todos` to plan.
2. Saves user request to `/research_request.md` via filesystem backend.
3. Calls `task(subagent_type=..., description=...)` per todo.
4. Sub-agent runs autonomously, returns one final report.
5. Orchestrator synthesizes findings, consolidates citations.
6. Writes `/final_report.md`.
7. Verifies — reads back the request, confirms coverage.

## Key trick — skill loading
`skills=["/skills/"]` loads every SKILL.md from that path and appends the catalogue to the orchestrator's system prompt. Skill bodies are NOT loaded until the model calls `load_skill(name)` — progressive disclosure, same pattern OpenMark already uses in `openmark/agent/middleware.py`.

## What this changes for OpenMark
- The current `openmark/agent/graph.py` (single `create_agent` + 8 middleware) becomes the **researcher sub-agent**, or stays as the chat agent unchanged. Decision: see MISSION.md §5.
- A new `openmark/composer/` module hosts the `create_deep_agent` orchestrator.
- Existing 21 tools partition across sub-agents by least-privilege.
- Existing 10 OpenMark skills work as-is — `deepagents.skills` module is compatible with the `.claude/skills/` directory layout.
- `humanizer-semitic` skills drop into `.claude/skills/` and become callable by `/humanizer-ar-msa` etc., same as the OpenMark skills today.

## Caveats
- `deepagents` is LangChain-team-maintained, ~young package. API may shift.
- Async-only flow for some features (`abefore_agent`). Gradio handles sync; need to verify mixed sync/async path.
- The skill catalogue trick uses `runtime.store` — verify it composes with the existing `MemorySaver()` checkpointer.
