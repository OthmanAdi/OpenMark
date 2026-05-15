# OpenMark Newsletter Mission — Profile

**Started:** 2026-05-15
**Owner:** Ahmad (OthmanAdi)
**Status:** research complete, build pending user sign-off
**Companion files in this dir:**
- `00_findings.md` — running notebook of everything sighted
- `ref_serving.md`, `ref_api_agent.md`, `ref_planner_executor.md`, `ref_system_prompts.md`, `ref_framework_guide.md`, `ref_production_principles.md`, `ref_tool_system.md` — local copies of the agent-blueprint references

---

## 1. Mission, in one line

A self-hosted, public-link composer that turns Ahmad's OpenMark bookmarks into beautiful LinkedIn posts (and longer newsletter formats), embedded as an input field on his own website, with zero paid SaaS in the loop.

## 2. Non-negotiable constraints

| # | Constraint | Source |
|---|---|---|
| C1 | Self-hosted or Docker. No SaaS lock-in. | User: "evrying must be self hosted or on docker" |
| C2 | Free transport. LLM cost: ambiguous — Ahmad has a local Ollama/Hermes path already wired (`AGENT_PROVIDER=local`). Default Phase 1 to local-first unless **Q0** below says otherwise. | User: "without having to pay anytihng" |
| C3 | Skills-driven. The composer IS a skill stack, not bespoke code. | User: "evrying has to be runing via skills" |
| C4 | One agent first, perfect it, then add sub-agents. | User: "before adding sub agents i need the perfect newsletter composing agent" |
| C5 | Best-in-class schema validation. | User: "best schema in langchai nand schema valdiatio nand tricks" |
| C6 | No tracing infrastructure. | User (prior turn): "i dont need tracing for my agent" |

These are non-negotiable. Every architectural choice in §5 is tested against this list.

## 3. Current state of the codebase (2026-05-15, post-commit `2693b4c`)

**Agent stack:**
- LangChain 1.2.18 + LangGraph 1.1.10 + langchain-openai 1.2.1 + pydantic 2.13.
- `openmark/agent/graph.py` builds a `create_agent` with 8 middleware layers (slash pre-load, skill catalogue, live stats + mode router, TodoList, Summarization, ToolRetry, tool event bus, ModelCallLimit).
- 21 tools registered in `openmark/agent/tools.py`. 15 graph-RAG tools + 6 web tools (Tavily + DDG + GitHub + Reddit + Crawl).
- `openmark/agent/schemas.py` defines `BookmarkHit`, `ToolResult`, `QuickAnswer`, `Report`, `LinkedInPost`-shaped types do NOT yet exist.
- `openmark/agent/skills.py` discovers `.claude/skills/openmark-*/SKILL.md` files at process start.

**Newsletter skills already shipped:**
- `openmark-newsletter` — analytical, 600-900w, weekly-recap angle
- `openmark-newsletter-essay` — single thesis, 600-900w
- `openmark-newsletter-thread` — 250-400w LinkedIn / X post, ONE link
- `openmark-newsletter-roundup` — bucketed categorical recap
- `openmark-newsletter-comparison` — A vs B table-driven
- House style locked: no "leveraging / delve / synergies", sentence fragments OK, citation discipline strict
- Auto-export hooks: any output with `# Title` + `## Sources cited` gets saved by the Gradio UI

**Knowledge base:**
- Neo4j Aura graph: **13,831 bookmarks** · 5,469 tags · 19 categories · 0 communities (Louvain blocked — GDS plugin missing)
- pplx-embed 1024-dim, vector index ready
- LinkedIn dedup hygiene step must follow every fresh ingest

**Web delivery surface:**
- Gradio chat UI on `127.0.0.1:7860`. NOT publicly reachable.
- MCP server `openmark/mcp/server.py` exposing 6 tools. Used by Claude Code in Ahmad's terminal, not by the public.
- No FastAPI / SSE / public endpoint exists yet.

**This session's recent commits:**
- `2693b4c` — warm-up + embed LRU + stats TTL + smarter mode router
- `f1c5fc1` — Grok routing on Azure Foundry
- `9419b4b` — web research stack (Tavily + DDG + GitHub + Reddit + Crawl)

## 4. Gap analysis — what's missing to ship the mission

| Gap | Why it matters | Fix surface |
|---|---|---|
| G1. No public HTTP surface | Ahmad's website cannot embed today | `ref_serving.md` / `ref_api_agent.md` recipe |
| G2. No `LinkedInPost` Pydantic schema | Composer outputs free-text markdown, can't ship a JSON contract | New shapes in `schemas.py` |
| G3. No `response_format` enforced composer | Bad shape goes through unchecked | LangChain `ToolStrategy(LinkedInPost)` pattern |
| G4. Chat agent and composer agent are fused | Chat-agent's clarifying questions break composer-only schema enforcement | Split into two endpoints |
| G5. System prompt rewrites the dynamic `MODE:` section | Breaks Azure / OpenAI prefix cache across sessions | Move dynamic context into a user-message preamble (Manus KV-cache principle) |
| G6. No rate limiting / cost cap for public endpoint | Public link = automated abuse risk | `RateLimiter` from `ref_serving.md` |
| G7. No session multiplexing | Same `thread_id` collides if two people hit the link | UUID-per-request or cookie-issued session ID |
| G8. No deploy target chosen | Docker / nginx / Caddy / fly.io / Hetzner all open | One pick in §7 |
| G9. No LinkedIn-API output path | "Post directly to LinkedIn" requires OAuth + posting permission | Open question — see §8 |
| G10. Skill catalogue addendum on every wrap_model_call | Stable text but wasteful network bytes; harmless under prefix-cache | Defer; not blocking |

## 5. Architecture proposal — single agent first, sub-agents later

### Phase 1: ONE composer agent, two endpoints. Hardened.

```
                   ┌──────────────────────────────────┐
ahmad.site/post →  │ FastAPI /compose POST            │
                   │ - rate limit (RPM + daily cap)   │
                   │ - issue X-Session-ID cookie      │
                   │ - SSE stream                     │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────────┐
                   │ ComposerAgent  (LangChain v1)    │
                   │ - LLM = build_executor()         │
                   │   honours AGENT_PROVIDER         │
                   │   (local Hermes / Azure Grok)    │
                   │ - response_format=               │
                   │     ToolStrategy(LinkedInPost)   │
                   │   OR ProviderStrategy(...) for   │
                   │   local 8B models if they fail   │
                   │   ToolStrategy retries (Q0/Q7)   │
                   │ - tools: search_semantic,        │
                   │   find_by_tag, search_linkedin,  │
                   │   get_bookmark_full,             │
                   │   web_fetch (no chat tools)      │
                   │ - tiny tool-loadout (≤8 tools)   │
                   │ - skill: openmark-newsletter-    │
                   │     linkedin (NEW)               │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────────┐
                   │ Pydantic validator → JSON ←──┐   │
                   └──────────────┬─────── retry ─┘   │
                                  │   (LangChain      │
                                  │    auto-handles   │
                                  │    on bad shape)  │
                                  ▼
                   ┌──────────────────────────────────┐
                   │ Renderer → HTML + plaintext      │
                   │ (Jinja2, no React needed)        │
                   └──────────────────────────────────┘
```

**Why one agent first:** Manus's data (`ref_production_principles.md` §1) — 85% per-step accuracy yields 20% success at 10 steps. Every agent we add multiplies failure. Get the single composer to 95% reliability before adding a planner / verifier.

**Why response_format here is safe (when it wasn't before):** the chat agent in `graph.py` intentionally avoids `response_format` because clarifying questions don't fit the schema. The composer endpoint does NOT chat. It takes a frozen `{topic, format, time_window, source_filters}` input, fans out searches, then emits ONE typed object. No clarification possible by design.

**Caveat for local LLM path:** small models (Hermes-3-8B, Qwen-2.5-7B) follow `ToolStrategy` schemas less reliably than grok-4.3 or codex. If Q0/Q7 lands on local-default, Phase 1 step 1 must also test `ProviderStrategy` (native structured output API) AND a fallback "best-effort markdown + manual parse" path. Measure during Phase 1 eval.

### Phase 2: planner + executor + verifier (when single agent caps out)

Triggered only if §6 fails-mode evidence accumulates. Pattern from `ref_planner_executor.md`:

```
/compose POST
    │
    ▼
PlannerAgent (no tools)
    Output: list[Subtask] — research, write, polish
    │
    ▼
WorkerAgents (parallel, tool-restricted)
    - Researcher: search_* tools, WebFetch
    - Composer: response_format=LinkedInPost, no search tools
    - Polisher: response_format=LinkedInPost, no tools — voice rewrite only
    │
    ▼
VerifierAgent (read-only)
    - Cites match tool outputs? URLs not invented?
    - Voice rules respected? Word count in range?
    - Returns pass/fail per subtask
    │
    ▼
LinkedInPost JSON
```

**Do NOT build Phase 2 until Phase 1 has logged enough real bad outputs to prove what's failing.** This is the trap Manus calls out: building orchestration before measuring per-step accuracy.

## 6. Schema design — the contract

Per-format schemas. The `sources` length differs by format (thread = 1 anchor; essay/roundup = many).

```python
# openmark/agent/schemas.py — additions

class PostSource(BaseModel):
    url: str = Field(description="MUST appear in a tool result this turn.")
    title: str
    note: str = Field(default="", description="One short phrase on what this source provides.")

class LinkedInPost(BaseModel):
    """Phone-readable LinkedIn post, 250-400 words, ONE link in body, no hashtags."""
    hook: str = Field(min_length=20, max_length=140,
                      description="Opening claim, 6-10 words, NO question marks.")
    body_paragraphs: list[str] = Field(min_length=4, max_length=6,
                                       description="4-6 short paragraphs. Each 1-3 sentences.")
    closer: str = Field(min_length=10, max_length=180,
                        description="One sentence. Quotable. NOT a question.")
    anchor_url: str = Field(description="The ONE link in the body. Must be in sources.")
    sources: list[PostSource] = Field(min_length=1, max_length=1,
                                      description="Exactly one anchor for the thread format.")
    word_count: int = Field(ge=180, le=420)
    voice_check: Literal["pass", "warn"] = "pass"

class NewsletterEssay(BaseModel):
    """Single-thesis essay, 600-900 words."""
    title: str = Field(max_length=64)
    thesis: str
    opening_paragraph: str
    sections: list["EssaySection"] = Field(min_length=3, max_length=5)
    counter: str = Field(description="The counter-argument paragraph; mandatory.")
    closing_paragraph: str
    sources: list[PostSource] = Field(min_length=5, max_length=8)
    word_count: int = Field(ge=550, le=950)

class EssaySection(BaseModel):
    heading: str = Field(description="Sub-claim phrasing, NOT a topic name.")
    body_markdown: str

# Same pattern for NewsletterRoundup, NewsletterComparison, NewsletterAnalytical.
# Source bounds vary per format and are the single most important guardrail.
```

**Validation moves:**
- `min_length` / `max_length` enforce word count and structure shape.
- `Literal["pass", "warn"]` channels the LLM's self-grade into a typed field.
- `sources[].url` MUST be cross-checked against `state["seen_urls"]` in an `@after_model` middleware — if a URL is in the schema but never appeared in any tool result, raise and retry. (This is the Anthropic-grade citation guarantee the skills already promise.)

**LangChain hook (`ref` context7 docs verified 2026-05-15):**
```python
agent = create_agent(
    model=composer_llm,
    tools=composer_tools,
    response_format=ToolStrategy(LinkedInPost),   # default handle_errors=True
    system_prompt=COMPOSER_PROMPT,
)
result = agent.invoke({"messages": [{"role": "user", "content": brief}]})
post = result["structured_response"]   # validated Pydantic instance
```

On validation failure, LangChain re-prompts the model with the Pydantic error message and the model retries. This is the validation trick the user asked about — it's free, built in.

## 7. Self-hosted public-surface options (free, comparison)

All options self-hostable. All free of recurring SaaS cost.

| Option | What it gives | Pros | Cons | Verdict |
|---|---|---|---|---|
| **FastAPI + SSE (per `ref_serving.md`)** | HTTP API + streaming, paired with a 30-line vanilla-JS embed | Matches Ahmad's own playbook, Docker-ready, nginx-safe (proxy_buffering off) | Need to write the HTML embed | **Phase 1 pick** |
| Gradio share link | `share=True` creates a free 72h Cloudflare tunnel | One line of code | URL rotates every 72h, not embeddable as input on his site, ugly iframe | Reject |
| Streamlit | Pretty UI, free Streamlit Cloud | Quick to ship | Streamlit Cloud is SaaS (violates C1), self-host is heavier | Reject |
| Chainlit | Chat-first UI, MIT, Docker-ready | Built for LLM apps, multi-user | Heavier than needed for a single input field | Defer |
| Open WebUI | Full chat interface, MIT, Docker | Polished UX | Designed as a destination site, not an embed | Defer |
| Cloudflare Tunnel (cloudflared) + local FastAPI | Free public HTTPS for `127.0.0.1:8000` | Zero infra; survives Ahmad's home IP | Tied to Cloudflare account but tunnel is free tier | **Phase 1 deploy hack** |
| Docker on Hetzner / Contabo / Oracle Free Tier | True self-host on a €4 box (Oracle free tier = €0) | Real production | More moving parts | **Phase 2 deploy** |
| fly.io free tier | Container + auto-deploy | Trivial Dockerfile push | Free tier shrinking; risk of paywall later | Defer |

**Recommendation:** FastAPI + SSE on the local box, exposed publicly via Cloudflare Tunnel for v1. Move to a Hetzner / Oracle Free Tier box when traffic justifies it. Both are zero-cost.

## 8. Open questions for Ahmad (gather before building)

Marked Q so you can answer with `Q1: ...` etc.

- **Q0 — BLOCKING.** Composer LLM: (a) local-only (Hermes / Qwen via Ollama, zero per-request cost, weaker schema adherence); (b) Azure-only (grok-4.3 / codex, costs per call, strong schema adherence); (c) local-default with Azure escape hatch for hard formats like essay? This determines whether Phase 1 needs a `ProviderStrategy` fallback path AND whether the public-link can be opened to strangers (Azure cost = liability) or stays token-gated to friends (local cost = electricity).
- **Q1.** LinkedIn posts only, or also the longer newsletter formats (essay, roundup, comparison) on day one?
- **Q2.** Does the embed live on Ahmad's existing personal site, or a fresh subdomain?
- **Q3.** Public open URL (rate-limited), token-gated (he hands out keys to friends), or session-cookied (anyone with the link)?
- **Q4.** Output delivery: stream in the browser only, also email to Ahmad's inbox, or post directly via LinkedIn API (OAuth scope = `w_member_social`)?
- **Q5.** Storage of drafts: SQLite (already used in `openmark/history.py`), Postgres, flat files in `drafts/`, Notion?
- **Q6.** Should the composer be allowed to call WebFetch (slower, fresher) or restrict to OpenMark bookmarks only (faster, deterministic)?
- **Q7.** Which LLM should the composer use? Current chat agent runs grok-4.3 on Azure Foundry. Cheaper option: gpt-5-mini for the composer, grok-4.3 for harder long-form (essay).
- **Q8.** Brand the surface as "Ahmad" / "OpenMark" / something new?

## 9. Phased plan

### Phase 0 — frozen scope (this session)
- [x] Optimize the existing chat agent (commit `2693b4c`)
- [x] Survey existing newsletter skills
- [x] Pull agent-blueprint references into `research/newsletter_mission/`
- [x] Write this mission doc
- [ ] Ahmad reviews + answers Q1-Q8

### Phase 1 — single composer endpoint (next session, ~1 day)
1. Add `LinkedInPost` + per-format Pydantic schemas in `openmark/agent/schemas.py`.
2. Build `openmark/composer/` module:
   - `compose.py` — single agent, `response_format=ToolStrategy(<FormatModel>)`, ≤8 tools
   - `validators.py` — `seen_urls` cross-check middleware
   - `formats.py` — registry mapping `format_name → (Pydantic class, skill name, tool subset)`
3. Add `openmark-newsletter-linkedin` SKILL.md (LinkedIn-native, no `# Title`, no `## Sources cited`, no markdown).
4. Build `openmark/server/app.py` FastAPI app per `ref_serving.md`:
   - POST `/compose` with `{topic, format, time_window?, source_filters?}` body, SSE stream
   - `SessionStore` (in-memory) + `RateLimiter` (20 rpm, $2/day cap)
   - CORS allowlist for Ahmad's website domain
5. Dockerfile + `docker-compose.yml` + nginx config (`proxy_buffering off`).
6. Write `frontend/embed.html` — a 60-line vanilla-JS input field + streaming output box Ahmad can iframe or paste.
7. Cloudflare Tunnel script `scripts/run_tunnel.ps1`.

### Phase 2 — production hardening (later)
- Move `SessionStore` to SQLite (reuse `openmark/history.py` pattern) so server restarts don't drop drafts.
- Per-session daily cost cap (currently per-IP).
- Add per-format eval harness: 20 saved briefs → run composer → score for citation validity, voice rules, word-count compliance.
- Move from Cloudflare Tunnel to a real box (Oracle Free Tier or Hetzner CX11).
- LinkedIn OAuth posting (only if Q4 answer = "post directly").

### Phase 3 — planner + executor split (only if Phase 1 logs failures)
- Implement the three-agent split in `ref_planner_executor.md`.
- Verifier becomes the new home for the `seen_urls` cross-check (which started life as middleware in Phase 1).
- Gate on measured Phase 1 failure rate ≥ 15% — otherwise this is yak shaving.

### Phase 4 — sub-agents and orchestration (out-of-scope until Phase 3 lands)
- Multi-format compose (one brief → LinkedIn + newsletter + thread in one fan-out).
- Scheduled compose (weekly auto-digest every Monday).
- LinkedIn-API direct posting with HITL approval per `ref_human-in-the-loop.md`.

## 10. Tricks worth banking (from research)

These are reusable across the in-app chat agent AND the composer endpoint.

1. **KV-cache stability (Manus principle 1).** Move every dynamic value out of the system prompt into a stable user-message preamble. The current `inject_live_stats` middleware (shipped in commit `2693b4c` earlier this session) rewrites the system prompt with `MODE: fast`, which breaks Azure prefix-cache stability across sessions. **This is unfinished business** — refactor to: system prompt = pure-static, the dynamic stats + mode line go into the FIRST user message as a `<context>...</context>` block. Treat as a Phase 2 cache-optimization task on the chat agent; the composer endpoint should bake this in from day one.
2. **Mask, don't remove (Manus principle 2).** Keep `ALL_TOOLS` stable. Use system-prompt instructions to restrict which tools are allowed in a given mode rather than dropping tools from the list.
3. **Tool-count budget < 30.** OpenMark sits at 21. Composer should run with ≤8 (search_semantic, find_by_tag, search_linkedin, get_bookmark_full, search_by_community, optionally web_fetch). Tighter loadout = better picks.
4. **Action namespace prefixes.** Tools follow `<area>_<verb>` so the model can reason about what's available without scanning descriptions. OpenMark mostly does this already; future tools should match.
5. **Deterministic JSON ordering.** When serializing dynamic context, use `sorted(...)` on lists / dicts. Different orderings of the same logical state kill cache hits.
6. **Verifier nodes after risky steps.** Cite-check is the highest-value verifier. Implement once, reuse in both chat and composer paths.
7. **Output schema is strict, retry is free.** LangChain's `ToolStrategy(...)` with `handle_errors=True` (default) re-prompts the model with the Pydantic error message on validation failure. Costs one round-trip; pays for itself when the model would otherwise emit garbage.
8. **Skills extend skills.** The five existing newsletter skills duplicate house-style + citation rules. Extract a `openmark-newsletter-base` skill the others `extends` (filesystem-level convention: a `base.md` symlinked-in section). Saves edit-five-files-on-every-tweak.

## 11. Out of scope for this mission (deliberately)

- Replacing the existing Gradio chat UI. It stays as a separate destination.
- Refactoring the entire `graph.py` middleware stack. Optimization work this session was enough.
- Building a fancy React frontend. Vanilla JS + fetch + SSE is sufficient.
- Tracing / observability dashboards (user explicit).
- LinkedIn OAuth posting until Q4 is answered.
- Substituting Claude Code as the composer (decision in `NEXT.md` 2026-05-14). The public-link requirement makes it unsuitable — Claude Code has no public URL.
- Reranker addition (bge-reranker-v2-m3). Possible later, additive.

## 12. Decision log

- 2026-05-15: Mission is a PUBLIC-LINK composer, not a Claude-Code-driven workflow. This reverses the 2026-05-14 entry in `NEXT.md`. Justification: Ahmad needs an input field on his own website; Claude Code cannot be embedded there.
- 2026-05-15: LangGraph stays. `ref_framework_guide.md` confirms it as the production Python pick. LangChain 1.x `create_agent` (already in use) compiles to LangGraph internally.
- 2026-05-15: Phase 1 is ONE agent with `response_format=ToolStrategy(LinkedInPost)`. Phase 2 split into planner/executor/verifier gated on measured failure rate.
- 2026-05-15: Transport is FastAPI + SSE per Ahmad's own `agent-blueprint/references/serving.md`. Deployed via Cloudflare Tunnel for v1.

---

**End of mission profile.** Next action is Ahmad answering Q1-Q8 in §8. Nothing builds until those answers exist.
