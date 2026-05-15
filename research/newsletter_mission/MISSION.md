# OpenMark Newsletter Mission — Profile (v2, 2026-05-15)

**Owner:** Ahmad (OthmanAdi)
**Status:** research complete · architecture locked · build pending Ahmad's go-signal on Phase 1 step list
**Companion files in this dir:**
- `00_findings.md` — running notebook
- `ref_decision_tree.md` — 5-question agentic-design-pattern decision tree (ML Mastery)
- `ref_deepagents.md` — LangChain `deepagents` runtime, the chosen backbone
- `ref_serving.md`, `ref_api_agent.md`, `ref_planner_executor.md`, `ref_system_prompts.md`, `ref_framework_guide.md`, `ref_production_principles.md`, `ref_tool_system.md` — agent-blueprint references
- `humanizer/humanizer-ar-msa.md`, `humanizer-ar-egt.md`, `humanizer-ar-shami.md`, `humanizer-he.md` — local copies of humanizer-semitic skill bodies

This is **v2**. v1 was committed at `2e6b3a9` and is now superseded; the file is overwritten in place. Decision diff is in §13.

---

## 1. Mission, in one line

Make the OpenMark agent dramatically smarter, give it sub-agents that specialize, expose tools and skills (including the agent's self-created ones) in the UI, and ship a LinkedIn-paste-ready export path. Newsletter email rollout is documented but deferred.

## 2. Non-negotiable constraints (v2 update)

| # | Constraint | Source |
|---|---|---|
| C1 | **LLM-neutral.** Works with ANY Azure Foundry model. No code path hard-coded to grok-4.3 or codex. | User: "doesnt matter wich llm the agent uses, this shit has to work with any llm from foudnary" |
| C2 | **Best 10x output.** Quality bar is "best I can achieve right now." Reflection / verifier is mandatory, not optional. | User: "best 10x output for me" |
| C3 | **Self-hosted / Docker.** No SaaS lock-in. | User: prior turn |
| C4 | **Skills-driven AND skill-creating.** Agent uses existing skills AND creates new skills it can reference immediately. | User: "create skills to use it it self" |
| C5 | **Sub-agent middleware with full inheritance.** Sub-agents get tools, web search, skills, AND access to the orchestrator's TodoList. | User: "sub agent middlwaer... use tools and web seach and skills and get the todolsit from the main composer agent" |
| C6 | **UI surface for tools + skills.** Gradio must show the tools the agent has AND the skills it created. | User: "see in the UI the tools my agent has, the skill my agent create" |
| C7 | **LinkedIn paste export.** A copy-paste-ready format, not just markdown. | User: "export special formats for me to copy and paste in linkedin" |
| C8 | **humanizer-semitic integration.** The agent can call humanizer skills mid-compose. | User: linked the repo + "use this skill" |
| C9 | **No tracing infrastructure.** | User: prior turn |

## 3. Architectural pattern — confirmed via decision tree

Walked the ML Mastery 5-question tree (`ref_decision_tree.md`) against this mission:
- Q1 path known? PARTIAL (5 formats, varying topic)
- Q2a fixed per-format? YES → sequential workflow per format
- Q2b tools? YES → Tool-Use is foundational
- Q3 structure articulable? YES → Planning with ReAct inside steps
- Q4 quality > speed? YES → add Reflection
- Q5 specialization OR scale? YES → Multi-Agent Specialist

**Result: Planner → Multi-Agent Specialist (parallel where independent) → Reflection.**

Implementation: LangChain `deepagents` framework. See `ref_deepagents.md`.

**Source provenance note:** the 5-question tree above was extracted from a mirror at `geekfence.com` because the original `machinelearningmastery.com` page returned HTTP 403 to WebFetch. The mirror may paraphrase. Pattern conclusion holds regardless of phrasing; treat the exact question wording as approximate.

## 4. Current state of the codebase (post-commit `2e6b3a9`)

**Agent stack:**
- LangChain 1.2.18 + LangGraph 1.1.10 + langchain-openai 1.2.1 + pydantic 2.13.
- `openmark/agent/graph.py` — single `create_agent` with 8 middleware layers, 21 tools.
- `openmark/agent/skills.py` — discovers `.claude/skills/openmark-*/SKILL.md`. Progressive disclosure already implemented.
- 10 OpenMark skills shipped: 5 newsletter formats + fast-search + deep-research + weekly-digest + bookmark-dive + repo-research.
- Optimizations from earlier this session: warm-up, embed LRU, stats TTL, slash/heuristic mode router (commit `2693b4c`).

**Knowledge base:**
- Neo4j Aura: 13,831 bookmarks · 5,469 tags · 19 categories · 0 communities (Louvain blocked, GDS missing).
- pplx-embed 1024-dim, vector index on `bookmark_embedding`.

**Web delivery surface:**
- Gradio chat UI at `127.0.0.1:7860`. Not publicly reachable.
- MCP server `openmark/mcp/server.py` exposing tools to Claude Code.
- No FastAPI / SSE / public endpoint yet.

**Newsletter skills:** five formats already mature (thread / essay / roundup / comparison / analytical). House style, citation discipline, voice tables all locked. The agent-side glue is what's missing.

## 5. Target architecture — deep-agent orchestrator

```
                  ┌─────────────────────────────────────────────┐
       USER ─────▶│ Gradio Chat OR /compose HTTP endpoint       │
                  └────────────────────┬────────────────────────┘
                                       ▼
                  ┌─────────────────────────────────────────────┐
                  │ ORCHESTRATOR  (create_deep_agent)           │
                  │ - write_todos: plans the run                │
                  │ - task(): delegates to sub-agents           │
                  │ - skills auto-exposed (progressive)         │
                  │ - LLM-neutral via build_executor()          │
                  └────────────────────┬────────────────────────┘
                                       │
                ┌──────────┬───────────┼──────────┬──────────┐
                ▼          ▼           ▼          ▼          ▼
        ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐ ┌──────────┐
        │RESEARCHER│ │ COMPOSER │ │HUMANIZER│ │POLISHER│ │ VERIFIER │
        │          │ │          │ │         │ │        │ │          │
        │ tools:   │ │ tools:   │ │ tools:  │ │tools:- │ │ tools:-  │
        │ semantic │ │ NONE     │ │ NONE    │ │skill   │ │ NONE     │
        │ tag      │ │ resp_fmt=│ │ skill:  │ │load    │ │ resp_fmt=│
        │ linkedin │ │ ToolStr- │ │humani-  │ │only    │ │ Verifi-  │
        │ youtube  │ │ ategy(   │ │zer-     │ │        │ │ cation   │
        │ web_fet- │ │ Linked-  │ │semitic  │ │        │ │ Report   │
        │ ch/sea-  │ │ InPost)  │ │ if Q9   │ │        │ │          │
        │ rch      │ │          │ │ enabled │ │        │ │          │
        └──────────┘ └──────────┘ └─────────┘ └────────┘ └──────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────────┐
                  │ EXPORT LAYER  (no LLM)                      │
                  │ - markdown for chat UI auto-save            │
                  │ - linkedin_plaintext (no #, no md, line     │
                  │   breaks tuned for LinkedIn editor)         │
                  │ - linkedin_html for paste-as-rich-text      │
                  │ - email_html (Jinja2 template, deferred)    │
                  └─────────────────────────────────────────────┘
```

**Why this passes the decision tree AND the user's constraints:**
- Planner: orchestrator's `write_todos` step.
- Multi-agent specialist: researcher / composer / humanizer / verifier with least-privilege tool sets (`ref_planner_executor.md` Manus principle 2).
- Reflection: verifier sub-agent is the reflection step.
- LLM-neutral: every sub-agent uses `build_executor()` which honours `AGENT_PROVIDER` and `AZURE_DEPLOYMENT_EXECUTOR` env. Swap the model, no code change.
- Sub-agents inherit skills via `deepagents.skills`. Inherit web access via shared tool registry. Inherit TodoList via orchestrator's state (already part of `deepagents` runtime).
- humanizer-semitic plugged in as a sub-agent skill.
- Verifier is the citation cross-check and voice-rule audit, gated on `response_format=ToolStrategy(VerificationReport)` so the orchestrator can branch on `passed: bool`.

## 6. Schema design — the contracts

Add to `openmark/agent/schemas.py`:

```python
class PostSource(BaseModel):
    url: str = Field(description="MUST appear in a researcher tool result this turn.")
    title: str
    note: str = Field(default="")

class LinkedInPost(BaseModel):
    """Phone-readable LinkedIn post, 250-400 words, ONE link in body."""
    hook: str = Field(min_length=20, max_length=140)
    body_paragraphs: list[str] = Field(min_length=4, max_length=6)
    closer: str = Field(min_length=10, max_length=180)
    anchor_url: str
    sources: list[PostSource] = Field(min_length=1, max_length=1)
    word_count: int = Field(ge=180, le=420)
    language: Literal["en", "ar-msa", "ar-egt", "ar-shami", "he"] = "en"
    humanizer_applied: bool = False
    voice_check: Literal["pass", "warn"] = "pass"

class NewsletterEssay(BaseModel):
    title: str = Field(max_length=64)
    thesis: str
    opening_paragraph: str
    sections: list["EssaySection"] = Field(min_length=3, max_length=5)
    counter: str   # mandatory by skill rules
    closing_paragraph: str
    sources: list[PostSource] = Field(min_length=5, max_length=8)
    word_count: int = Field(ge=550, le=950)
    language: Literal["en", "ar-msa", "ar-egt", "ar-shami", "he"] = "en"

# Same pattern for NewsletterRoundup, NewsletterComparison, NewsletterAnalytical.

class VerificationReport(BaseModel):
    cite_check: Literal["pass", "fail"]
    cite_fail_reason: str = Field(default="")
    voice_check: Literal["pass", "fail"]
    voice_fail_reason: str = Field(default="")
    word_count_check: Literal["pass", "fail"]
    word_count_fail_reason: str = Field(default="")
    overall_passed: bool
    fix_instructions: str = Field(default="", description="If overall_passed=false, exact instructions for the composer to retry.")
```

**Why `response_format=ToolStrategy(...)`** (per LangChain docs verified 2026-05-15):
- Works with ANY tool-calling model (LLM-neutral, satisfies C1).
- `handle_errors=True` is the default — on Pydantic validation failure the agent retries with the error message stuffed back into the prompt. This is the "schema trick" the user asked about.
- For models that don't tool-call well (rare on Foundry), fall back to `ProviderStrategy(LinkedInPost)` which uses provider-native structured output.

## 7. Tools — partitioned by sub-agent (least-privilege, Manus principle 2)

| Sub-agent | Tools | Skills accessible |
|---|---|---|
| Orchestrator | `write_todos`, `task`, `load_skill`, NONE else | all openmark-* + humanizer-semitic |
| Researcher | `search_semantic`, `find_by_tag`, `explore_tag_cluster`, `search_by_community`, `search_by_category`, `search_linkedin`, `search_youtube`, `find_by_domain`, `find_recent`, `search_by_date_range`, `get_bookmark_full`, `graph_expand`, `web_search`, `web_fetch`, `web_extract`, `reddit_search`, `github_repo_intel` | openmark-deep-research, openmark-fast-search, openmark-repo-research |
| Composer | NONE (gets researcher's findings via task return) | openmark-newsletter, openmark-newsletter-thread, openmark-newsletter-essay, openmark-newsletter-roundup, openmark-newsletter-comparison |
| Humanizer | NONE | humanizer-ar-msa, humanizer-ar-egt, humanizer-ar-shami, humanizer-he |
| Polisher | NONE | (style polish skill, see §10 do-now item D5) |
| Verifier | `web_fetch` (optional, for offline URL audit) | (verifier skill, see §10 do-now item D6) |

Tool count per sub-agent ≤ 17 (researcher is the heaviest, still under the 30-tool sweet spot).

## 8. UI surface — Gradio tools/skills panel (NEW)

C6 requires UI visibility into tools + skills. Two new Gradio tabs:

**Tab: "Agent Tools"**
- One row per registered tool, grouped by sub-agent.
- Columns: name, sub-agent, description, args schema (collapsed JSON), call count this session.
- Refresh button to re-read `ALL_TOOLS`.

**Tab: "Agent Skills"**
- One row per `.claude/skills/*/SKILL.md`.
- Columns: short_name, type, description, source ("openmark" / "humanizer-semitic" / "agent-created"), path, last-used timestamp.
- Section: "Created this session" — skills the agent wrote during its run, with one-click promote-to-disk.

The "agent-created skills" path is new (C4). See §10 D8.

## 9. Skill self-authoring — the agent creating new skills (NEW)

C4 says the agent can create skills it uses immediately. Pattern:
- Add a `write_skill(name, frontmatter, body)` tool. Writes to `.claude/skills/agent-generated/{name}/SKILL.md`.
- Drop it into the orchestrator's tool set.
- After write, call `reload_skills()` (already exists in `openmark/agent/skills.py`).
- The new skill becomes loadable on the SAME turn via the existing `load_skill` mechanism.

**Safety:** sandbox writes to `.claude/skills/agent-generated/` only. Disallow overwrite of `openmark-*` or `humanizer-*`. Validate frontmatter before write. Cap session-creation count at 5 to prevent runaway skill spam.

**When to fire:** the orchestrator notices it's run the same prompt-shape twice in a session and decides to bake it as a skill. Triggered by a system-prompt instruction, not hard-coded.

## 10. Phased plan

### Phase 0 — done this session
- [x] Commit `2693b4c` — agent optimizations (warm-up + LRU + stats TTL + mode router)
- [x] Commit `2e6b3a9` — v1 research dossier
- [x] Pull 7 agent-blueprint references locally
- [x] Pull 4 humanizer-semitic skill bodies locally
- [x] Walk the ML Mastery 5-question decision tree
- [x] Confirm `deepagents` framework as the runtime
- [x] Rewrite MISSION.md (this file) for v2

### Phase 1 — DO NOW (this session if Ahmad green-lights, otherwise next session)
**D0.** Upgrade `langchain` 1.2.18 → 1.3.x and `langgraph` 1.1.10 → 1.2.x (required by `deepagents 0.6.1`). Re-run the agent build and verify all 8 existing middleware layers still compose. Roll back if `inject_live_stats` / `OpenMarkSkillMiddleware` / `slash_skill_loader` break (likely minor API drift, not blocking).
**D1.** Install `deepagents`. Verify `from deepagents import create_deep_agent` succeeds. PyPI confirmed real at version 0.6.1 (checked 2026-05-15).
**D2.** Define schemas in `openmark/agent/schemas.py`: `PostSource`, `LinkedInPost`, `NewsletterEssay`, `NewsletterRoundup`, `NewsletterComparison`, `NewsletterAnalytical`, `VerificationReport`.
**D3.** Build sub-agent factories in `openmark/agent/subagents.py`: `build_researcher()`, `build_composer(format_name)`, `build_humanizer()`, `build_verifier()`. Each is a thin `create_agent()` call wrapped in least-privilege tool slicing.
**D4.** Add `openmark/agent/orchestrator.py` using `create_deep_agent(subagents=...)`. Skill auto-load wired to `.claude/skills/`.
**D5.** Write a new skill `openmark-polisher` — style sweep + AI-tell removal for English drafts.
**D6.** Write a new skill `openmark-verifier` — citation cross-check + voice rule audit instructions.
**D7.** Drop the humanizer-semitic skills into `.claude/skills/` (sync from the GitHub repo or use the `npx skills add` install). Verify the existing `openmark/agent/skills.py` picks them up.
**D8.** Add `write_skill` tool + sandbox path + 5/session cap. Wire to orchestrator.
**D9.** Build the export layer `openmark/composer/export.py`:
  - `to_markdown(post)` — current chat UI flow
  - `to_linkedin_plaintext(post)` — strip headings, expand inline links to footnote URLs at end, line-break tuning for LinkedIn editor
  - `to_linkedin_html(post)` — for paste-as-rich-text
**D10.** New Gradio tabs: "Agent Tools" + "Agent Skills" per §8.
**D11.** Wire a "Compose" button that runs the orchestrator end-to-end with a frozen brief and renders the final output via the export layer.

### Phase 2 — public-link endpoint (next session after Phase 1 lands)
- FastAPI + SSE per `ref_serving.md`.
- Cloudflare Tunnel for v1 public reach.
- Session store, rate limiter, CORS.
- `<iframe>` embed snippet for Ahmad's website.

### Phase 3 — email rollout (deferred but mapped, see §11)

### Phase 4 — measurement + LinkedIn-API direct post
- Per-format eval harness: 20 saved briefs, run composer, score for schema compliance, citation validity, voice rules, word-count.
- LinkedIn OAuth posting (`w_member_social` scope) with HITL approval per `ref_human-in-the-loop.md` (not yet downloaded, deferred fetch).

## 11. Email rollout — accounts + env required (deferred, planned now)

User: *"the fuckign newslter has to roll out with a beautifu lcustomizbale tempalte to the emaisl."* Email = Phase 3. What we need to provision now so we're not blocked later:

| Need | Free option | Env / account required |
|---|---|---|
| Subscriber list capture | Self-hosted, single `subscribers` SQLite table; the website form POSTs to a public `/subscribe` endpoint | Already in scope; no external account |
| Double-opt-in confirmation email | Free SMTP via Brevo (300/day), Mailjet (200/day), Resend (3k/month) free tiers | `BREVO_API_KEY` OR `RESEND_API_KEY`. Free tier is enough up to ~3k subscribers/month. |
| Outbound newsletter send | Same SMTP provider. For >3k sends/month, switch to Postal (self-hosted SMTP) on a VPS | Same key. Postal needs a VPS + DNS DKIM/SPF records. |
| HTML template engine | Jinja2 + MJML (open source, MIT) for responsive email | None — pip install. |
| Inbox-deliverability checks | mail-tester.com (free 3/day) | None |
| Bounce/complaint handling | Provider's webhook → `/webhook/email-event` endpoint we write | Whichever provider above |
| Open / click tracking | Provider built-in OR self-host (Plausible has email analytics in beta) | Same key |

**DNS records to add to Ahmad's domain when ready:**
- SPF: `v=spf1 include:<provider-spf> ~all`
- DKIM: per-provider CNAME pair
- DMARC: `v=DMARC1; p=quarantine; rua=mailto:dmarc@<domain>`

**Accounts to register NOW (free, no card):**
1. Resend (3000 emails/mo free) — easiest API.
2. Brevo (300/day free) — fallback.
3. Cloudflare account (for the Tunnel in Phase 2, and DNS hosting if not already there).

Do these in 10 minutes total. Document API keys in `.env` with the same `load_dotenv(override=False)` pattern OpenMark already uses.

## 12. LinkedIn rollout — what's needed (deferred until output quality proves out)

To post directly via the LinkedIn API:
- LinkedIn Developer App (free): https://www.linkedin.com/developers/apps
- OAuth 2.0 scope `w_member_social` (Share on LinkedIn).
- Token storage in `.env` per environment. Token rotates every 60 days for LinkedIn — needs a refresh flow.
- HITL approval before each post (use `HumanInTheLoopMiddleware`, already in langchain-middleware skill).

Until then: copy-paste workflow via `to_linkedin_plaintext` export.

## 13. Open questions for Ahmad

- **Q9.** humanizer-semitic supports Arabic + Hebrew only. Are your LinkedIn posts in English (most likely), Arabic, Hebrew, or all of the above? The composer needs to know which humanizer to invoke (or skip).
- **Q10.** Should the orchestrator be the EXISTING chat agent (extended with `task` tool) or a NEW orchestrator at a separate route? Two are valid — extending keeps one entry point; separate keeps Phase 1 isolated.
- **Q11.** Skill self-authoring (§9) sounds powerful but risky. Confirm: 5 skills/session cap, sandboxed to `agent-generated/`, no overwrite of curated skills. Acceptable?
- **Q12.** Output language detection: pass through composer (model guesses), OR explicit `language=` field in the user brief? **Provisionally decided** — schema in §6 carries `language: Literal[...] = "en"`. Brief field overrides; no guessing. Confirm.
- **Q13.** "Best 10x output" — measurable how? Suggest: each composer run is scored by the verifier on 4 dimensions (cite_check, voice_check, word_count_check, schema_check). Track per-format pass rate over time. Aim for >90% pass rate per format on the first try.

## 14. Diff from v1 (committed at `2e6b3a9`)

| v1 said | v2 says | Why |
|---|---|---|
| One agent, response_format on composer endpoint | Multi-agent via `deepagents`: researcher + composer + humanizer + polisher + verifier | Decision tree Q5 = YES (specialization), and user explicitly asked for sub-agents |
| Q0 = pick composer LLM (local/Azure/hybrid) | LLM-neutral via `build_executor()`. No pick required. | User: "doesnt matter wich llm" |
| Cloudflare Tunnel + FastAPI + SSE was Phase 1 | Public endpoint moved to Phase 2; Phase 1 = upgrade the in-app agent + UI tabs + export layer | User: "what is imrptoant for now is the optmaizaiotn of my agent, the tools and abiltiies" |
| No mention of skill self-creation | §9 adds `write_skill` tool, sandbox, 5/session cap | User: "create skills to use it it self" |
| No mention of UI tools/skills panel | §8 adds two Gradio tabs | User: "see in the UI the tools my agent has, the skill my agent create" |
| No humanizer integration | §7 + §13 Q9 — humanizer sub-agent with humanizer-semitic skills | User: "use this skill https://github.com/OthmanAdi/humanizer-semitic" |
| Email infra unmentioned | §11 maps accounts + DNS + env vars to register NOW | User: "what tehcnillgiies or fuckign accoutns or env set up i will ened" |
| No export layer | §10 D9 — markdown + linkedin_plaintext + linkedin_html + email_html (last is Phase 3) | User: "export special formats for me to copy and paste in linkedin" |

## 15. Decision log

- 2026-05-15 v2: Switched runtime to LangChain `deepagents` package. Justification: built-in `task` tool + `write_todos` + skills auto-loader = exactly what C5 + C6 require; rolling our own would duplicate `deepagents`.
- 2026-05-15 v2: Multi-agent specialist pattern confirmed via ML Mastery decision tree Q5 = YES.
- 2026-05-15 v2: LLM-neutral confirmed. `build_executor()` already honours `AGENT_PROVIDER` + `AZURE_DEPLOYMENT_EXECUTOR`. No code change needed at the model edge; the sub-agent factories accept a model arg defaulting to the executor.
- 2026-05-15 v2: Public-link endpoint moved from Phase 1 to Phase 2. Phase 1 is upgrading the in-app agent first.
- 2026-05-15 v2: humanizer-semitic added to the skill set. Used optionally based on Q9 answer.
- 2026-05-15 v2: Email infra accounts/env documented in §11. Register accounts now so Phase 3 is unblocked.
- 2026-05-15 v2: `deepagents` PyPI presence verified — version 0.6.1 exists, transitive deps add `langchain>=1.3.0`, `langgraph>=1.2.0`, `langchain-anthropic`, `langchain-google-genai`, `anthropic`. OpenMark currently sits on `langchain==1.2.18` + `langgraph==1.1.10`. A minor framework bump is the very first Phase 1 step (D0).
- 2026-05-15 v2: Decision-tree source is the geekfence.com mirror, not the MLM original (403 on WebFetch). Pattern conclusion holds regardless; exact wording is approximate.

---

**End of mission profile v2.** Next action is Ahmad green-lighting Phase 1 (§10 D1-D11) OR answering Q9-Q13 first. Whichever order is fine.
