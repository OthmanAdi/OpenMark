# Newsletter Mission — Live Research Dossier

**Started:** 2026-05-15
**Goal:** ship a self-hosted public-link newsletter + LinkedIn post composer that Ahmad can embed an input field for on his own site. No paid services. Skills-driven. Self-hosted or Docker.

**Rule for this doc:** write findings as they land. Do not start building. Each section is appended to, never deleted. "Do later" lists at the bottom of each section.

---

## §1 Existing newsletter skills (surveyed 2026-05-15)

Five `.claude/skills/openmark-newsletter*/SKILL.md` files exist. Already cover the major output shapes.

| Skill | Purpose | Best use |
|---|---|---|
| `openmark-newsletter` | Punchy analytical newsletter, 600-900 words, weekly-recap angle | Default newsletter format |
| `openmark-newsletter-essay` | Long-form thesis, narrative paragraphs, 600-900 words | "Make the case for X" |
| `openmark-newsletter-thread` | 250-400 word LinkedIn / X post, ONE link, phone-readable | **PRIMARY for the public-link goal** — LinkedIn |
| `openmark-newsletter-roundup` | Bucketed categorical recap, scannable | "What dropped this week" |
| `openmark-newsletter-comparison` | A vs B vs C table-driven | Tool / model comparisons |

### Strengths already in place
- **House style locked.** No "leveraging / delve / synergies." Sentence fragments OK. Light swearing OK.
- **Citation discipline strict.** "Never invent a URL. Only cite URLs that appeared in a tool result."
- **Format rules strict.** H1 only, blank lines between every numbered item, mandatory `## Sources cited` section that triggers auto-export.
- **Tool sequence per skill.** Each skill names its retrieval recipe (semantic + community / category + LinkedIn + WebFetch top 5).
- **Voice calibration tables.** Generic → Ahmad's voice mapping in every skill — a built-in style guide for the LLM.

### Gaps for the public-link mission
- **No public-facing endpoint.** Skills run inside Claude Code (Ahmad's terminal) or in the Gradio chat. Neither is reachable from an external website.
- **No schema validation.** Skills emit free-form markdown. No Pydantic / JSON-schema guard before output, so the auto-export heuristic is the only correctness check.
- **No retry-on-bad-output loop.** If the LLM emits the wrong shape, the user sees the bad draft.
- **No structured LinkedIn-post object.** The thread skill outputs markdown with prose. A LinkedIn-API-ready object would have `hook`, `body_paragraphs[]`, `closer`, `link`, `hashtags[]`, `word_count`, `sources[]` as typed fields.

### Do later — newsletter skills
- Add a `openmark-newsletter-linkedin` skill that ONLY produces the LinkedIn-post shape (no markdown wrapping, no H1, no `## Sources` — those break native LinkedIn rendering).
- Extract the voice-calibration tables and citation rules into ONE shared skill that the other five `extend` from (deduplication).

---

## §2 LangChain structured output + validation — research notes

(to be filled in §2 below as context7 / web hits come in)

### Do later — schemas
- Define `LinkedInPost`, `NewsletterEssay`, `NewsletterRoundup`, `NewsletterComparison`, `NewsletterThread` as Pydantic v2 models in `openmark/agent/schemas.py`.
- Wire `response_format=<Model>` on `create_agent` per mode, or per-skill via dispatcher.

---

## §3 Agent-blueprint repo (OthmanAdi/agent-blueprint)

(filled in §3 below from `github_repo_intel`)

### Do later
- After dossier compiles, extract concrete patterns from agent-blueprint that map onto the OpenMark newsletter agent.

---

## §4 Self-hostable web serving options (free)

(filled in §4 below from web + repo research)

### Do later
- Pick a single transport: Gradio share / FastAPI minimal form / Streamlit / Chainlit / Open WebUI / etc.

---

## §5 Sub-agent architecture — DECIDED 2026-05-15 v2

User reversed prior position. Now: sub-agents are Phase 1, not future-work. Runtime = LangChain `deepagents` package. Pattern = orchestrator + researcher + composer + humanizer + polisher + verifier. Decision tree (`ref_decision_tree.md`) Q5 = YES on specialization. Full details in MISSION.md §3 + §5 + §7.

### Resolved
- ONE orchestrator → MANY specialists. Not "one agent only".
- `deepagents.create_deep_agent` provides task tool, write_todos, skills auto-loader.
- Each sub-agent gets least-privilege tool slicing (Manus principle 2).

### Still to verify in Phase 1 D1
- `deepagents` composes cleanly with our langchain 1.2.18 + langgraph 1.1.10 + Azure Foundry path.
- Skill auto-loader picks up `.claude/skills/` directory without changes.
- Sync/async mix works under Gradio's event loop.

---

## §6 Open questions — see MISSION.md §13 for the v2 list

v1 questions Q1-Q8 were superseded by user's second message. v2 questions are Q9-Q13 and live in MISSION.md §13.

## §7 New: LLM-neutral mandate (2026-05-15 v2)

User: *"doesnt matter wich llm the agent uses, this shit has to work with any llm from foudnary."*

Implication: schema enforcement must use `ToolStrategy(...)` (works with any tool-calling model), NOT `ProviderStrategy(...)` (provider-locked). Sub-agent factories accept `model=` arg defaulting to `build_executor()` which already honours `AGENT_PROVIDER` + `AZURE_DEPLOYMENT_EXECUTOR`.

### Do later
- Phase 1 D1 — measure schema-pass rate per Foundry model on 20 saved briefs.
- If a specific Foundry model fails ToolStrategy >20% of the time, add ProviderStrategy fallback for that model only.

## §8 New: skill self-authoring (2026-05-15 v2)

User: *"create skills to use it it self."*

Pattern: `write_skill(name, frontmatter, body)` tool, sandboxed to `.claude/skills/agent-generated/`. After write call `reload_skills()`. New skill becomes loadable same turn via existing `load_skill` mechanism. Cap 5/session.

### Do later
- Phase 1 D8 — implement the tool + cap + UI display in "Created this session" subsection.

## §9 New: UI tools/skills panel (2026-05-15 v2)

User: *"see in the UI the tools my agent has, the skill my agent create."*

Two new Gradio tabs ("Agent Tools" + "Agent Skills"). Auto-populated from `ALL_TOOLS` and `list_skills()`. "Created this session" subsection in Skills tab is the agent-self-authoring view.

### Do later
- Phase 1 D10 — wire the tabs into `openmark/ui/app.py`.

## §10 New: LinkedIn paste export (2026-05-15 v2)

User: *"export special formats for me to copy and paste in linkedin."*

Three rendering targets from one `LinkedInPost` Pydantic object:
- `to_markdown(post)` — current chat UI auto-save flow
- `to_linkedin_plaintext(post)` — no #, no md, URL footnotes, line-break tuning
- `to_linkedin_html(post)` — paste-as-rich-text variant

### Do later
- Phase 1 D9 — implement in `openmark/composer/export.py`.

## §11 New: humanizer-semitic integration (2026-05-15 v2)

Local copies of all 4 skill bodies are in `humanizer/`. Drop them into `.claude/skills/humanizer-*/SKILL.md` (or symlink). Existing OpenMark skill loader picks them up automatically — frontmatter format matches.

Caveat: humanizer-semitic is Arabic + Hebrew only. Question Q9 tracks whether Ahmad's audience is English (default) or Semitic. Composer skips humanizer for English drafts; the new `openmark-polisher` skill (Phase 1 D5) handles English-side AI-tell removal.

### Do later
- Phase 1 D5 — write `openmark-polisher` skill for English drafts.
- Phase 1 D7 — drop humanizer-semitic skills into `.claude/skills/`.

## §12 New: email rollout infra (2026-05-15 v2, deferred to Phase 3)

Accounts to register NOW so Phase 3 is unblocked:
1. Resend (3k emails/month free)
2. Brevo (300/day free)
3. Cloudflare (DNS + Tunnel)

DNS records to plan: SPF, DKIM, DMARC for Ahmad's domain.

### Do later
- Register accounts. Save API keys in `.env`.
- Provision DNS records once domain is decided.
- Build `email_html(post)` Jinja2 template in `openmark/composer/export.py`.
