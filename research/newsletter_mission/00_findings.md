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

## §5 Sub-agent architecture (deferred)

User explicit: "before adding sub agents i need the perfect newsletter composing agent tools." So sub-agents are §5 / future-work, not phase 1.

### Do later
- Decide: is it ONE agent with response_format dispatch, OR a planner-executor pair (deep-research → newsletter-composer), OR a fan-out of micro-agents per section?
- Reference patterns: agent-blueprint (§3) and LangGraph Send fanout (already in this repo's skill set).

---

## §6 Open questions for Ahmad (collect, don't act)

- Is the public-link goal LinkedIn posts ONLY, or also the longer newsletter formats?
- Does the embed live on Ahmad's existing personal site or a fresh subdomain?
- Auth model: open URL (rate-limited), token-gated, or session-cookied?
- Output delivery: stream in the browser, email to Ahmad, post directly to LinkedIn API?
- Storage of drafts: SQLite (already used for chat history), Postgres, files, Notion?
