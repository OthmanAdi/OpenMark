# OpenMark — Next

Honest list of what still needs to happen. One item at a time, in order.

## Mission this session
Ingest fresh bookmarks + fresh LinkedIn into Neo4j, then keep the app running so Ahmad can pull material for a newsletter. Nothing else.

## Active goals (this session, in order)

1. Commit current working tree (Phase 0-2 agent refactor + Live Add tab + Neo4j driver singleton + URL canonicalization).
2. Diff `C:\Users\oasrvadmin\Downloads\favorites_5_14_26.html` against Neo4j. Inject only new URLs.
3. Fetch fresh LinkedIn saved posts (needs `li_at` cookie from Ahmad).
4. Merge new LinkedIn posts into Neo4j via the same injector.
5. Run `python openmark/ui/app.py` on 127.0.0.1:7860 and keep it up.
6. Capture concrete agent + UI complaints surfaced while Ahmad uses it. Write them here, no fixing.

## Post-injection gaps (run before next agent session)
- **SIMILAR_TO edges** for the 1,782 new bookmarks + 1,856 LinkedIn posts not built. The injector scripts only rebuilt tag co-occurrence. Need `neo4j_store.build_similar_to_edges()` over the full graph.
- **Louvain communities** not rerun. New nodes have no `community_id`. `search_by_community` will miss them.
- **Category heuristic** put 1,083/1,782 HTML items in "News & Articles" because the bare-URL `_guess_category` fallback fires for anything not in the small domain map. A later LLM-categorization pass would lift that into the right buckets.
- **LinkedIn URL duplicates**: LinkedIn nav URLs carry an `updateEntityUrn` tracking query param that changes per fetch. The diff against Neo4j found 0 matches even though 1,260 LinkedIn posts already existed in DB, so we now have 1,260 old + 1,856 new entries that overlap heavily by `urn:li:activity:N` (the stable ID). Fix: strip `?updateEntityUrn=...` from LinkedIn URLs in `normalize_item`, then dedupe by stripped URL. Adds `activity_urn` property for stable joins.

## Strategic decision 2026-05-14 — OpenMark v2 direction

After studying three reference projects (`migraven-openfang-mvp`, `migRaven-MAX-thClaws`, the Dario starter kit) and re-reading `openmark/mcp/server.py`, the decision is:

**Stop building an in-house agent. Make OpenMark a tool, let Claude Code be the agent.**

### Why
- An MCP server already exists at `openmark/mcp/server.py` with 6 tools, registered via `.mcp.json`. Claude Code auto-loads it when run from this repo. Half the integration work is already done.
- Claude Code already has every agent capability Ahmad listed: visible thinking, plan/research sub-agents, WebFetch, structured tool calls, persistent sessions, skills system, Read/Write/Edit for newsletter drafts.
- The LangChain v1 `create_agent` middleware approach committed today gives ~30% of those abilities. Reaching parity (Phases 3-6 of `research/agent_upgrade/04_phased_plan.md`) is multi-day work to recreate what Claude Code does natively.
- The user said "UI is uncomfortable but it doesn't fucking matter for now" — so a separate branded chat UI is not the bottleneck.

### What dies
- `openmark/agent/graph.py`, `openmark/agent/llms.py` — superseded by Claude Code. Keep for reference, mark deprecated.
- `openmark/agent/schemas.py` Pydantic types — KEEP. Reused in MCP tool returns.
- `openmark/ui/app.py` Chat tab — repurpose into "Open in Claude Code" deep link, OR delete.
- `research/agent_upgrade/04_phased_plan.md` Phases 3-6 — superseded. Phases 0-2 stand (they shipped useful Pydantic typing and tool refactor).

### What gets built (priority order)

**Block A — MCP completeness (1 day)**
1. Port the 6 tools missing from MCP: `find_by_tag`, `explore_tag_cluster`, `find_by_domain`, `search_linkedin`, `search_youtube`, `run_cypher` (read-only).
2. Switch MCP tool returns from markdown strings to structured Pydantic shapes (`BookmarkHit`, `ToolResult`). Re-use `agent/schemas.py`.
3. Add 3 new tools the newsletter workflow needs:
   - `search_by_date_range(from_iso, to_iso, query?)` — needs `created_at` backfill on existing nodes.
   - `find_recent(days, query?)`.
   - `get_bookmark_full(url)` — full doc_text + tags + similar URLs for synthesizer context.

**Block B — Claude Code skills for OpenMark (0.5 day)**
Create `.claude/skills/openmark/` directory with:
- `newsletter-compose.md` — house style, structure, citation rules, "always pull from OpenMark first, WebFetch second."
- `topic-research.md` — pattern: search_bookmarks → graph_expand winners → WebFetch top 3 URLs → synthesize.
- `weekly-digest.md` — what did Ahmad save in the last 7 days, clustered by topic, summarized.

**Block C — graph hygiene (0.5 day, can run in background)**
- One-shot script to rebuild SIMILAR_TO edges for the 1,782 newly injected bookmarks + 1,856 newly injected LinkedIn posts.
- One-shot script to re-run Louvain over the full 13k+ graph.
- Backfill `created_at` from source timestamps where available (LinkedIn has them; Edge HTML export typically does not — defer those).

### What stays exactly as-is
- Neo4j store + pplx-embed pipeline.
- Gradio UI: Search, Stats, Graph 3D, +Add tabs. Useful as a browsing surface independent of the agent.
- Live injection in `openmark/pipeline/injector.py`.

### Out of scope (for now, not never)
- Reranker (bge-reranker-v2-m3) — useful but additive, not blocking.
- Eval harness — re-add only after Block A ships, measure recall@20 of MCP tools.
- Heterogeneous LLM routing for Claude Code path — moot, Claude Code already picks Sonnet/Opus.

## New requirements from 2026-05-14 chat (commit later, in this order)

### Web crawler tools (for the agent, not the embedding pipeline)
Goal: agent has fast page-content tools so it can compare what Ahmad saved vs what's actually on the open web today. DO NOT crawl into the embedding pipeline yet — these are runtime tools only.

Shortlist of OSS / free-tier options to evaluate when this block lands:
- **Crawl4AI** (Python, MIT) — async crawler with markdown extraction, JS rendering via Playwright. Built for LLM ingestion. Good default.
- **Firecrawl** (Python/TS, AGPL self-host or paid SaaS) — `/scrape` + `/crawl` + `/search` endpoints, very clean markdown. Free tier limited but generous.
- **Trafilatura** (Python, GPL) — pure-text article extraction, no browser. Fast and lightweight, no JS support.
- **Playwright** wrappers (Python) — full browser, slower but handles JS-only sites; use as fallback when Trafilatura/Crawl4AI miss content.
- **Serper / Brave Search API / Tavily** — paid web search APIs for the "fresh URLs" angle, none truly free but Brave has a free tier.

Implementation plan (deferred):
1. Add `openmark/web/` module with `fetch_clean(url)` and `web_search(query, n)` functions wrapping Crawl4AI + Tavily/Brave.
2. Expose as 2 new MCP tools: `web_fetch_clean(url)` and `web_search(query, n)`.
3. New skill `openmark-compare-web`: pulls top 5 from OpenMark + top 5 from `web_search`, shows side-by-side coverage gaps.

### Agent classifier (shipped 2026-05-14)
The Gradio agent now classifies every first turn into one of `fast / deep / newsletter / digest / dive` via a cheap tier (`gpt-5-mini` by default, settable to `model-router` via `AZURE_DEPLOYMENT_CLASSIFIER`). The mode hint is injected into the system prompt so the executor (gpt-5.3-codex high) tailors effort. Visible in stdout: `[classifier] mode=...`.

### Model router (deferred)
If Ahmad sets `AZURE_DEPLOYMENT_CLASSIFIER=model-router` in `.env`, the classifier call routes through Azure Foundry's `model-router` deployment instead of `gpt-5-mini`. Foundry then picks the smallest viable model per call. No code change needed beyond the env var.

### GitHub push (planned for end of this session)
- Confirm `.gitignore` excludes: `.env`, `data/chroma_db/`, `data/*.json` (large dumps), `nsfw_*.json`, `drafts/`, model caches.
- Do NOT push: `.env`, any LinkedIn cookie, the raw `linkedin_saved*.json` dumps, the HTML export.
- Code, MCP server, skills, scripts, docs, NEXT.md — all fair game to push.

### Stale repo state
- `README.md` still says ChromaDB + dual-store + gpt-4o-mini. Reality: Neo4j only, codex-5-3. Rewrite when Phase 3 lands.
- `task_plan.md`, `findings.md`, `progress.md`, `_purge_now.py`, `nsfw_*.py`, `scan_nsfw.py`, `ai_teaching_bookmarks.md` — cruft. Decide keep/delete in a cleanup pass.
- `app.py` at repo root — unclear if needed alongside `openmark/ui/app.py`.

### UI complaints to gather (filled during use)
- (record concrete issues here as Ahmad hits them — do not pre-fill)

## Decision log
- 2026-05-14: README outdated, do not believe it. Trust `openmark/config.py` and the agent files.
- 2026-05-14: Neo4j driver is a module-level singleton. Do NOT close it in tool calls.
- 2026-05-14: URL canonical form is `lower().rstrip("/")` — applied in `normalize_item`.
