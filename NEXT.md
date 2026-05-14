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

## Deferred (do AFTER this session, not now)

### Agent — Phase 3 and beyond
- Phase 3: heterogeneous LLM routing (planner / executor / synthesizer split). `llms.py` factories already exist, only executor is wired in `graph.py`.
- Phase 4: plan-execute-synthesize graph (`graph_v2.py`). `AGENT_MODE=v2` flag exists in `config.py` but does nothing.
- Reranker (bge-reranker-v2-m3) before synthesis.
- Citation validator edge — block answers that cite URLs not in `seen_urls`.
- Phase 6 eval harness — recall@20 on 20 hand-curated queries.

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
