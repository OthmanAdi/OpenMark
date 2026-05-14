# Phased Upgrade Plan — OpenMark Retrieval Agent

Goal: agent retrieves the right bookmarks PERFECTLY, with grounded citations, controlled reasoning cost, parallel multi-angle search.

Architecture target (refined with Azure constraints from `02_azure_reasoning.md`):
```
Planner       gpt-5.3-codex via Responses API, reasoning.effort=high
   ↓ Send fan-out
Executor      gpt-5-mini via Chat Completions, reasoning_effort=low
              (NOTE: NOT "minimal" — that disables parallel tool calls)
              [parallel multi-tool]
   ↓
Reranker      bge-reranker-v2-m3 local (no LLM)
   ↓
Synthesizer   gpt-5.3-codex via Responses API,
              reasoning.effort=medium, text.verbosity=low,
              with_structured_output(Answer, strict=True)
   ↓
Validator     URL-grounding guard edge (Python only, no LLM)
```

**Key Azure gotchas (verified May 2026):**
- All `gpt-5.x-codex` models are **Responses-API only** → must set `use_responses_api=True` in `AzureChatOpenAI`
- `gpt-5.3-codex` is limited-access — verify your Foundry deployment exists first via `aka.ms/OAI/gpt53codexaccess`
- `reasoning_effort="minimal"` **disables parallel tool calls** → use `"low"` for the executor
- Reasoning models reject `temperature`, `top_p`, `max_tokens` → use `max_completion_tokens` / `max_output_tokens`
- `verbosity` works only via Responses API with `model_kwargs={"text": {"verbosity": "low"}}`
- LangChain GH #32714: if direct `reasoning_effort=` kwarg throws DeploymentNotFound, fall back to `model_kwargs={"reasoning": {"effort": ...}}`

---

## Phase 0 — Foundation (do first, no risk)

**Goal:** Fix the easy wins so future work has a clean base. ~30 min.

1. **Upgrade langchain-openai** to latest (supports `reasoning_effort` kwarg on Azure).
2. **Add `reasoning_effort` env vars** to `.env` (corrected for parallel-tool-call constraint):
   ```
   AZURE_DEPLOYMENT_PLANNER=codex-5-3            # your gpt-5.3-codex deployment alias
   AZURE_DEPLOYMENT_EXECUTOR=gpt-5-mini
   AZURE_DEPLOYMENT_SYNTHESIZER=codex-5-3
   AZURE_REASONING_EFFORT_PLANNER=high
   AZURE_REASONING_EFFORT_EXECUTOR=low            # NOT minimal — disables parallel tools
   AZURE_REASONING_EFFORT_SYNTH=medium
   AZURE_VERBOSITY_SYNTH=low
   ```
   Action item: confirm your Foundry deployment name. The user said "codex 5.3" — likely an alias for base model `gpt-5.3-codex`.
3. **Update system prompt** — fix stale "8,831+" → dynamic count from `neo4j_store.get_stats()` at agent build time.
4. **Pin `run_cypher` to read-only** — block `CREATE|MERGE|SET|DELETE|REMOVE|DROP` regex on input.

**Files touched:** `openmark/agent/graph.py`, `openmark/config.py`, `.env`, `openmark/agent/tools.py`.

---

## Phase 1 — Structured tool outputs (recall + grounding foundation)

**Goal:** Tools return Pydantic, not markdown. This is the load-bearing change for citation correctness. ~1 hour.

1. **Add Pydantic schemas** in new file `openmark/agent/schemas.py`:
   ```python
   class BookmarkHit(BaseModel):
       url: HttpUrl
       title: str
       similarity: float
       source_tool: Literal["semantic","category","tag","community","graph_expand","cypher"]
       tags: list[str]
       category: str | None
       community_id: int | None
       bm_score: float

   class ToolResult(BaseModel):
       hits: list[BookmarkHit]
       strategy: str
       query: str
       total_found: int
   ```
2. **Rewrite all 8 tools** to return `ToolResult`. Keep `_fmt()` only for `run_cypher`.
3. **Update `tools.py` signatures** — LangGraph will serialize Pydantic into tool messages automatically.

**Why first:** Once tools return URLs as typed fields, the agent CAN'T fabricate URLs because the URLs are right there in structured form. This alone closes 80% of the hallucination gap.

---

## Phase 2 — Migrate to `create_agent` + middleware

**Goal:** Move off `create_react_agent` (prebuilt) → `create_agent` (LangChain v1 with middleware). ~2 hours.

1. **Install `langchain >= 1.0`** if not already. Check `pip show langchain`.
2. **New file** `openmark/agent/middleware.py`:
   - `SeenUrlsMiddleware` — adds `seen_urls: set` to state via `before_model`, injects "already retrieved: N URLs" into prompt.
   - `QueryRewriteMiddleware` (HyDE) — `before_model` generates hypothetical bookmark description for first turn.
   - `ModelCallLimitMiddleware(run_limit=10)` — built-in, prevents infinite loops.
   - `TodoListMiddleware()` — built-in, adds `write_todos` planning tool.
3. **Replace `build_agent()`** in `graph.py` to use `create_agent(model, tools, middleware=[...])`.

---

## Phase 3 — Heterogeneous LLM routing (Azure-aware)

**Goal:** Use gpt-5.3-codex with high reasoning for planning, gpt-5-mini with `low` for tool routing, gpt-5.3-codex with medium for synthesis. ~1 hour.

```python
# openmark/agent/llms.py
from langchain_openai import AzureChatOpenAI
from openmark import config

def build_planner():
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_PLANNER,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.AZURE_ENDPOINT,
        use_responses_api=True,                       # codex = Responses only
        model_kwargs={
            "reasoning": {"effort": "high", "summary": "auto"},
            "text": {"verbosity": "low"},
        },
        max_completion_tokens=4000,                   # NOT max_tokens
    )

def build_executor():
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_EXECUTOR,   # gpt-5-mini
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.AZURE_ENDPOINT,
        reasoning_effort="low",                        # NOT minimal
        streaming=True,
    )

def build_synthesizer():
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_SYNTHESIZER,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.AZURE_ENDPOINT,
        use_responses_api=True,
        model_kwargs={
            "reasoning": {"effort": "medium", "summary": "auto"},
            "text": {"verbosity": "low"},
        },
    )
```

**Fallback path if `reasoning_effort=` kwarg fails (LangChain bug #32714):** drop the direct kwarg, use `model_kwargs={"reasoning": {"effort": "low"}}` + `use_responses_api=True`.

---

## Phase 4 — Plan-Execute-Synthesize graph (full architecture)

**Goal:** Custom LangGraph state machine for high recall. ~3-4 hours.

1. **New file** `openmark/agent/graph_v2.py`:
   - `AgentState` with `query, plan, hits, reranked, seen_urls, citations, retries`.
   - Node `planner`: `gpt-5-codex high` → emits `SearchPlan` Pydantic with N queries × M strategies.
   - Node `executor`: uses `Send` to fan out N×M parallel tool calls. Collects all `BookmarkHit`s into `hits`.
   - Node `reranker`: cross-encoder rerank top 100 → top 20. Use `bge-reranker-v2-m3` via sentence-transformers (already a dep).
   - Node `synthesizer`: `gpt-5-codex` + `with_structured_output(Answer, strict=True)` — must cite URLs.
   - Edge `validate_citations`: if any citation URL ∉ seen_urls, loop back to synthesizer with correction (max 2x).
2. **Keep `graph.py` (ReAct)** as fallback agent — UI tab switch to choose.

---

## Phase 5 — New tools to close coverage gaps

**Goal:** Cover the things the current 8 tools miss. ~2 hours.

| Tool | Cypher |
|---|---|
| `find_by_domain(domain)` | `MATCH (b)-[:FROM_DOMAIN]->(d:Domain {name: $domain})` |
| `find_by_source(source)` | `MATCH (b) WHERE b.source = $source` |
| `find_recent(days)` | requires adding `created_at` property on ingest — needs schema change |
| `search_youtube(query)` | wrapper: `vector_search(q, source='youtube_*')` |
| `search_linkedin(query)` | wrapper: `vector_search(q, source='linkedin')` |
| `dump_subgraph(urls, hops)` | full neighborhood expansion for synthesis |

**Note:** `find_recent` requires backfill — defer unless user wants it.

---

## Phase 6 — Evaluation harness

**Goal:** Measure "PERFECT" objectively. ~2 hours.

1. **Build `eval/` directory** with hand-curated query → expected_urls pairs.
   - Use Ahmad's real past queries. Start with 20 queries.
2. **Metric**: recall@20 (how many expected URLs are in top-20 returned).
3. **Run nightly**: compare ReAct (current) vs PlanExecute (new). Track regressions.

---

## Sequencing recommendation

1. **Do today**: Phase 0 (foundation) + Phase 1 (structured outputs) — biggest quality lift, lowest risk.
2. **Do next**: Phase 3 (model routing) — needs Phase 0.
3. **Do later**: Phase 2 (middleware) + Phase 4 (full graph) — biggest architecture change.
4. **Do iteratively**: Phase 5 (new tools) — as gaps surface in real use.
5. **Background**: Phase 6 (eval) — only needed once we're iterating on quality.

## Risk / blast radius

- Phases 0-1 are pure refactor, backward-compatible.
- Phase 2-3 change agent init but keep same interface (`ask(agent, q)`).
- Phase 4 is a NEW agent — runs alongside the old one. Toggle in UI.
- No phase requires re-embedding bookmarks.
- No phase requires Neo4j schema changes (except optional Phase 5 `created_at`).

## What I still need from you
1. **Confirm Foundry deployment name** — open Foundry portal → Deployments → look for `gpt-5.3-codex` (base model column). Tell me the **deployment alias** (e.g. `codex-5-3`, `codex53`, etc.). The user-typed alias is what we put in `.env`.
2. **Confirm access** — `gpt-5.3-codex` is limited-access (Feb 2026). If your deployment shows ✓ active, we're good. Otherwise we drop to `gpt-5.2-codex` or `gpt-5.1-codex-max` (xhigh available there).
3. **Approve Phase 0+1 to start coding now** — those are pure refactor, no architecture change, ~90 min total.

## Reference files in this folder
- `00_expert_guidance.md` — opinionated architecture (from langchain-agents-expert)
- `01_langchain_latest.md` — LangChain v1.x + LangGraph 1.0 cookbook patterns
- `02_azure_reasoning.md` — Azure Foundry reasoning model integration details
- `03_current_state.md` — current OpenMark agent gap analysis (11 gaps)
- `04_phased_plan.md` — this file
