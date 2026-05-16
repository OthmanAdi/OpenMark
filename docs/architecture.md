# Architecture (v3)

## Overview

OpenMark v3 is a single-store, single-orchestrator system. **Neo4j Graph RAG** holds every bookmark with a 1024-dim vector index, tag co-occurrence edges, Louvain communities, and SIMILAR_TO neighbor edges. A LangChain v1 `create_agent` orchestrator on Azure AI Foundry delegates work to 10 specialist sub-agents via pure middleware.

ChromaDB and deepagents were both removed in v3. Everything middleware-related now comes from `langchain.agents.middleware`.

```
                  User query (Gradio Chat tab or CLI)
                                │
                       Orchestrator (one create_agent graph)
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        classify_intent   dynamic_prompt    14 middleware
        (@before_model)  (@dynamic_prompt)  (langchain v1)
                                │
                 task_* tools (10 sub-agent delegators)
                                │
   ┌──────────────┬─────────────┼─────────────┬──────────────┐
   │              │             │             │              │
researcher    composer_*   humanizer      polisher       verifier
(21 tools)   (0 tools,    (ar-* / he,   (en, ai-tell    (Pydantic
              ToolStrategy  loads        scrub)         VerificationReport)
              schemas)      humanizer-*
                            skill)
                                │
                          Neo4j Graph RAG
                          + web (Tavily / Brave / DDG /
                                 GitHub / Reddit)
```

---

## Embedding layer

Provider-agnostic. Swap with one env var.

```
EMBEDDING_PROVIDER=pplx   →  LocalEmbedder (pplx-embed-context-v1-0.6b for docs,
                                            pplx-embed-v1-0.6b for queries; 1024 dim)
EMBEDDING_PROVIDER=azure  →  AzureEmbedder (text-embedding-3-large)
```

### Why two pplx-embed models?

Perplexity ships matched query + doc encoders. Using the right model for each role measurably improves retrieval. Most public examples use one model for both — pplx ships a real production pattern.

### Compatibility patches

`openmark/embeddings/local.py` applies two monkey patches before model load:

1. `sentence_transformers 4.x` removed `Module` base class that pplx-embed imports. We alias `torch.nn.Module` to `sentence_transformers.models.Module`.
2. `transformers 4.57+` adds `list_repo_templates()` and 404s on pplx repos that lack `additional_chat_templates`. We patch the function to return `[]` on exception.

Pin `sentence-transformers==3.3.1` (4.x is incompatible).

---

## Neo4j Graph RAG schema

```
(:Bookmark {url, title, score, source, category, created_at, embedding[1024]})
    -[:IN_CATEGORY]->   (:Category {name})
    -[:TAGGED]->        (:Tag {name})
    -[:FROM_SOURCE]->   (:Source {name})
    -[:FROM_DOMAIN]->   (:Domain {name})
    -[:IN_COMMUNITY]->  (:Community {id})         ← Louvain (GDS plugin)
    -[:SIMILAR_TO {score}]->  (:Bookmark)         ← top-5 cosine neighbors

(:Tag)-[:CO_OCCURS_WITH {count}]-(:Tag)            ← tags co-saved on same bookmark
```

Vector index: `bookmark_embedding` on `b.embedding`. Used by `SEARCH b IN (VECTOR INDEX bookmark_embedding ...)` Cypher constructs.

### Useful Cypher

```cypher
// Counts
MATCH (b:Bookmark) RETURN count(b) AS bookmarks;
MATCH (t:Tag)      RETURN count(t) AS tags;

// Vector search
CALL () {
  MATCH (b) SEARCH b IN (VECTOR INDEX bookmark_embedding FOR $embedding LIMIT 100) SCORE AS score
  RETURN b, score LIMIT 10
}
OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
RETURN b.url, b.title, score, collect(t.name)[..6] AS tags
ORDER BY score DESC;

// Tag cluster (2 hops)
MATCH (t:Tag {name: 'langchain'})-[:CO_OCCURS_WITH*1..2]-(r:Tag)
RETURN r.name, count(*) AS strength ORDER BY strength DESC;

// SIMILAR_TO from a URL
MATCH (b:Bookmark {url: $url})-[r:SIMILAR_TO]-(o:Bookmark)
RETURN o.title, o.url, r.score ORDER BY r.score DESC LIMIT 6;

// Community peers
MATCH (b:Bookmark {url: $url})-[:IN_COMMUNITY]->(c:Community)<-[:IN_COMMUNITY]-(peer:Bookmark)
WHERE peer.url <> $url
RETURN peer.title, peer.url LIMIT 6;
```

---

## The orchestrator (v3)

`openmark/agent/graph.py`. Single `create_agent`, 14 middleware, 11 tools.

### Middleware stack

| # | Middleware | Source | Purpose |
|---|---|---|---|
| 1 | `classify_intent` | `openmark.agent.classification` (@before_model) | Slash → regex → fast LLM. Writes `state.intent` once per thread. |
| 2 | `dynamic_orchestrator_prompt` | `openmark.agent.classification` (@dynamic_prompt) | Reads `state.intent`, injects matching system prompt with live KB stats. |
| 3 | `ContextEditingMiddleware(ClearToolUsesEdit)` | langchain | Trims old tool outputs at 120k tokens, keeps last 4. |
| 4 | `SummarizationMiddleware` | langchain | Triggers at 100k tokens / 80 messages, keeps last 24. Uses cheap summarizer. |
| 5 | `TodoListMiddleware` | langchain | Adds `write_todos` so agent plans before fan-out. |
| 6 | `ModelCallLimitMiddleware` | langchain | 30 model calls / turn hard cap. |
| 7 | `ToolCallLimitMiddleware` | langchain | Global 40 + per-tool caps on expensive composers. |
| 8 | `ModelRetryMiddleware` | langchain | 3 retries, exponential backoff. |
| 9 | `slash_skill_loader` | `openmark.agent.middleware` (@before_model) | `/<skill>` pre-loads SKILL.md, strips slash. |
| 10 | `OpenMarkSkillMiddleware` | `openmark.agent.middleware` | Skill catalogue + `load_skill` tool. |
| 11 | `ToolRetryMiddleware` | langchain | 2 retries on flaky sub-agent calls. |
| 12 | `tool_event_middleware` | `openmark.agent.middleware` (@wrap_tool_call) | Captures every tool start/end/error for UI cards. |

Custom state schema: `OrchestratorState(AgentState)` adds `intent: str | None` and `intent_source: str | None`.

Checkpointer: `SqliteSaver` at `data/openmark_agent.db` — threads survive restart.

---

## Sub-agents

Each lives in its own file under `openmark/agent/subagents/` and is exposed to the orchestrator as a `task_<role>` tool. The tool body invokes a compiled `create_agent` graph (cached at module load) and formats the result for the orchestrator's view (compacted text + JSON of `structured_response` where applicable).

| Sub-agent | Tools | response_format | Triggered by |
|---|---|---|---|
| `task_researcher` | 21 (15 graph + 6 web) | — | every retrieval path |
| `task_compose_linkedin` | 0 | `LinkedInPost` | `/newsletter-thread`, "LinkedIn post on X" |
| `task_compose_essay` | 0 | `NewsletterEssay` | `/newsletter-essay` |
| `task_compose_roundup` | 0 | `NewsletterRoundup` | `/newsletter-roundup` |
| `task_compose_comparison` | 0 | `NewsletterComparison` | `/newsletter-comparison` |
| `task_compose_analytical` | 0 | `NewsletterAnalytical` | `/newsletter on X` (default) |
| `task_humanize` | 0 | — | language = `ar-msa / ar-egt / ar-shami / he` |
| `task_polish` | 0 | — | language = `en` |
| `task_verify` | 0 | `VerificationReport` | post-compose retry-loop guard |
| `task_author_skill` | `write_skill` | — | "bake this prompt into a reusable skill" |

Sub-agents have their own middleware stack (`ContextEditingMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `ToolRetryMiddleware`, `tool_event_middleware`). The shared event bus means sub-agent tool calls appear in the same Gradio UI cards as the orchestrator's own calls.

---

## Classification flow

`openmark/agent/classification.py`. Resolution order (cheapest to most expensive):

```
1. Slash command           /newsletter, /deep-research, /weekly-digest, ...
                           → _SLASH_TO_INTENT lookup, no LLM call.
2. Regex heuristic         "this week", "compare A and B", "expand <URL>", ...
                           → _heuristic_intent, no LLM call.
3. Fast LLM classifier     Fallback only. Uses gpt-4.1-mini (or whatever
                           role_model_id('classifier') resolves to) with
                           .with_structured_output(IntentLabel).
```

Sticky per thread: once `state.intent` is set, subsequent turns short-circuit. Intent labels: `fast | deep | newsletter | digest | dive`. The dynamic_prompt reads the label and appends a matching hint to the orchestrator system prompt.

---

## Foundry model bank

`openmark/models/foundry.py` — typed registry of 25 frontier deployments sourced from models.dev (snapshot 2026-05-16):

```
openai:     gpt-5.5, gpt-5.3-codex, gpt-5.3-chat-latest, gpt-5, gpt-5-mini,
            gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini,
            o1, o1-pro, o3, o3-mini
anthropic:  claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5
xai:        grok-4.3, grok-4.20-0309-reasoning, grok-4.20-0309-non-reasoning
deepseek:   deepseek-reasoner, deepseek-chat
meta:       llama-4-maverick-17b-128e-instruct-fp8, llama-4-scout-...
mistral:    mistral-large-2411
```

Each record carries: `context_window`, `max_output`, `reasoning`, `tool_use`, `modalities_in/out`, `price_in/out_per_1m`, `release` date.

`openmark/models/router.py` maps roles to deployments with `OPENMARK_MODEL_<ROLE>` overrides taking precedence over legacy `AZURE_DEPLOYMENT_*` keys.

`openmark/agent/llms.py` builds the right `AzureChatOpenAI` wrapper per deployment:

- Reasoning OpenAI models (gpt-5*, codex, o-series) → Responses API with top-level `reasoning={"effort":..., "summary":"detailed"}` and `verbosity`.
- Grok deployments → Chat Completions with top-level `reasoning_effort` and the Foundry `model=deployment` body quirk.
- Plain chat models (gpt-4.x, mistral, llama) → vanilla `AzureChatOpenAI`.

---

## Gradio UI

7 tabs:

| Tab | What it does |
|---|---|
| Search | Direct semantic search on Neo4j vector index. Cards with similarity bar, category color, tags, source icon. |
| Chat | Full orchestrator conversation. Live cards stream every sub-agent + tool call. Per-turn thinking bubbles. SQLite history dropdown. Slash command picker. |
| Stats | Knowledge graph totals, by-category breakdown, top tags. |
| Graph 3D | Force-directed 3D graph (Search mode + Explore mode). |
| + Add | Drop URLs or upload Edge/Chrome HTML / JSON / TXT — parsed, deduped, embedded, indexed. |
| Agent Tools | Live registry of every tool the orchestrator and researcher can call. |
| Agent Skills | Live registry of SKILL.md recipes (openmark-* / humanizer-* / agent-generated-*). |

Run: `python -m openmark.ui.app` → `http://127.0.0.1:7860`

---

## Data flow

```
Source files (Raindrop JSON, Edge HTML, LinkedIn JSON, YouTube JSON)
        │
   merge.py → normalize.py
        │
   ~13k items with doc_text + tags + category + source
        │
   EmbeddingProvider.embed_documents() (pplx-embed-context, 1024 dim)
        │
   Neo4j MERGE (nodes + IN_CATEGORY + TAGGED + FROM_SOURCE + FROM_DOMAIN)
        │
   CO_OCCURS_WITH edges (tag pairs co-saved on same bookmark)
        │
   SIMILAR_TO edges (top-5 cosine neighbors per bookmark; skip with --skip-similar)
        │
   Optional: Louvain community detection (GDS plugin) → IN_COMMUNITY edges
```

---

## Tests

```
tests/
├── agent/
│   ├── test_subagent_registry.py        unit
│   ├── test_classification.py            unit
│   ├── test_llms_factory.py              unit
│   └── test_live_foundry.py              live (gated on OPENMARK_RUN_LIVE_E2E=1)
├── composer/
│   ├── test_schemas.py                   unit (pure pydantic)
│   ├── test_export.py                    unit (pure renderers)
│   ├── test_verifier_scoring.py          unit
│   └── test_write_skill.py               unit
└── middleware/
    └── test_summarization_wiring.py      unit (introspects compiled graph)
```

99 unit tests + 7 live Foundry tests. Live tests cover every sub-agent: classifier, researcher, all 5 composer paths, verifier, polisher, humanizer.
