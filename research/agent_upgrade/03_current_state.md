# Current Agent — Gap Analysis

## What exists today (verified 2026-05-10)

**File:** `openmark/agent/graph.py` (101 lines)
**Pattern:** `langgraph.prebuilt.create_react_agent` — basic prebuilt ReAct
**LLM:** `AzureChatOpenAI`, deployment from `AZURE_DEPLOYMENT_LLM` env var (currently `gpt-5-mini`)
**Memory:** `MemorySaver` (in-process, thread_id keyed)
**Tools:** 8 tools in `tools.py` — all return markdown strings

```python
# Current init (graph.py:47-55)
AzureChatOpenAI(
    azure_endpoint=...,
    api_key=...,
    azure_deployment=...,
    api_version=...,
    streaming=True,
)
# No reasoning_effort, no temperature, no verbosity, no model_kwargs.
```

## 11 concrete gaps preventing "perfect" retrieval

### Architecture
1. **ReAct improv loop** — agent does one search, sees results, replies. No planning, no parallel multi-angle search, no escalation when recall is low.
2. **No `Send` fan-out** — even though 8 tools exist, the agent calls them serially. Graph data (SIMILAR_TO + Louvain) is barely used because each tool call is a separate LLM turn.
3. **No state beyond messages** — `MemorySaver` only persists the message list. No `seen_urls`, no `tried_strategies`, no `confidence_score`.

### LLM config
4. **No `reasoning_effort` set** — gpt-5-mini supports `minimal|low|medium|high`. Default behavior wastes tokens on simple lookups and underthinks complex multi-angle queries.
5. **No model routing** — same LLM for planning AND tool-arg generation. Should be heterogeneous (codex/high for planning, mini/minimal for executor).
6. **No `verbosity` control** — gpt-5 supports `verbosity={low|medium|high}`. Currently get medium-default everywhere.

### Tools
7. **Tools return markdown** — `_fmt(results)` in `tools.py:20` builds a string. Agent has to *parse* its own tool output via the LLM to extract URLs. This is exactly where hallucinated URLs leak in. Should return structured Pydantic.
8. **No deduplication across tool calls** — calling `search_semantic` then `search_by_category` for similar query returns overlapping results, wasting context.
9. **No reranking** — Neo4j returns top-N by vector cosine. No cross-encoder rerank. For a query like "agent memory patterns", cosine puts generic "memory" articles before LangGraph-specific ones.
10. **`run_cypher` is dangerous** — agent can execute arbitrary Cypher. Could `DETACH DELETE`. No read-only enforcement.

### System prompt
11. **Stale prompt** — `graph.py:18` says "8,831+ saved bookmarks". Real count is 11,443. Agent reasoning uses this stale number. Also: source counts in prompt are stale (4,359/2,094/1,260/430/189).

## What it does well (keep these)
- 8 tools cover the main retrieval angles (semantic, category, community, tag, graph_expand)
- `MemorySaver` checkpointer works for per-thread memory
- Graceful exception handling on every tool (no crashes)
- Streaming enabled

## Failure modes I expect from the gaps

| User query | Likely failure |
|---|---|
| "find my langgraph stuff" | Returns top-10 vector matches, misses 95% of relevant items in the LangChain community. Should escalate to community search. |
| "all my AI agent papers from arxiv" | Returns 10 generic agent results, doesn't filter by domain or source. No tool exists. |
| "what did I save about RAG yesterday" | No temporal awareness, no `created_at` filter. |
| "show me everything related to this URL: …" | Works via `graph_expand` BUT agent often doesn't call it without prompting. |
| Vague "what's interesting" | Agent gives up or returns whatever cosine ranks high — useless. No exploratory mode. |

## Cypher needed for new tools (graph already supports)
- `MATCH (b:Bookmark)-[:FROM_DOMAIN]->(d:Domain {name: $domain})` — domain filter (data exists, no tool)
- `MATCH (b:Bookmark)-[:FROM_SOURCE]->(:Source)` ... wait, source is denormalized property, not a node — check `_write_batch` in neo4j_store.py:115 (`b.source = $source` set as property)

## Conclusion
Current agent is a thin ReAct wrapper. Solid foundation (good tools, good graph schema) but no orchestration, no rigor on citations, no reasoning control. The graph is way richer than the agent uses.
