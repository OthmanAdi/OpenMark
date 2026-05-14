# LangChain v1.x + LangGraph 1.0 — Retrieval Agent Patterns (May 2026)

Scope: cookbook-grade patterns for high-recall RAG agents, scraped from
official LangChain / LangGraph docs and the `langgraph/examples/rag` folder
at tag `1.0.8`. Targeted at the OpenMark bookmark retriever.

---

## 1. LangChain v1.x — what is new

`langchain` v1 collapses `AgentExecutor` and most chains into a single
`create_agent` factory plus a **middleware system**. Middleware are
composable lifecycle hooks (`before_model`, `wrap_model_call`, `after_model`,
`before_tool`, `after_tool`) that replace the old `pre_model_hook` /
`post_model_hook` callbacks. They can mutate state, swap the tool list,
short-circuit the model call, or interrupt for HITL.

Stock middleware shipped with v1:

- `HumanInTheLoopMiddleware` — approve/edit/reject per tool.
- `ModelCallLimitMiddleware` — `thread_limit`, `run_limit`, `exit_behavior`.
- `TodoListMiddleware` — auto-injects a `write_todos` planning tool.
- `SummarizationMiddleware` — trims long histories.
- `AnthropicPromptCachingMiddleware`.

Source: <https://docs.langchain.com/oss/python/langchain/middleware/built-in>

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware, ModelCallLimitMiddleware, TodoListMiddleware,
)
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_bookmarks, graph_expand, find_by_source],
    middleware=[TodoListMiddleware(), ModelCallLimitMiddleware(run_limit=8)],
)
```

### Custom middleware = RAG injection
The official docs ship a `RetrieveDocumentsMiddleware` pattern that fires
`before_model`, runs `vector_store.similarity_search(last_message.text)`,
and rewrites the user turn with the docs inline. This is the v1-native
replacement for the old `RetrievalQA` chain.

Source: <https://docs.langchain.com/oss/python/langchain/rag>

---

## 2. LangGraph 1.0 cookbook patterns

All paths are under `github.com/langchain-ai/langgraph/blob/1.0.8/examples/rag/`.

### Agentic RAG (`langgraph_agentic_rag.ipynb`)
LLM agent decides *whether* to call the retriever tool. Nodes:
`agent → retrieve(ToolNode) → grade_documents → {generate | rewrite}`.
`rewrite` loops back to `agent`. This is the recommended baseline for any
retrieval system where the question may not need retrieval at all.

### Corrective RAG / CRAG (`langgraph_crag.ipynb`)
After retrieval, each doc is graded. If *any* doc fails the grader, the
graph rewrites the query and hits **Tavily web search** as a fallback.
Edges: `retrieve → grade_documents → {transform_query → web_search →
generate, generate}`.

### Self-RAG (`langgraph_self_rag_local.ipynb`)
Adds a second grader **on the generation itself** — checks both
hallucination (is the answer grounded?) and usefulness. Returns to
`transform_query` if not useful, regenerates if not grounded.
Edges include `generate → {"not supported": "generate", "useful": END,
"not useful": "transform_query"}`. This is the canonical
citation-grounded pattern.

### Adaptive RAG (`langgraph_adaptive_rag.ipynb`)
Pydantic `RouteQuery` model with `Literal["vectorstore", "web_search"]`
+ `llm.with_structured_output(RouteQuery)` routes per question. Combine
with CRAG for the strongest recall.

### Plan-and-Execute
Lives under `examples/plan-and-execute/`. A planner emits a list, an
executor runs each step, a replanner revises after each result. Pattern
of choice when a query decomposes into 3+ subqueries.

### Query decomposition / HyDE
Not a separate notebook in v1.0.8 — implemented as preprocessing nodes
in adaptive_rag and `rag-from-scratch` repo. HyDE = generate a
hypothetical answer with an LLM, embed it, search with that embedding.

---

## 3. Deep Agents (v0.2, May 2026)

`pip install deepagents`. `create_deep_agent(model, tools, system_prompt,
subagents=[...])` returns a compiled LangGraph with:

- `write_todos` planning tool baked in
- pluggable filesystem (InMemory, LocalDisk, LangGraph Store, Modal, Daytona)
- `task` tool to spawn **subagents with isolated context** — the main
  agent only sees the final summary, not every retrieval step

Source: <https://docs.langchain.com/oss/python/deepagents/overview>,
<https://github.com/langchain-ai/deepagents>

This is the pattern when a retriever agent must try semantic + graph +
grep + tag-filter strategies: each strategy gets its own subagent, main
agent sees clean summaries only.

---

## 4. Neo4j graph RAG

`langchain-neo4j` ships:

- `Neo4jVector` — `from_existing_index`, `similarity_search`,
  `as_retriever()`. Drop-in retriever tool.
- `GraphCypherQAChain` — NL → Cypher → results → NL answer. Two LLM calls.
- `Neo4jGraph.query()` for raw Cypher inside custom tools.

Source: <https://neo4j.com/labs/genai-ecosystem/langchain/>,
<https://python.langchain.com/docs/integrations/vectorstores/neo4jvector/>

Recommended pattern: expose Neo4jVector as `search_bookmarks_semantic`
tool **and** a hand-rolled `graph_expand(node_id, hops)` tool that runs
Cypher directly. Let the agent pick.

---

## 5. Citation-grounded generation
Pattern from self-rag: after `generate`, run a `GradeHallucinations`
Pydantic grader (`grounded_in_facts: bool`) using
`llm.with_structured_output`. Regenerate on fail. Cite by passing
`doc.metadata["source"]` (or Neo4j node id) verbatim into the prompt and
forbidding the model from inventing URLs.

Sources:
- <https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/>
- <https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/>
- <https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/>
- <https://docs.langchain.com/oss/python/langchain/middleware/built-in>
- <https://docs.langchain.com/oss/python/deepagents/overview>
