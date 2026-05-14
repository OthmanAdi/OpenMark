# Expert Guidance — Retrieval Agent Upgrade

Source: langchain-agents-expert (LangChain v1.x cookbook knowledge, May 2026)

## TL;DR pattern

```
Planner (gpt-5-codex, reasoning_effort=high)
   ↓ Send() fan-out
Executor (gpt-5-mini, reasoning_effort=minimal)  [parallel]
   ↓
Reranker node (bge-reranker-v2-m3 local OR Cohere Rerank)
   ↓
Synthesizer with structured_output(strict=True)
   ↓
URL-grounding guard edge (validates citations ∈ retrieved set, max 2 loops)
```

Custom state with `seen_urls: Annotated[set, operator.or_]`. HyDE in `pre_model_hook`. Keep ReAct only as fallback inside synthesizer when recall < threshold.

## 1. Architecture: Plan-Execute-Synthesize, not ReAct

ReAct = "one good action wins" improv. Wrong for retrieval. We want **high recall via parallel multi-angle search**.

LangGraph cookbook: `examples/plan-and-execute/plan-and-execute.ipynb`.

Three nodes:
- **planner** — produces typed `SearchPlan` (Pydantic): `{queries: list[Query], strategies: list[Literal["semantic","category","tag","community","cypher"]], k_per: int}`
- **executor** — fans out tool calls in parallel via `langgraph.constants.Send` (map-reduce)
- **synthesizer** — dedupes, reranks, cites

`Send` is the killer feature — planner emits N `Send("execute_search", {...})`, all run concurrently. ReAct can't do this cleanly.

## 2. Middleware Hooks (LangChain v1-alpha `create_agent`)

`create_agent` is the v1 successor to `create_react_agent` in `langchain.agents`:

- **`pre_model_hook`** — before every LLM call. Use for HyDE query rewriting (generate hypothetical bookmark description, embed it, search with it). Also message trimming.
- **`post_model_hook`** — after LLM, before tool dispatch. Validate tool args (force `k>=20` for vague queries).
- **`ToolNode(handle_tool_errors=...)`** — catches tool exceptions, feeds error back to model.
- **`modify_model_request`** middleware — inject reranking context, dedupe seen URLs.

Dedup across tool calls: `seen_urls: Annotated[set[str], operator.or_]` on custom AgentState. Pre_model_hook injects "already retrieved: {n} URLs, find DIFFERENT ones."

Reranking: dedicated node after executor, not middleware. `bge-reranker-v2-m3` local via sentence-transformers, OR Cohere Rerank. Top 100 → top 20.

## 3. Tool Design: Many small typed tools, Pydantic returns

Many small tools > one super-tool. Reasoning models route on tool *names* better than string enum params. Keep current 8-tool split.

Return **structured Pydantic**, NOT markdown:

```python
class BookmarkHit(BaseModel):
    url: HttpUrl
    title: str
    score: float
    source_tool: str
    community_id: int | None
    tags: list[str]

class ToolResult(BaseModel):
    hits: list[BookmarkHit]
    strategy_used: str
    query_echo: str
```

LangGraph serializes Pydantic into tool messages automatically. Markdown-in-tool-out is where hallucination breeds.

Add a `multi_strategy_search` **meta-tool** that takes `strategies: list[str]` and fans out internally — planner gets one-shot multi-angle without 8 tool calls.

## 4. Reasoning Effort on Azure gpt-5-codex

`AzureChatOpenAI` (langchain-openai >= 0.3) supports it directly:

```python
llm = AzureChatOpenAI(
    azure_deployment="gpt-5-codex",
    api_version="2025-04-01-preview",
    reasoning_effort="high",   # "minimal" | "low" | "medium" | "high"
    model_kwargs={"reasoning": {"summary": "auto"}},
)
```

**Heterogeneous models per node:**
- Planner → `gpt-5-codex` with `reasoning_effort="high"` (rare call, quality matters)
- Executor → `gpt-5-mini` with `reasoning_effort="minimal"` (latency, cost)

Trivial in LangGraph — bind different LLMs to different nodes.

## 5. Citation Grounding (Hard Constraint — 3 layers, all required)

1. **Structured output on synthesizer**: `llm.with_structured_output(Answer, strict=True)` where `citations: list[HttpUrl]` is required.
2. **Post-validation guard edge**: check every URL in `citations` exists in `state["seen_urls"]`. If not, route back to synthesizer: `"URL X was not retrieved, remove it."` Max 2 loops.
3. **Prompt-level**: synthesizer system prompt — *"You may ONLY cite URLs present in <retrieved_urls>. Citing any other URL is a hard failure."*

Layer 2 is the load-bearing one. Prompts alone never reach PERFECT.

## References
- `langchain-ai/langgraph` repo `examples/plan-and-execute`
- `examples/rag/langgraph_agentic_rag.ipynb`
- v1-alpha `create_agent` middleware: `docs.langchain.com/oss/python/langchain/middleware`
- `python.langchain.com/docs/integrations/chat/azure_chat_openai/` (reasoning section)
