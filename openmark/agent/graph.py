"""
OpenMark agent — LangChain v1.x `create_agent` with middleware.

Architecture:
  - codex 5.3, reasoning=high, summary=detailed (thinking always visible)
  - TodoListMiddleware → agent plans before executing
  - ModelCallLimitMiddleware(thread_limit=20) → prevents runaway loops
  - ToolRetryMiddleware → resilient against transient Neo4j hiccups
  - Custom @before_model → injects live stats and dedupe hint into prompt
  - Custom @after_model → captures retrieved URLs for citation grounding

The 12 tools (typed Pydantic returns) cover:
  semantic, category, community, tag, tag-cluster, graph_expand,
  domain, source, linkedin, youtube, stats, raw-cypher (read-only).

Memory: MemorySaver checkpointer keyed by thread_id.
"""

from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
    before_model,
    after_model,
)
from langgraph.checkpoint.memory import MemorySaver

from openmark import config
from openmark.agent.llms import build_executor, build_classifier
from openmark.agent.tools import ALL_TOOLS


# ── Lazy classifier ───────────────────────────────────────────────────────────
_classifier = None
def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = build_classifier()
    return _classifier


CLASSIFIER_PROMPT = """Classify the user's query about Ahmad's bookmark knowledge base.

Pick exactly ONE label from this list, output the label and nothing else:

- fast        : a single concrete lookup ("find my bookmarks on X", one-tool answer)
- deep        : multi-angle research, comparison, landscape questions
- newsletter  : asked to draft / compose / write a newsletter
- digest      : "what did I save this week / last N days", time-window recap
- dive        : a single URL with "expand", "dig into", "neighbors of"

If unsure, output `fast`.

User query: {query}

Label:"""


SYSTEM_PROMPT_TEMPLATE = """You are OpenMark — Ahmad's personal AI knowledge assistant.

You have access to his curated knowledge base of {bookmarks:,} bookmarks, LinkedIn posts,
and YouTube videos — all stored in Neo4j with semantic embeddings (pplx-embed 1024-dim),
tag co-occurrence edges, Louvain community structure, and SIMILAR_TO neighbor edges.

CURRENT KNOWLEDGE BASE: {bookmarks:,} bookmarks · {tags:,} tags · {categories} categories · {communities} communities

YOUR JOB
- Find EXACTLY what Ahmad saved. Never invent URLs, never paraphrase results away from their real URLs.
- Surface hidden connections via tags, communities, and SIMILAR_TO edges.
- When a query is vague, run MULTIPLE searches with different strategies in parallel before answering.

RETRIEVAL STRATEGY (use write_todos to plan before searching)
1. Start broad with search_semantic. If results are sparse or off-topic, do NOT give up — escalate.
2. Escalate paths in parallel when possible:
   - search_by_category if you can map the query to a canonical category
   - search_by_community to surface a coherent topic cluster
   - find_by_tag if a specific tag exists (e.g. 'rag', 'agents')
   - search_linkedin / search_youtube when the user mentions a source
   - find_by_domain when the user mentions a site (github.com, arxiv.org, etc.)
3. graph_expand(url) ONLY after you have a specific bookmark URL the user wants explored.
4. run_cypher is a last resort for advanced graph questions — read-only enforced.

CITATION RULES — STRICT
- You may ONLY cite URLs that appeared in a tool result. Never fabricate a URL.
- When listing bookmarks for the user, include the EXACT URL from the tool output.
- If you don't have enough evidence, say so. Suggest a follow-up search instead of guessing.

OUTPUT FORMAT
- Be direct. No "Sure!", no "Here's what I found:". Get to the answer.
- For bookmark lists: numbered list with `Title — URL — short why`.
- For analytical questions: 2-4 sentence summary, then a citations block.
- If asked "what did I save about X" → return the actual URLs and titles, ranked.
"""


# ── Custom middleware ─────────────────────────────────────────────────────────

@before_model
def inject_live_stats(state: AgentState, runtime: Any) -> dict | None:
    """
    On the first turn:
      1. Run the cheap classifier (gpt-5-mini) to pick a mode label.
      2. Rewrite the system prompt with live Neo4j counts AND the mode hint.
    """
    messages = state.get("messages") or []
    if not messages or any(m.type == "system" for m in messages if hasattr(m, "type")):
        return None

    # Pull the user's first user message text for classification
    last_user_text = ""
    for m in messages:
        if getattr(m, "type", "") == "human":
            last_user_text = getattr(m, "content", "") or ""
            if isinstance(last_user_text, list):
                last_user_text = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in last_user_text
                )
            break

    # Cheap classification — fall back to "fast" on any error
    mode = "fast"
    try:
        if last_user_text.strip():
            cls_resp = _get_classifier().invoke(CLASSIFIER_PROMPT.format(query=last_user_text.strip()))
            raw = (cls_resp.content if hasattr(cls_resp, "content") else str(cls_resp)).strip().lower()
            if isinstance(raw, list):
                raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
            for cand in ("newsletter", "digest", "deep", "dive", "fast"):
                if cand in raw:
                    mode = cand
                    break
    except Exception as e:
        print(f"[classifier] failed: {e}; defaulting mode=fast")

    try:
        from openmark.stores import neo4j_store
        s = neo4j_store.get_stats()
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            bookmarks=s.get("bookmarks", 0),
            tags=s.get("tags", 0),
            categories=s.get("categories", 0) or 19,
            communities=s.get("communities", 0),
        )
    except Exception:
        prompt = SYSTEM_PROMPT_TEMPLATE.format(bookmarks=13000, tags=5000, categories=19, communities=0)

    prompt += f"\n\nMODE: {mode}  (classifier set this; tailor your effort accordingly)\n"
    if mode == "fast":
        prompt += "- Single tool call. Render results in 10 seconds. No graph_expand, no Cypher.\n"
    elif mode == "deep":
        prompt += "- Multi-angle parallel search (semantic + community or tag). 2-3 graph_expand on winners. Cite only observed URLs.\n"
    elif mode == "newsletter":
        prompt += "- This is newsletter material. Pull broad source, rank, recommend 5-7 anchor bookmarks. Don't draft prose here — the Claude Code skill openmark-newsletter handles drafts.\n"
    elif mode == "digest":
        prompt += "- Time window query. Use find_recent or search_by_date_range. Group by category/tag. Compact output.\n"
    elif mode == "dive":
        prompt += "- One URL focus. Use get_bookmark_full + graph_expand. Return structured neighborhood, not prose.\n"

    print(f"[classifier] mode={mode}")
    from langchain_core.messages import SystemMessage
    return {"messages": [SystemMessage(content=prompt)] + list(messages)}


# ── Build agent ────────────────────────────────────────────────────────────────

def build_agent():
    """Compile the agent graph with codex 5.3 + middleware stack."""
    llm = build_executor()
    print(f"Agent LLM: {config.AZURE_DEPLOYMENT_EXECUTOR} (Responses API, reasoning={config.AZURE_REASONING_EXECUTOR}, summary=detailed)")

    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        # System prompt is dynamic via @before_model — pass minimal here.
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(
            bookmarks=11000, tags=5000, categories=19, communities=0,
        ),
        middleware=[
            inject_live_stats,
            TodoListMiddleware(),
            ToolRetryMiddleware(max_retries=2),
            ModelCallLimitMiddleware(thread_limit=20, run_limit=15),
        ],
        checkpointer=MemorySaver(),
    )
    return agent


# ── Ask with thinking surfaced ─────────────────────────────────────────────────

def _extract_thinking(messages: list) -> str:
    """Pull reasoning summaries out of AIMessage.additional_kwargs / content_blocks."""
    out = []
    for m in messages:
        if not hasattr(m, "additional_kwargs"):
            continue
        ak = m.additional_kwargs or {}
        reasoning = ak.get("reasoning")
        if isinstance(reasoning, dict):
            summary = reasoning.get("summary")
            if isinstance(summary, list):
                for blk in summary:
                    if isinstance(blk, dict) and blk.get("text"):
                        out.append(blk["text"])
            elif isinstance(summary, str) and summary.strip():
                out.append(summary)
        # langchain-openai 1.x also exposes via content_blocks
        for blk in getattr(m, "content_blocks", []) or []:
            if isinstance(blk, dict) and blk.get("type") == "reasoning":
                t = blk.get("text") or blk.get("reasoning") or ""
                if t:
                    out.append(t)
    return "\n\n---\n\n".join(out).strip()


def ask(agent, question: str, thread_id: str = "default") -> dict:
    """
    Run the agent. Returns a dict with both answer and reasoning trace.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=cfg,
    )
    messages = result.get("messages", [])
    final = messages[-1].content if messages else ""
    if isinstance(final, list):
        # New content_blocks format — extract text parts
        final = "".join(blk.get("text", "") if isinstance(blk, dict) else str(blk) for blk in final)
    thinking = _extract_thinking(messages)
    return {
        "answer": final,
        "thinking": thinking,
        "tool_calls": _count_tool_calls(messages),
    }


def _count_tool_calls(messages: list) -> int:
    n = 0
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            n += len(m.tool_calls)
    return n
