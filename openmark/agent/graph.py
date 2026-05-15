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

import re
import time
from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
    before_model,
    after_model,
)
from langgraph.checkpoint.memory import MemorySaver

from openmark import config
from openmark.agent.llms import build_executor, build_classifier
from openmark.agent.tools import ALL_TOOLS, warm_up as _warm_up_tools
from openmark.agent.middleware import (
    OpenMarkSkillMiddleware,
    slash_skill_loader,
    tool_event_middleware,
    drain_events,
    log,
)
from openmark.agent import skills as _skill_loader


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


# ── Mode classification ──────────────────────────────────────────────────────
# Slash command → mode lookup. The skill name carries the user's intent, so we
# can skip the classifier LLM call entirely when slash is used. ~150-400ms saved
# per slash query on the first turn.
_SLASH_TO_MODE: dict[str, str] = {
    "fast-search":     "fast",
    "deep-research":   "deep",
    "newsletter":      "newsletter",
    "weekly-digest":   "digest",
    "bookmark-dive":   "dive",
    "repo-research":   "deep",
    "newsletter-essay":      "newsletter",
    "newsletter-roundup":    "newsletter",
    "newsletter-thread":     "newsletter",
    "newsletter-comparison": "newsletter",
}

# Regex pre-classifier — catches the easy 80% so the LLM only handles ambiguous
# queries. Patterns derived from the labels in CLASSIFIER_PROMPT below.
_URL_RE = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
_DIVE_VERBS = re.compile(r"\b(expand|dig\s+into|neighbors\s+of|related\s+to|dive\s+into)\b", re.IGNORECASE)
_NEWSLETTER_RE = re.compile(r"\bnewsletter\b", re.IGNORECASE)
_DIGEST_RE = re.compile(
    r"\b(this\s+week|last\s+\d+\s*(day|week)s?|past\s+\d+\s*days?|"
    r"weekly|yesterday|today|recap)\b",
    re.IGNORECASE,
)
_DEEP_RE = re.compile(
    r"\b(research|compare|comparison|landscape|deep[-\s]?dive|"
    r"overview|state\s+of|survey|map\s+out)\b",
    re.IGNORECASE,
)


def _heuristic_mode(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    if _URL_RE.search(t) and _DIVE_VERBS.search(t):
        return "dive"
    if _NEWSLETTER_RE.search(t):
        return "newsletter"
    if _DIGEST_RE.search(t):
        return "digest"
    if _DEEP_RE.search(t):
        return "deep"
    return None


def _slash_mode(text: str) -> str | None:
    """If text starts with /<skill>, map skill → mode. Returns None on no match."""
    name, _ = _skill_loader.parse_slash(text or "")
    if not name:
        return None
    short = name.lower().lstrip("/")
    if short.startswith("openmark-"):
        short = short[len("openmark-"):]
    return _SLASH_TO_MODE.get(short)


# ── Custom middleware ─────────────────────────────────────────────────────────

_STATS_MARKER = "<!-- openmark-stats-injected -->"


@before_model
def inject_live_stats(state: AgentState, runtime: Any) -> dict | None:
    """
    On the first turn:
      1. Pick a mode (slash → heuristic regex → LLM classifier fallback).
      2. Rewrite the system prompt with live Neo4j counts AND the mode hint.

    Idempotent: a sentinel marker in the injected SystemMessage prevents
    re-injection on subsequent turns and lets us co-exist with the slash
    pre-loader (which also injects a SystemMessage).
    """
    messages = state.get("messages") or []
    if not messages:
        return None

    # Already injected? Check our marker in any existing system message.
    for m in messages:
        if getattr(m, "type", "") != "system":
            continue
        content = getattr(m, "content", "")
        if isinstance(content, str) and _STATS_MARKER in content:
            return None
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and _STATS_MARKER in (b.get("text", "") or ""):
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

    # Pick a mode. Cheap paths first; LLM classifier only when nothing else fires.
    mode = _slash_mode(last_user_text) or _heuristic_mode(last_user_text)
    classifier_source = "slash" if _slash_mode(last_user_text) else ("heuristic" if mode else "")

    if mode is None:
        mode = "fast"
        try:
            if last_user_text.strip():
                t0 = time.time()
                cls_resp = _get_classifier().invoke(CLASSIFIER_PROMPT.format(query=last_user_text.strip()))
                raw = (cls_resp.content if hasattr(cls_resp, "content") else str(cls_resp))
                if isinstance(raw, list):
                    raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
                raw = str(raw).strip().lower()
                for cand in ("newsletter", "digest", "deep", "dive", "fast"):
                    if cand in raw:
                        mode = cand
                        break
                classifier_source = "llm"
                log.info(f"[classifier] mode={mode} (raw={raw[:60]!r}) in {round((time.time()-t0)*1000,1)}ms")
        except Exception as e:
            log.info(f"[classifier] failed: {e}; defaulting mode=fast")
    else:
        log.info(f"[classifier] mode={mode} (source={classifier_source}, no LLM call)")

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

    prompt += f"\n\n{_STATS_MARKER}\nMODE: {mode}  (classifier set this; tailor your effort accordingly)\n"
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
    """Compile the agent graph with the configured provider + middleware stack."""
    # Pre-load embedder weights + Neo4j driver + stats cache so the first user
    # query doesn't pay model-load and connection-handshake latency. Failures
    # are non-fatal and surfaced again when the first real call retries.
    _t_warm = time.time()
    _warm_up_tools()
    log.info(f"[warm_up] tools warmed in {round((time.time()-_t_warm)*1000)}ms")

    llm = build_executor()
    provider = (getattr(config, "AGENT_PROVIDER", "azure") or "azure").lower()
    if provider == "local":
        print("=" * 72)
        print(f"AGENT PROVIDER = LOCAL   (Azure is NOT being used)")
        print(f"  endpoint : {config.BONSAI_URL}")
        print(f"  model    : {config.BONSAI_MODEL}")
        print("  all roles (executor, classifier, summarizer) -> same local endpoint")
        print("=" * 72)
    else:
        print(f"AGENT PROVIDER = AZURE   model={config.AZURE_DEPLOYMENT_EXECUTOR} "
              f"(Responses API, reasoning={config.AZURE_REASONING_EXECUTOR}, summary=detailed)")

    # Cheap model for SummarizationMiddleware — same gpt-5-mini we already build for the classifier.
    summarizer_llm = build_classifier()

    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        # NO response_format. Skills tell the agent how to structure output
        # (title + tl;dr + sections + sources for reports; numbered list for
        # quick answers). Enforcing ToolStrategy(Union[QuickAnswer, Report])
        # made the agent loop whenever it wanted to ask a clarifying question,
        # because clarification doesn't fit either schema. UI detects "looks
        # like a report" from the final markdown and auto-saves it.
        # System prompt is dynamic via @before_model — pass minimal here.
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(
            bookmarks=11000, tags=5000, categories=19, communities=0,
        ),
        middleware=[
            # 1. Live KB stats + mode hint. Runs FIRST so it can read the raw
            #    user message (including any leading `/skill-name`) to pick the
            #    mode via slash → heuristic → LLM classifier waterfall before
            #    slash_skill_loader rewrites the message.
            inject_live_stats,
            # 2. Slash-command pre-loader: if user typed `/<skill>`, eager-inject the
            #    SKILL.md body as a SystemMessage and strip the slash from their query.
            slash_skill_loader,
            # 3. Skill catalogue (progressive disclosure) — registers `load_skill` tool
            #    and appends the skill list to every system prompt so the agent can
            #    self-select recipes when the user didn't use a slash command.
            OpenMarkSkillMiddleware(),
            # 4. Built-in planner — the agent writes its own todo list as the first action.
            TodoListMiddleware(),
            # 5. Auto-compaction: once history crosses 8k tokens or 30 messages,
            #    summarize older turns via gpt-5-mini, keep the most recent 12.
            #    Prevents "thread limit exceeded" from accumulating across turns.
            SummarizationMiddleware(
                model=summarizer_llm,
                trigger=[("tokens", 8000), ("messages", 30)],
                keep=("messages", 12),
            ),
            # 6. Resilient against transient Neo4j hiccups.
            ToolRetryMiddleware(max_retries=2),
            # 7. Capture tool events so the UI can render them live as cards.
            tool_event_middleware,
            # 8. Hard ceiling against runaway loops PER TURN ONLY.
            #    No thread_limit — let SummarizationMiddleware handle long chats.
            ModelCallLimitMiddleware(run_limit=15),
        ],
        checkpointer=MemorySaver(),
    )
    return agent


# ── Ask with thinking surfaced ─────────────────────────────────────────────────

def _extract_thinking(messages: list) -> str:
    """Pull reasoning summaries out of AIMessage.additional_kwargs / content_blocks."""
    out = []
    found_keys = []
    for m in messages:
        mtype = getattr(m, "type", "?")
        ak = getattr(m, "additional_kwargs", {}) or {}
        if ak:
            found_keys.append(f"{mtype}:{list(ak.keys())}")
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
        # Some langchain-openai versions stash it under m.content as list of dicts
        c = getattr(m, "content", None)
        if isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") in ("reasoning", "thinking"):
                    t = blk.get("text") or blk.get("reasoning") or ""
                    if t:
                        out.append(t)
    log.info(f"[thinking] sources_found={found_keys[:8]} chars={sum(len(x) for x in out)}")
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


# ── Streaming variant — yields events as the agent runs ──────────────────────

def ask_stream(agent, question: str, thread_id: str = "default"):
    """
    Generator. Yields a sequence of typed events as the agent runs:

      {"kind": "user",     "text": ...}           — once, at start
      {"kind": "tool_start", "tool": ..., "args": ...}
      {"kind": "tool_end",   "tool": ..., "duration_ms": ..., "preview": ...}
      {"kind": "tool_error", "tool": ..., "error": ...}
      {"kind": "thinking", "text": ...}           — at end, full reasoning trace
      {"kind": "final",    "text": ..., "tool_calls": N}

    The UI consumes this and renders each event as a chat bubble or inline card.

    Uses agent.stream(stream_mode='updates') under the hood, polling
    middleware.drain_events between updates to interleave tool events.
    """
    log.info(f"[ask_stream] q={question[:80]!r} thread={thread_id}")
    yield {"kind": "user", "text": question}

    # Drain any stale events from a previous run on this thread.
    drain_events(thread_id)

    cfg = {"configurable": {"thread_id": thread_id}}
    chunk_count = 0
    seen_msg_ids: set = set()
    streamed_turn_thinking = False

    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config=cfg,
            stream_mode="updates",
        ):
            chunk_count += 1
            log.info(f"[stream] chunk #{chunk_count} nodes={list((chunk or {}).keys())}")

            # Surface per-turn reasoning the moment a new AIMessage lands.
            # gpt-5.3-codex with summary=detailed stashes reasoning as content
            # blocks of type="reasoning" inside AIMessage.content.
            for node_name, node_state in (chunk or {}).items():
                if not isinstance(node_state, dict):
                    continue
                for m in (node_state.get("messages", []) or []):
                    mid = getattr(m, "id", None) or id(m)
                    if mid in seen_msg_ids:
                        continue
                    seen_msg_ids.add(mid)
                    if getattr(m, "type", "") != "ai":
                        continue
                    t = _extract_thinking([m])
                    if t:
                        log.info(f"[turn_thinking] node={node_name} chars={len(t)}")
                        streamed_turn_thinking = True
                        yield {"kind": "turn_thinking", "text": t}

            # Flush tool events that landed during this update.
            for ev in drain_events(thread_id):
                phase = ev.get("phase")
                if phase == "start":
                    yield {"kind": "tool_start",
                           "tool": ev.get("tool"), "args": ev.get("args", {})}
                elif phase == "end":
                    yield {"kind": "tool_end",
                           "tool": ev.get("tool"),
                           "duration_ms": ev.get("duration_ms"),
                           "preview": ev.get("result_preview", "")}
                elif phase == "error":
                    yield {"kind": "tool_error",
                           "tool": ev.get("tool"),
                           "error": ev.get("error", "")}
    except Exception as e:
        log.info(f"[ask_stream] exception={e!r}")
        yield {"kind": "tool_error", "tool": "agent", "error": str(e)}
        return

    # Pull authoritative final state from the checkpointer.
    final_messages: list = []
    try:
        state = agent.get_state(cfg)
        if state and state.values:
            final_messages = state.values.get("messages", []) or []
        log.info(f"[ask_stream] final state has {len(final_messages)} messages")
    except Exception as e:
        log.info(f"[ask_stream] get_state failed: {e!r}")

    # Extract the LAST AIMessage's TEXT content only — skip reasoning blocks.
    # Azure Responses API returns AIMessage.content as a list of typed blocks.
    # Each text block can be a sentence, a paragraph, or even a single list item;
    # we must join them with a paragraph break ("\n\n") so markdown renders the
    # numbered list / sections instead of collapsing into one wrapped paragraph.
    final_text = ""
    if final_messages:
        last = final_messages[-1]
        c = getattr(last, "content", "") or ""
        if isinstance(c, list):
            text_blocks: list[str] = []
            for b in c:
                if isinstance(b, dict):
                    btype = b.get("type", "")
                    if btype in ("text", "output_text") or btype == "":
                        txt = (b.get("text", "") or "").strip()
                        if txt:
                            text_blocks.append(txt)
                    # Skip "reasoning" / "thinking" blocks — they go to turn_thinking.
                elif isinstance(b, str):
                    if b.strip():
                        text_blocks.append(b.strip())
            log.info(f"[ask_stream] final has {len(text_blocks)} text block(s); "
                     f"sizes={[len(t) for t in text_blocks][:8]}")
            # Paragraph break between blocks so markdown lists / headers survive.
            final_text = "\n\n".join(text_blocks)
        else:
            final_text = c

    thinking = _extract_thinking(final_messages)
    n_calls = _count_tool_calls(final_messages)
    log.info(f"[ask_stream] DONE n_calls={n_calls} thinking_chars={len(thinking)} "
             f"answer_chars={len(final_text)} streamed_turn_thinking={streamed_turn_thinking}")

    # Only emit the batched thinking trace if we DIDN'T stream per-turn thinking
    # (else the UI would show the same reasoning twice — inline + final).
    if thinking and not streamed_turn_thinking:
        yield {"kind": "thinking", "text": thinking}
    yield {"kind": "final",
           "text": final_text,
           "tool_calls": n_calls}
