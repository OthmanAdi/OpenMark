"""
Researcher sub-agent.

Owns the heavy retrieval surface — 21 OpenMark graph + web tools. The
orchestrator never sees these directly; it delegates retrieval missions to
`task_researcher(brief)` and gets back a compact anchor list.
"""

from __future__ import annotations

from openmark.agent.llms import build_researcher
from openmark.agent.subagents._common import (
    format_for_orchestrator,
    invoke_subagent,
    make_subagent_graph,
    task_tool,
)
from openmark.agent.tools import (
    explore_tag_cluster,
    find_all_in_range,
    find_by_domain,
    find_by_source,
    find_by_tag,
    find_recent,
    get_bookmark_full,
    get_stats,
    github_repo_intel,
    graph_expand,
    reddit_search,
    run_cypher,
    search_hybrid,
    search_by_category,
    search_by_community,
    search_by_date_range,
    search_linkedin,
    search_semantic,
    search_youtube,
    web_crawl,
    web_extract,
    web_fetch,
    web_search,
)


RESEARCHER_TOOLS = [
    # OpenMark graph retrieval (16)
    search_hybrid,
    search_semantic,
    search_by_category,
    search_by_community,
    find_by_tag,
    explore_tag_cluster,
    graph_expand,
    find_by_domain,
    find_by_source,
    search_linkedin,
    search_youtube,
    find_recent,
    search_by_date_range,
    find_all_in_range,     # NEW: full-recall, no-cap date window for "ALL OF THEM" queries
    get_bookmark_full,
    get_stats,
    run_cypher,            # read-only enforced inside the tool
    # Web research (6)
    web_search,
    web_fetch,
    web_extract,
    web_crawl,
    github_repo_intel,
    reddit_search,
]


RESEARCHER_PROMPT = """You are the OpenMark Researcher sub-agent.

Your one job: gather anchors and citations for the orchestrator's mission.

SOURCE SELECTION (read this carefully — wrong source = wasted turn)

You have THREE retrieval surfaces:

  A. OpenMark KB (Neo4j Graph RAG): search_hybrid (PREFERRED), search_semantic,
     search_by_category, search_by_community, find_by_tag, find_by_source,
     search_linkedin, search_youtube, find_recent, search_by_date_range,
     find_all_in_range, get_bookmark_full, graph_expand, explore_tag_cluster,
     run_cypher.
     -> Use when the brief references the user's SAVED content:
        "my bookmarks", "my saves", "what I read", "what I bookmarked",
        explicit URLs the user owns, weekly/monthly digests of saves.

  B. Open web: web_search (Tavily/Brave/DDG fallback), web_fetch, web_extract,
     web_crawl, reddit_search, github_repo_intel.
     -> Use when the brief is about CURRENT / EXTERNAL information:
        "what's the state of X", "compare A vs B", "research X",
        "trending today", "what's new in X", general research.

  C. TrendRadar MCP (when enabled): trendradar_get_latest_news,
     trendradar_get_trending_topics, trendradar_search_news,
     trendradar_analyze_sentiment, trendradar_aggregate_news, etc.
     -> Use when the brief is about REAL-TIME trends / hot topics /
        cross-platform aggregation.

The brief's language tells you the surface. DO NOT default to OpenMark when
the question isn't about saved content. DO NOT default to web when the
question is about saved content. When unclear, fan out parallel: one OpenMark
call AND one web call, then steer by what came back.

If a tool errors (e.g. Neo4j down -> "Couldn't connect to 127.0.0.1:7687"),
skip that surface for the rest of the turn and lean on the others. NEVER
retry the same dead tool.

TOOL CHOICE — OpenMark KB
- `search_hybrid` is the DEFAULT for OpenMark queries. It fuses BM25 over
  title + doc_text with pplx-embed cosine ANN via Reciprocal Rank Fusion.
  Catches exact terms ("muapi", "shai-hulud", "Hermes adapter") that pure
  vector search drops. Always prefer it over `search_semantic` unless the
  query is purely conceptual/paraphrased.
- `find_all_in_range(from_iso, to_iso?, page?)` returns EVERY bookmark in a
  date window with NO LIMIT and NO ranking. Use whenever the user asks for
  "all bookmarks from <date>", "everything I saved on/between X", or any
  phrasing demanding full recall. Paginate via `page` if the first call says
  more remain. This tool is the answer to "FULLY ALL OF THEM" style asks.
  Do NOT use `search_by_date_range` for those — it caps at n=30.
- `search_by_date_range` is for time-bounded SEMANTIC ranking ("the most
  relevant LangChain posts from May"). It does NOT return everything.
- For older nodes without `created_at` (older than the backfill window),
  the date tools won't see them — say so in `notes` if a date query
  returns thin results.

WORKFLOW
1. Parse the brief. Identify topic, time window, format hint, URLs.
   Classify which surface(s) the question demands per the rules above.
2. Fan out parallel tool calls when queries are independent.
3. For strong winners, escalate: graph_expand on URLs already in OpenMark,
   web_fetch / web_extract on URLs from web_search results.
4. Always return a structured anchor list, even if short. Mark each anchor
   with its source: "openmark" | "web" | "trendradar".

OUTPUT (always emit this JSON-ish block at the end of your final answer)
```json
{
  "anchors":   [{"url": "...", "title": "...", "why": "...", "source": "openmark|web"}, ...],
  "secondary": [{...}, ...],
  "notes":     "one short paragraph: what was rich, what was sparse, what to ignore"
}
```

CITATION DISCIPLINE
- Every URL you list MUST have appeared in a tool result this turn.
- Never invent a URL. If a tool returns thin results, say so in `notes`.
- Don't paraphrase URLs away from their canonical form.

You have access to read-only Cypher. Use it ONLY when higher-level tools cannot
answer (e.g. counting edges of a specific type, exact tag intersections).
"""


_RESEARCHER_GRAPH = None


def _get_graph():
    global _RESEARCHER_GRAPH
    if _RESEARCHER_GRAPH is None:
        _RESEARCHER_GRAPH = make_subagent_graph(
            model=build_researcher(),
            tools=RESEARCHER_TOOLS,
            system_prompt=RESEARCHER_PROMPT,
            run_limit=12,                       # research can chain many tool calls
            summarization_trigger=("tokens", 60_000),
            context_edit_trigger=80_000,
            context_edit_keep=6,
            mcp_scope="researcher",             # merges TrendRadar et al when enabled
        )
    return _RESEARCHER_GRAPH


@task_tool(
    "researcher",
    """Delegate research / retrieval to the OpenMark Researcher sub-agent.

Use for: finding bookmarks on a topic, building an anchor list before
composition, time-window digests, single-URL exploration, deep landscape
research. The sub-agent has access to 21 tools (15 graph + 6 web) and
returns a structured anchor list with citations the orchestrator can use.

Pass a clear mission `brief` — topic, angle, time window, any constraints.
Returns a compact summary + JSON anchor block + telemetry.
""",
)
def task_researcher(brief: str) -> str:
    result, dur = invoke_subagent(_get_graph(), brief, role="researcher")
    return format_for_orchestrator(role="researcher", result=result,
                                   duration_ms=dur, include_structured=False)
