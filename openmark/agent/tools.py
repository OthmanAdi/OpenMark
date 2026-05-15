"""
LangGraph tools for OpenMark — typed Pydantic returns, no markdown lies.

Every tool returns a ToolResult; the .to_compact_markdown() rendering goes
into the tool message. URLs live in named fields so the synthesizer can
validate citations against state["seen_urls"].
"""

import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from langchain_core.tools import tool
from openmark.embeddings.factory import get_embedder
from openmark.stores import neo4j_store
from openmark.agent.schemas import BookmarkHit, ToolResult

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


# Cache query embeddings — pplx-embed query model is deterministic, so the same
# text always produces the same vector. Vague queries ("rag", "agents") repeat
# across turns and tool fan-outs; caching saves 100-500ms per repeat call.
@lru_cache(maxsize=512)
def _embed_query_tuple(query: str) -> tuple[float, ...]:
    return tuple(_get_embedder().embed_query(query))


def _embed_query(query: str) -> list[float]:
    return list(_embed_query_tuple(query))


def warm_up() -> None:
    """
    Pre-load embedder weights, open the Neo4j driver, and prime the stats cache
    before the first user query. Called once from build_agent(). Failures are
    non-fatal — first real call will pay the cost and surface the same error.
    """
    try:
        _embed_query("warm")
    except Exception as e:
        print(f"[warm_up] embedder skipped: {e}")
    try:
        neo4j_store.get_driver()
        neo4j_store.get_stats()
    except Exception as e:
        print(f"[warm_up] neo4j skipped: {e}")


def _row_to_hit(r: dict) -> BookmarkHit:
    return BookmarkHit(
        url=r.get("url", ""),
        title=r.get("title", "") or "",
        similarity=float(r.get("similarity", 0) or 0),
        bm_score=float(r.get("bm_score", 0) or 0),
        source=r.get("source", "") or "",
        category=r.get("category"),
        tags=[t for t in (r.get("tags") or []) if t],
        community_id=r.get("community_id"),
    )


# ── Primary retrieval tools ────────────────────────────────────────────────────

@tool
def search_semantic(query: str, n: int = 10) -> str:
    """
    Vector + graph search across all 11,000+ bookmarks.
    USE THIS FIRST for any query about saved content.
    Returns up to n hits ranked by cosine similarity to the query embedding.
    """
    try:
        q_embed = _embed_query(query)
        rows = neo4j_store.vector_search(q_embed, n=n)
        hits = [_row_to_hit(r) for r in rows]
        return ToolResult(
            hits=hits, strategy="semantic",
            query_echo=query, total_found=len(hits),
        ).to_compact_markdown()
    except Exception as e:
        return ToolResult(strategy="semantic", query_echo=query, note=f"ERROR: {e}").to_compact_markdown()


@tool
def search_by_category(category: str, query: str = "", n: int = 15) -> str:
    """
    Vector search restricted to one of the 19 canonical categories.
    Categories: 'RAG & Vector Search', 'Agent Development', 'LangChain / LangGraph',
    'MCP & Tool Use', 'Context Engineering', 'LLM Fine-tuning', 'AI Tools & Platforms',
    'GitHub Repos & OSS', 'Learning & Courses', 'YouTube & Video', 'Web Development',
    'Cloud & Infrastructure', 'Data Science & ML', 'Knowledge Graphs & Neo4j',
    'Career & Jobs', 'Finance & Crypto', 'Design & UI/UX', 'News & Articles',
    'Entertainment & Other'.
    Pass an empty query to get the top-scored bookmarks in the category.
    """
    try:
        q_embed = _embed_query(query or category)
        rows = neo4j_store.vector_search(q_embed, n=n, category=category)
        hits = [_row_to_hit(r) for r in rows]
        return ToolResult(
            hits=hits, strategy="category",
            query_echo=f"{category} :: {query}", total_found=len(hits),
        ).to_compact_markdown()
    except Exception as e:
        return ToolResult(strategy="category", query_echo=category, note=f"ERROR: {e}").to_compact_markdown()


@tool
def search_by_community(query: str, n: int = 20) -> str:
    """
    Find the Louvain topic community closest to the query, return all its members.
    Use for broad-topic discovery ('everything I saved about agents').
    Returns hits in one tight cluster — different from semantic search which
    returns the global top-n.
    """
    try:
        q_embed = _embed_query(query)
        rows = neo4j_store.search_by_community(q_embed, n=n)
        hits = [_row_to_hit(r) for r in rows]
        note = "" if hits else "Louvain not run yet (GDS plugin?) — try search_semantic."
        return ToolResult(
            hits=hits, strategy="community",
            query_echo=query, total_found=len(hits), note=note,
        ).to_compact_markdown()
    except Exception as e:
        return ToolResult(strategy="community", query_echo=query, note=f"ERROR: {e}").to_compact_markdown()


@tool
def find_by_tag(tag: str, n: int = 20) -> str:
    """
    Find bookmarks tagged exactly with the given tag (case-insensitive).
    Use when you know the exact tag (e.g. 'rag', 'agents', 'fine-tuning').
    """
    try:
        rows = neo4j_store.find_by_tag(tag, limit=n)
        hits = [BookmarkHit(url=r["url"], title=r["title"], bm_score=float(r.get("score", 0) or 0)) for r in rows]
        note = f"No matches — try search_semantic('{tag}')." if not hits else ""
        return ToolResult(
            hits=hits, strategy="tag",
            query_echo=tag, total_found=len(hits), note=note,
        ).to_compact_markdown()
    except Exception as e:
        return ToolResult(strategy="tag", query_echo=tag, note=f"ERROR: {e}").to_compact_markdown()


@tool
def graph_expand(url: str) -> str:
    """
    Given a specific bookmark URL, return its tags, similar bookmarks (SIMILAR_TO edges),
    and community peers. Use AFTER finding one interesting bookmark to discover related items.
    """
    try:
        text = neo4j_store.graph_expand(url)
        return f"[strategy=graph_expand] URL: {url}\n{text}"
    except Exception as e:
        return f"[strategy=graph_expand] ERROR: {e}"


@tool
def explore_tag_cluster(tag: str, n: int = 25) -> str:
    """
    Tags 2 hops away from the given tag via CO_OCCURS_WITH edges → their bookmarks.
    Use to expand from one tag into a wider thematic neighborhood.
    """
    try:
        rows = neo4j_store.find_tag_cluster(tag, hops=2, limit=n)
        hits = []
        for r in rows:
            hits.append(BookmarkHit(
                url=r["url"], title=r["title"],
                bm_score=float(r.get("score", 0) or 0),
                tags=[r.get("via_tag", "")] if r.get("via_tag") else [],
            ))
        return ToolResult(
            hits=hits, strategy="tag",
            query_echo=f"cluster around '{tag}'", total_found=len(hits),
        ).to_compact_markdown()
    except Exception as e:
        return ToolResult(strategy="tag", query_echo=tag, note=f"ERROR: {e}").to_compact_markdown()


# ── New coverage tools — domain / source / linkedin / youtube ──────────────────

@tool
def find_by_domain(domain: str, n: int = 20) -> str:
    """
    Filter bookmarks by domain (e.g. 'github.com', 'arxiv.org', 'youtube.com', 'linkedin.com').
    Pass naked domain WITHOUT 'www.' or scheme.
    Use when user mentions a specific source site.
    """
    domain = domain.strip().lower().replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
    try:
        rows = neo4j_store.query("""
            MATCH (b:Bookmark)-[:FROM_DOMAIN]->(d:Domain)
            WHERE toLower(d.name) = $domain OR toLower(d.name) ENDS WITH $suffix
            RETURN b.url AS url, b.title AS title, b.score AS bm_score,
                   b.source AS source, b.category AS category
            ORDER BY b.score DESC LIMIT $n
        """, {"domain": domain, "suffix": "." + domain, "n": n})
        hits = [_row_to_hit(r) for r in rows]
        return ToolResult(
            hits=hits, strategy="domain",
            query_echo=domain, total_found=len(hits),
        ).to_compact_markdown()
    except Exception as e:
        return ToolResult(strategy="domain", query_echo=domain, note=f"ERROR: {e}").to_compact_markdown()


@tool
def find_by_source(source: str, query: str = "", n: int = 20) -> str:
    """
    Filter by source: 'linkedin', 'youtube_liked_videos', 'youtube_watch_later',
    'youtube_playlists', 'raindrop', 'edge', 'dailydev', 'manual'.
    Optional query does semantic ranking within the source.
    """
    try:
        if query.strip():
            q_embed = _embed_query(query)
            rows = neo4j_store.vector_search(q_embed, n=n, source=source)
        else:
            rows = neo4j_store.query("""
                MATCH (b:Bookmark) WHERE b.source = $source
                RETURN b.url AS url, b.title AS title, b.score AS bm_score,
                       b.source AS source, b.category AS category
                ORDER BY b.score DESC LIMIT $n
            """, {"source": source, "n": n})
        hits = [_row_to_hit(r) for r in rows]
        return ToolResult(
            hits=hits, strategy="source",
            query_echo=f"{source} :: {query}", total_found=len(hits),
        ).to_compact_markdown()
    except Exception as e:
        return ToolResult(strategy="source", query_echo=source, note=f"ERROR: {e}").to_compact_markdown()


@tool
def search_linkedin(query: str, n: int = 15) -> str:
    """Semantic search restricted to saved LinkedIn posts only."""
    return find_by_source.invoke({"source": "linkedin", "query": query, "n": n})


@tool
def search_youtube(query: str, n: int = 15) -> str:
    """Semantic search across YouTube saves (liked + watch_later + playlists)."""
    try:
        q_embed = _embed_query(query)
        rows = neo4j_store.query("""
            CALL () {
                MATCH (b)
                  SEARCH b IN (
                    VECTOR INDEX bookmark_embedding
                    FOR $embedding
                    LIMIT 200
                  ) SCORE AS score
                WHERE b.source STARTS WITH 'youtube'
                RETURN b, score LIMIT $n
            }
            OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
            RETURN b.url AS url, b.title AS title, b.score AS bm_score,
                   b.source AS source, b.category AS category,
                   score AS similarity, collect(t.name)[..6] AS tags
            ORDER BY similarity DESC
        """, {"embedding": q_embed, "n": n})
        hits = [_row_to_hit(r) for r in rows]
        return ToolResult(
            hits=hits, strategy="source",
            query_echo=f"youtube :: {query}", total_found=len(hits),
        ).to_compact_markdown()
    except Exception:
        # Fallback if pre-filter SEARCH syntax not supported on this Neo4j version
        return find_by_source.invoke({"source": "youtube_liked_videos", "query": query, "n": n})


# ── Time-aware tools (depend on graph_hygiene backfilling created_at) ─────────

@tool
def find_recent(days: int = 7, query: str = "", n: int = 25) -> str:
    """
    Bookmarks added in the last N days. Currently only LinkedIn nodes have
    reliable timestamps (decoded from activity URN); Edge/Raindrop/YouTube
    need their own created_at backfill before they show up.
    Optional query semantically ranks within the time window.
    """
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        if query.strip():
            q_embed = _embed_query(query)
            rows = neo4j_store.query("""
                MATCH (b)
                  SEARCH b IN (
                    VECTOR INDEX bookmark_embedding
                    FOR $embedding
                    LIMIT 200
                  ) SCORE AS score
                WHERE b.created_at IS NOT NULL AND b.created_at >= datetime($cutoff)
                OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
                RETURN b.url AS url, b.title AS title, b.score AS bm_score,
                       b.source AS source, b.category AS category,
                       score AS similarity, collect(t.name)[..6] AS tags
                ORDER BY similarity DESC LIMIT $n
            """, {"embedding": q_embed, "cutoff": cutoff, "n": n})
        else:
            rows = neo4j_store.query("""
                MATCH (b:Bookmark)
                WHERE b.created_at IS NOT NULL AND b.created_at >= datetime($cutoff)
                OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
                RETURN b.url AS url, b.title AS title, b.score AS bm_score,
                       b.source AS source, b.category AS category,
                       collect(t.name)[..6] AS tags
                ORDER BY b.created_at DESC LIMIT $n
            """, {"cutoff": cutoff, "n": n})
        hits = [_row_to_hit(r) for r in rows]
        note = ("Only LinkedIn nodes have timestamps. Edge / Raindrop / YouTube "
                "need backfill before they appear here.") if not hits else ""
        return ToolResult(
            hits=hits, strategy="source",
            query_echo=f"last {days}d :: {query}",
            total_found=len(hits), note=note,
        ).to_compact_markdown()
    except Exception as e:
        return ToolResult(strategy="source", query_echo=f"last {days}d", note=f"ERROR: {e}").to_compact_markdown()


@tool
def search_by_date_range(from_iso: str, to_iso: str, query: str = "", n: int = 30) -> str:
    """
    Bookmarks created between two ISO timestamps. Examples:
      from_iso='2026-05-01', to_iso='2026-05-14'.
    Optional query semantically ranks within the window.
    """
    try:
        if query.strip():
            q_embed = _embed_query(query)
            rows = neo4j_store.query("""
                MATCH (b)
                  SEARCH b IN (
                    VECTOR INDEX bookmark_embedding
                    FOR $embedding
                    LIMIT 300
                  ) SCORE AS score
                WHERE b.created_at IS NOT NULL
                  AND b.created_at >= datetime($from_iso)
                  AND b.created_at <  datetime($to_iso)
                OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
                RETURN b.url AS url, b.title AS title, b.score AS bm_score,
                       b.source AS source, b.category AS category,
                       score AS similarity, collect(t.name)[..6] AS tags
                ORDER BY similarity DESC LIMIT $n
            """, {"embedding": q_embed, "from_iso": from_iso, "to_iso": to_iso, "n": n})
        else:
            rows = neo4j_store.query("""
                MATCH (b:Bookmark)
                WHERE b.created_at IS NOT NULL
                  AND b.created_at >= datetime($from_iso)
                  AND b.created_at <  datetime($to_iso)
                OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
                RETURN b.url AS url, b.title AS title, b.score AS bm_score,
                       b.source AS source, b.category AS category,
                       collect(t.name)[..6] AS tags
                ORDER BY b.created_at DESC LIMIT $n
            """, {"from_iso": from_iso, "to_iso": to_iso, "n": n})
        hits = [_row_to_hit(r) for r in rows]
        return ToolResult(
            hits=hits, strategy="source",
            query_echo=f"{from_iso}..{to_iso} :: {query}",
            total_found=len(hits),
        ).to_compact_markdown()
    except Exception as e:
        return ToolResult(strategy="source", query_echo=f"{from_iso}..{to_iso}", note=f"ERROR: {e}").to_compact_markdown()


@tool
def get_bookmark_full(url: str) -> str:
    """
    Full record for one bookmark: title, category, tags, source, score,
    created_at, SIMILAR_TO neighbors (titles + urls), community peers.
    Use for newsletter context-building after you picked which bookmarks to feature.
    """
    try:
        rows = neo4j_store.query("""
            MATCH (b:Bookmark {url: $url})
            OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
            OPTIONAL MATCH (b)-[:SIMILAR_TO]->(s:Bookmark)
            OPTIONAL MATCH (b)-[:IN_COMMUNITY]->(c:Community)<-[:IN_COMMUNITY]-(peer:Bookmark)
            WHERE peer.url <> $url
            RETURN b.url AS url, b.title AS title, b.category AS category,
                   b.source AS source, b.score AS score,
                   b.created_at AS created_at,
                   collect(DISTINCT t.name)                                AS tags,
                   collect(DISTINCT {url: s.url, title: s.title})[..6]     AS similar,
                   collect(DISTINCT peer.title)[..6]                       AS community_peers
        """, {"url": url})
        if not rows:
            return f"[strategy=detail] Bookmark not found: {url}"
        r = rows[0]
        lines = [
            f"[strategy=detail] {r.get('title','(no title)')}",
            f"  URL:        {r.get('url')}",
            f"  Source:     {r.get('source','—')}",
            f"  Category:   {r.get('category','—')}",
            f"  Score:      {r.get('score',0)}",
            f"  Created:    {r.get('created_at','—')}",
            f"  Tags:       {', '.join(r.get('tags') or []) or '—'}",
            f"  SIMILAR_TO ({len(r.get('similar') or [])}):",
        ]
        for s in (r.get("similar") or [])[:6]:
            lines.append(f"    - {s.get('title','')} — {s.get('url','')}")
        if r.get("community_peers"):
            lines.append(f"  Community peers ({len(r['community_peers'])}):")
            for p in r["community_peers"][:6]:
                lines.append(f"    - {p}")
        return "\n".join(lines)
    except Exception as e:
        return f"[strategy=detail] ERROR: {e}"


# ── Stats + raw Cypher (read-only) ─────────────────────────────────────────────

@tool
def get_stats() -> str:
    """Get total bookmark/tag/category/community counts in the knowledge base."""
    try:
        s = neo4j_store.get_stats()
        return (
            f"[strategy=stats] Knowledge base:\n"
            f"  Bookmarks:   {s.get('bookmarks', 0):,}\n"
            f"  Tags:        {s.get('tags', 0):,}\n"
            f"  Categories:  {s.get('categories', 0)}\n"
            f"  Communities: {s.get('communities', 0)}"
        )
    except Exception as e:
        return f"[strategy=stats] ERROR: {e}"


_WRITE_CYPHER = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|CALL\s+db\.|CALL\s+apoc|CALL\s+gds\.[a-z]+\.write)\b",
    re.IGNORECASE,
)


@tool
def run_cypher(cypher: str) -> str:
    """
    Run a READ-ONLY Cypher query against the Neo4j knowledge graph.
    Writes (CREATE/MERGE/SET/DELETE/REMOVE/DROP) are blocked.
    Use for advanced graph questions only — prefer the higher-level tools.
    """
    if _WRITE_CYPHER.search(cypher):
        return "[strategy=cypher] BLOCKED: write operations are not allowed via this tool."
    try:
        rows = neo4j_store.query(cypher)
        if not rows:
            return "[strategy=cypher] No rows."
        return "[strategy=cypher]\n" + "\n".join(str(r) for r in rows[:30])
    except Exception as e:
        return f"[strategy=cypher] ERROR: {e}"


# ── Web research tools ───────────────────────────────────────────────────────
# Built 2026-05-14. Implementations live in openmark/agent/web.py so the
# transport (httpx, DDG, GitHub, Reddit) is testable without a tool decorator.
from openmark.agent import web as _web


@tool
def web_search(query: str, n: int = 8) -> str:
    """
    Search the open web. Use when the user's question needs FRESH information
    beyond Ahmad's saved bookmarks — current events, latest releases, "what
    does the internet say about X."

    Provider stack (auto-fallback): Tavily (if TAVILY_API_KEY set) > Brave
    (if BRAVE_API_KEY set) > DuckDuckGo HTML (no key).

    Returns up to n hits as a numbered markdown list with title, url, snippet,
    and provider tag.
    """
    try:
        hits = _web.web_search(query, n=n)
        if not hits:
            return f"[strategy=web_search] No hits for '{query}'."
        lines = [f"[strategy=web_search] {len(hits)} hits for '{query}' (via {hits[0].get('source')}):"]
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. {h.get('title','')}")
            lines.append(f"   URL: {h.get('url','')}")
            sn = (h.get("snippet") or "").replace("\n", " ").strip()
            if sn:
                lines.append(f"   {sn[:240]}")
        return "\n".join(lines)
    except Exception as e:
        return f"[strategy=web_search] ERROR: {e}"


@tool
def web_fetch(url: str, max_chars: int = 12000) -> str:
    """
    Fetch a single URL and return the main content as clean markdown.
    Skips nav, footer, ads, scripts. Caps output at max_chars.
    Use AFTER web_search to read the most promising hits.
    """
    try:
        doc = _web.web_fetch(url, max_chars=max_chars)
        if doc["status"] != "ok":
            return f"[strategy=web_fetch] FAILED url={url} :: {doc['status']}"
        return (
            f"[strategy=web_fetch] OK url={url}\n"
            f"Title: {doc['title']}\n"
            f"Chars: {len(doc['markdown'])}\n\n"
            f"{doc['markdown']}"
        )
    except Exception as e:
        return f"[strategy=web_fetch] ERROR: {e}"


@tool
def github_repo_intel(slug_or_url: str, days: int = 30) -> str:
    """
    Pull a quick intel snapshot of a public GitHub repo:
      meta (stars, forks, language, license, default_branch, topics, dates),
      README (markdown, truncated to 16k chars),
      recent commits on default branch over the last `days`,
      top 15 open PRs.

    Pass either a repo slug ('owner/name') or a github.com URL.
    Use this BEFORE web_search when the user names a specific repo.
    """
    try:
        intel = _web.github_repo_intel(slug_or_url, days=days)
        if intel.get("status", "").startswith("error"):
            return f"[strategy=github] {intel['status']}"
        m = intel.get("meta", {})
        lines = [
            f"[strategy=github] slug={intel.get('slug')}",
            f"  {m.get('name')} ({m.get('language')}, {m.get('license')})",
            f"  stars={m.get('stars'):,}  forks={m.get('forks'):,}  watchers={m.get('watchers')}",
            f"  open_issues={m.get('open_issues')}  default_branch={m.get('default_branch')}",
            f"  topics: {', '.join(m.get('topics', []) or []) or '—'}",
            f"  homepage: {m.get('homepage') or '—'}",
            f"  pushed_at: {m.get('pushed_at')}",
            "",
            f"## README (first 8k chars)",
            (intel.get("readme") or "")[:8000],
            "",
            f"## Recent commits ({len(intel.get('recent_commits') or [])} in last {intel.get('recent_commits_window_days', days)}d)",
        ]
        for c in (intel.get("recent_commits") or [])[:20]:
            lines.append(f"  {c['sha']}  {c['date'][:10]}  {c['author']}: {c['message']}")
        if intel.get("open_prs"):
            lines.append("")
            lines.append(f"## Open PRs (top {len(intel['open_prs'])})")
            for pr in intel["open_prs"]:
                lines.append(f"  #{pr['number']}  {pr['title']}  ({pr['user']}, updated {pr['updated_at'][:10]})")
                lines.append(f"    {pr['url']}")
        return "\n".join(lines)
    except Exception as e:
        return f"[strategy=github] ERROR: {e}"


@tool
def web_extract(urls: list[str], depth: str = "advanced") -> str:
    """
    Tavily-powered multi-URL extraction. Pass a list of URLs (up to ~20),
    get clean text content for each in ONE call — faster + more reliable
    than calling web_fetch multiple times. Best for JS-heavy pages,
    GitHub blob views, LinkedIn posts, paywall previews.

    depth: 'basic' (faster, cheaper) or 'advanced' (richer, slower).
    Returns markdown with per-URL sections.
    Requires TAVILY_API_KEY in .env.
    """
    if not urls:
        return "[strategy=web_extract] No URLs provided."
    try:
        results = _web.tavily_extract(urls, depth=depth)
        if not results:
            return "[strategy=web_extract] Empty result. TAVILY_API_KEY missing or rate-limited."
        lines = [f"[strategy=web_extract] {len(results)} URL(s) extracted (depth={depth}):"]
        for r in results:
            rc = r.get("raw_content", "") or ""
            status = r.get("status", "ok")
            lines.append(f"\n=== {r.get('url')} ({status}, {len(rc)} chars) ===")
            if rc:
                lines.append(rc[:6000] + ("\n…(truncated)" if len(rc) > 6000 else ""))
        return "\n".join(lines)
    except Exception as e:
        return f"[strategy=web_extract] ERROR: {e}"


@tool
def web_crawl(
    seed_url: str,
    max_depth: int = 1,
    max_breadth: int = 5,
    limit: int = 8,
    instructions: str = "",
) -> str:
    """
    Tavily-powered crawl. Follow links from seed_url up to max_depth hops,
    max_breadth links per page, collecting up to `limit` pages.
    Optional `instructions` is a natural-language steer Tavily uses to
    score which links to follow ('focus on documentation pages', 'skip
    marketing copy', etc).

    Use to map a site, gather all docs pages, or sweep a repo's blog +
    related posts in one call. Returns per-page extracted content.
    Requires TAVILY_API_KEY in .env.
    """
    if not seed_url:
        return "[strategy=web_crawl] No seed URL provided."
    try:
        results = _web.tavily_crawl(
            seed_url, max_depth=max_depth, max_breadth=max_breadth,
            limit=limit, instructions=instructions or None,
        )
        if not results:
            return f"[strategy=web_crawl] No pages crawled from {seed_url}. TAVILY_API_KEY missing or rate-limited."
        lines = [f"[strategy=web_crawl] {len(results)} page(s) from {seed_url} "
                 f"(depth={max_depth}, breadth={max_breadth}):"]
        if instructions:
            lines.append(f"  instructions: {instructions[:120]}")
        for r in results:
            rc = r.get("raw_content", "") or ""
            lines.append(f"\n=== {r.get('url')} ({len(rc)} chars) ===")
            if rc:
                lines.append(rc[:4000] + ("\n…(truncated)" if len(rc) > 4000 else ""))
        return "\n".join(lines)
    except Exception as e:
        return f"[strategy=web_crawl] ERROR: {e}"


@tool
def reddit_search(query: str, subreddit: str = "", n: int = 15) -> str:
    """
    Search Reddit. Optional `subreddit` narrows to one sub (e.g. 'muapi',
    'LocalLLaMA', 'StableDiffusion'). Returns posts with score, comments,
    permalink, and body snippet. No auth; uses public JSON endpoint.
    """
    try:
        sub = (subreddit or "").strip() or None
        posts = _web.reddit_search(query, subreddit=sub, n=n)
        if not posts:
            return f"[strategy=reddit] No hits for '{query}'" + (f" in r/{sub}" if sub else "")
        lines = [f"[strategy=reddit] {len(posts)} posts for '{query}'" + (f" in r/{sub}" if sub else "") + ":"]
        for i, p in enumerate(posts, 1):
            lines.append(f"{i}. [{p['score']:>4}↑ {p['num_comments']:>3}💬] r/{p['subreddit']} — {p['title']}")
            lines.append(f"   {p['url']}")
            if p.get("external_url") and p["external_url"] != p["url"]:
                lines.append(f"   → {p['external_url']}")
            if p.get("body_snippet"):
                lines.append(f"   {p['body_snippet']}")
        return "\n".join(lines)
    except Exception as e:
        return f"[strategy=reddit] ERROR: {e}"


ALL_TOOLS = [
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
    get_bookmark_full,
    get_stats,
    run_cypher,
    # Web research
    web_search,
    web_fetch,
    web_extract,
    web_crawl,
    github_repo_intel,
    reddit_search,
]
