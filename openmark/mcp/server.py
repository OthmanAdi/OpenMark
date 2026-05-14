"""
OpenMark MCP server — Graph RAG over Ahmad's personal knowledge graph.

Exposes ~15 tools for any MCP-compatible host (Claude Code, Claude Desktop).

Design choices:
  - Tools return STRUCTURED dicts (serialized from Pydantic ToolResult) so the
    host LLM cannot fabricate URLs — every URL lives in a named field.
  - All graph access is read-only. run_cypher blocks writes with a regex gate.
  - Personal use, no auth / no rate limit / no PII redaction.

Run standalone:  python -m openmark.mcp.server
Registered via .mcp.json — Claude Code auto-loads it from the project root.
"""

import sys
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.stderr.write("OpenMark MCP loading...\n")

from fastmcp import FastMCP
from openmark.embeddings.factory import get_embedder
from openmark.stores import neo4j_store
from openmark.agent.schemas import BookmarkHit, ToolResult

mcp = FastMCP("openmark")

# ── Lazy embedder ──────────────────────────────────────────────────────────────
_embedder = None
def _get_embedder():
    global _embedder
    if _embedder is None:
        sys.stderr.write("Loading pplx-embed (first call only)...\n")
        _embedder = get_embedder()
    return _embedder


# ── Helpers ────────────────────────────────────────────────────────────────────
def _row_to_hit(r: dict) -> BookmarkHit:
    return BookmarkHit(
        url=r.get("url", "") or "",
        title=(r.get("title") or "")[:300],
        similarity=float(r.get("similarity", 0) or 0),
        bm_score=float(r.get("bm_score", r.get("score", 0)) or 0),
        source=r.get("source", "") or "",
        category=r.get("category"),
        tags=[t for t in (r.get("tags") or []) if t],
        community_id=r.get("community_id"),
    )


def _result(hits: list[BookmarkHit], strategy: str, query_echo: str = "", note: str = "") -> dict:
    """ToolResult → dict so FastMCP serializes it cleanly into JSON tool output."""
    return ToolResult(
        hits=hits,
        strategy=strategy,
        query_echo=query_echo,
        total_found=len(hits),
        note=note,
    ).model_dump()


def _err(strategy: str, msg: str) -> dict:
    return ToolResult(strategy=strategy, note=f"ERROR: {msg}").model_dump()


# ── Primary retrieval ─────────────────────────────────────────────────────────

@mcp.tool
def search_semantic(query: str, n: int = 10) -> dict:
    """
    PRIMARY search. Vector + graph search across all bookmarks.
    USE THIS FIRST for any query about Ahmad's saved content.
    Returns up to n bookmarks ranked by cosine similarity to the query embedding,
    each with title, url, source, category, tags, similarity score, community_id.
    """
    try:
        q_embed = _get_embedder().embed_query(query)
        rows = neo4j_store.vector_search(q_embed, n=n)
        return _result([_row_to_hit(r) for r in rows], "semantic", query)
    except Exception as e:
        return _err("semantic", str(e))


@mcp.tool
def search_by_category(category: str, query: str = "", n: int = 15) -> dict:
    """
    Vector search restricted to one of the 19 canonical categories.
    Categories: 'RAG & Vector Search', 'Agent Development', 'LangChain / LangGraph',
    'MCP & Tool Use', 'Context Engineering', 'LLM Fine-tuning', 'AI Tools & Platforms',
    'GitHub Repos & OSS', 'Learning & Courses', 'YouTube & Video', 'Web Development',
    'Cloud & Infrastructure', 'Data Science & ML', 'Knowledge Graphs & Neo4j',
    'Career & Jobs', 'Finance & Crypto', 'Design & UI/UX', 'News & Articles',
    'Entertainment & Other'. Empty query returns top by score.
    """
    try:
        q_embed = _get_embedder().embed_query(query or category)
        rows = neo4j_store.vector_search(q_embed, n=n, category=category)
        return _result([_row_to_hit(r) for r in rows], "category", f"{category} :: {query}")
    except Exception as e:
        return _err("category", str(e))


@mcp.tool
def search_by_community(query: str, n: int = 20) -> dict:
    """
    Find the Louvain topic community closest to the query, return all its members.
    Use for broad-topic discovery ('everything I saved about agents'). Returns hits
    in one tight cluster, different from semantic search which returns the global top-n.
    """
    try:
        q_embed = _get_embedder().embed_query(query)
        rows = neo4j_store.search_by_community(q_embed, n=n)
        note = "" if rows else "No Louvain match (community not assigned yet?). Try search_semantic."
        return _result([_row_to_hit(r) for r in rows], "community", query, note=note)
    except Exception as e:
        return _err("community", str(e))


@mcp.tool
def find_by_tag(tag: str, n: int = 20) -> dict:
    """
    Find bookmarks tagged with exactly the given tag (case-insensitive).
    Use when you know the tag (e.g. 'rag', 'agents', 'fine-tuning', 'mcp').
    """
    try:
        rows = neo4j_store.find_by_tag(tag, limit=n)
        hits = [BookmarkHit(url=r["url"], title=r["title"], bm_score=float(r.get("score", 0) or 0))
                for r in rows]
        note = f"No matches. Try search_semantic('{tag}')." if not hits else ""
        return _result(hits, "tag", tag, note=note)
    except Exception as e:
        return _err("tag", str(e))


@mcp.tool
def explore_tag_cluster(tag: str, n: int = 25) -> dict:
    """
    Walk 2 hops out from the tag via CO_OCCURS_WITH edges, return bookmarks of
    neighboring tags. Use to expand from one tag into a wider thematic neighborhood.
    """
    try:
        rows = neo4j_store.find_tag_cluster(tag, hops=2, limit=n)
        hits = [BookmarkHit(
            url=r["url"], title=r["title"],
            bm_score=float(r.get("score", 0) or 0),
            tags=[r.get("via_tag", "")] if r.get("via_tag") else [],
        ) for r in rows]
        return _result(hits, "tag", f"cluster around '{tag}'")
    except Exception as e:
        return _err("tag", str(e))


@mcp.tool
def graph_expand(url: str) -> dict:
    """
    Given a specific bookmark URL, return its tags, similar bookmarks (SIMILAR_TO),
    and community peers. Use AFTER finding one interesting bookmark to discover
    related saved content along multiple graph dimensions.
    """
    try:
        text = neo4j_store.graph_expand(url)
        return {"strategy": "graph_expand", "url": url, "expansion": text}
    except Exception as e:
        return _err("graph_expand", str(e))


@mcp.tool
def find_by_domain(domain: str, n: int = 20) -> dict:
    """
    Filter bookmarks by domain (e.g. 'github.com', 'arxiv.org', 'youtube.com',
    'linkedin.com'). Pass naked domain WITHOUT 'www.' or scheme.
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
        return _result([_row_to_hit(r) for r in rows], "domain", domain)
    except Exception as e:
        return _err("domain", str(e))


@mcp.tool
def find_by_source(source: str, query: str = "", n: int = 20) -> dict:
    """
    Filter by source. Valid sources: 'linkedin', 'youtube_liked_videos',
    'youtube_watch_later', 'youtube_playlists', 'raindrop', 'edge', 'dailydev', 'manual'.
    Optional query does semantic ranking within the source.
    """
    try:
        if query.strip():
            q_embed = _get_embedder().embed_query(query)
            rows = neo4j_store.vector_search(q_embed, n=n, source=source)
        else:
            rows = neo4j_store.query("""
                MATCH (b:Bookmark) WHERE b.source = $source
                RETURN b.url AS url, b.title AS title, b.score AS bm_score,
                       b.source AS source, b.category AS category
                ORDER BY b.score DESC LIMIT $n
            """, {"source": source, "n": n})
        return _result([_row_to_hit(r) for r in rows], "source", f"{source} :: {query}")
    except Exception as e:
        return _err("source", str(e))


@mcp.tool
def search_linkedin(query: str, n: int = 15) -> dict:
    """Semantic search across saved LinkedIn posts only."""
    return find_by_source.fn(source="linkedin", query=query, n=n)


@mcp.tool
def search_youtube(query: str, n: int = 15) -> dict:
    """
    Semantic search across YouTube saves (liked + watch_later + playlists).
    Uses pre-filtered vector index for source='youtube*'.
    """
    try:
        q_embed = _get_embedder().embed_query(query)
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
        return _result([_row_to_hit(r) for r in rows], "source", f"youtube :: {query}")
    except Exception:
        return find_by_source.fn(source="youtube_liked_videos", query=query, n=n)


# ── Time-aware (uses LinkedIn created_at, requires graph_hygiene backfill) ──

@mcp.tool
def find_recent(days: int = 7, query: str = "", n: int = 20) -> dict:
    """
    Bookmarks added in the last N days. Currently only LinkedIn nodes have reliable
    timestamps (decoded from activity URN); others return empty unless backfilled.
    Optional query semantically ranks within the time window.
    """
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        if query.strip():
            q_embed = _get_embedder().embed_query(query)
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
        note = "Only LinkedIn nodes have timestamps. Edge / Raindrop / YouTube need backfill." \
               if not rows else ""
        return _result([_row_to_hit(r) for r in rows], "time", f"last {days}d :: {query}", note=note)
    except Exception as e:
        return _err("time", str(e))


@mcp.tool
def search_by_date_range(from_iso: str, to_iso: str, query: str = "", n: int = 25) -> dict:
    """
    Bookmarks created between two ISO timestamps (e.g. '2026-05-01' to '2026-05-14').
    Optional query semantically ranks within the window.
    """
    try:
        if query.strip():
            q_embed = _get_embedder().embed_query(query)
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
        return _result([_row_to_hit(r) for r in rows], "time",
                       f"{from_iso}..{to_iso} :: {query}")
    except Exception as e:
        return _err("time", str(e))


# ── Detail / context-fetching tools ───────────────────────────────────────────

@mcp.tool
def get_bookmark_full(url: str) -> dict:
    """
    Full record for one bookmark: url, title, category, tags, source, score,
    SIMILAR_TO neighbors (titles + urls), community peers (titles), created_at.
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
                   collect(DISTINCT t.name) AS tags,
                   collect(DISTINCT {url: s.url, title: s.title})[..6] AS similar,
                   collect(DISTINCT peer.title)[..6] AS community_peers
        """, {"url": url})
        if not rows:
            return {"strategy": "detail", "url": url, "note": "Bookmark not found."}
        return {"strategy": "detail", **rows[0]}
    except Exception as e:
        return _err("detail", str(e))


# ── Stats + advanced ──────────────────────────────────────────────────────────

@mcp.tool
def get_knowledge_base_stats() -> dict:
    """Total bookmark/tag/category/community counts in the knowledge base."""
    try:
        s = neo4j_store.get_stats()
        return {"strategy": "stats", **s}
    except Exception as e:
        return _err("stats", str(e))


_WRITE_CYPHER = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|CALL\s+db\.|CALL\s+apoc|CALL\s+gds\.[a-z]+\.write)\b",
    re.IGNORECASE,
)


@mcp.tool
def run_cypher(cypher: str) -> dict:
    """
    Run a READ-ONLY Cypher query against the Neo4j knowledge graph.
    Writes (CREATE/MERGE/SET/DELETE/REMOVE/DROP) are blocked at the gate.
    Use for advanced graph questions when higher-level tools aren't enough.
    Returns first 50 rows.
    """
    if _WRITE_CYPHER.search(cypher):
        return {"strategy": "cypher", "blocked": True,
                "note": "Write operations not allowed via run_cypher."}
    try:
        rows = neo4j_store.query(cypher)
        return {"strategy": "cypher", "rows": rows[:50], "total_returned": len(rows)}
    except Exception as e:
        return _err("cypher", str(e))


if __name__ == "__main__":
    sys.stderr.write("OpenMark MCP server running (stdio)\n")
    mcp.run()
