"""
LangGraph tools for OpenMark — typed Pydantic returns, no markdown lies.

Every tool returns a ToolResult; the .to_compact_markdown() rendering goes
into the tool message. URLs live in named fields so the synthesizer can
validate citations against state["seen_urls"].
"""

import re
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
        q_embed = _get_embedder().embed_query(query)
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
        q_embed = _get_embedder().embed_query(query or category)
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
        q_embed = _get_embedder().embed_query(query)
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
            q_embed = _get_embedder().embed_query(query)
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
        hits = [_row_to_hit(r) for r in rows]
        return ToolResult(
            hits=hits, strategy="source",
            query_echo=f"youtube :: {query}", total_found=len(hits),
        ).to_compact_markdown()
    except Exception:
        # Fallback if pre-filter SEARCH syntax not supported on this Neo4j version
        return find_by_source.invoke({"source": "youtube_liked_videos", "query": query, "n": n})


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
    get_stats,
    run_cypher,
]
