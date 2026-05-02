"""
OpenMark MCP server — Graph RAG edition. Neo4j only.
Exposes Ahmad's bookmark knowledge graph as tools for Claude Code.

Run:      python -m openmark.mcp.server
Register: .mcp.json in project root
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.stderr.write("OpenMark MCP loading...\n")

from fastmcp import FastMCP
from openmark.embeddings.factory import get_embedder
from openmark.stores import neo4j_store

mcp = FastMCP("openmark")

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def _fmt(results: list[dict], query: str = "") -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        tags = ", ".join(r.get("tags") or []) or "—"
        sim  = r.get("similarity", 0)
        lines.append(
            f"{i}. [{r.get('category', '')}] {r.get('title', '')}\n"
            f"   URL: {r.get('url', '')}\n"
            f"   Similarity: {sim:.3f} | Source: {r.get('source', '')} | Tags: {tags}"
        )
        similar = r.get("similar_urls") or []
        if similar:
            lines.append(f"   Related: {' | '.join(similar[:2])}")
    prefix = f"Top {len(results)} results" + (f" for '{query}'" if query else "")
    return prefix + ":\n\n" + "\n\n".join(lines)


@mcp.tool
def search_bookmarks(query: str, n: int = 10) -> str:
    """
    Search Ahmad's personal knowledge base of 8,831 bookmarks by semantic meaning.
    Uses Neo4j vector index + graph context (tags, related bookmarks).
    Sources: Edge browser, Raindrop, LinkedIn posts, daily.dev, YouTube.
    Returns top N results with title, URL, category, similarity, tags, and related bookmarks.
    """
    try:
        q_embed = _get_embedder().embed_query(query)
        results = neo4j_store.vector_search(q_embed, n=n)
        return _fmt(results, query)
    except Exception as e:
        return f"Search error: {e}"


@mcp.tool
def search_by_category(category: str, query: str = "", n: int = 15) -> str:
    """
    Find bookmarks in a specific category, optionally filtered by a search query.
    Available categories: Agent Development, RAG & Vector Search, LangChain / LangGraph,
    MCP & Tool Use, Context Engineering, LLM Fine-tuning, AI Tools & Platforms,
    GitHub Repos & OSS, Learning & Courses, YouTube & Video, Web Development,
    Cloud & Infrastructure, Data Science & ML, Knowledge Graphs & Neo4j,
    Career & Jobs, Finance & Crypto, Design & UI/UX, News & Articles, Entertainment & Other
    """
    try:
        q_embed = _get_embedder().embed_query(query or category)
        results = neo4j_store.vector_search(q_embed, n=n, category=category)
        return _fmt(results, query or category)
    except Exception as e:
        return f"Search error: {e}"


@mcp.tool
def find_bookmarks_by_source(source: str, query: str, n: int = 10) -> str:
    """
    Search bookmarks from a specific source only.
    Sources: 'edge', 'raindrop', 'linkedin', 'dailydev', 'youtube_liked_videos'
    """
    try:
        q_embed = _get_embedder().embed_query(query)
        results = neo4j_store.vector_search(q_embed, n=n, source=source)
        return _fmt(results, query)
    except Exception as e:
        return f"Search error: {e}"


@mcp.tool
def graph_expand(url: str) -> str:
    """
    Expand a bookmark in the knowledge graph.
    Returns: tags, similar bookmarks (SIMILAR_TO edges), community members (Louvain cluster).
    Use when you found a relevant bookmark and want to discover related saved content.
    """
    try:
        return neo4j_store.graph_expand(url)
    except Exception as e:
        return f"Graph error: {e}"


@mcp.tool
def search_by_community(query: str, n: int = 20) -> str:
    """
    Find all bookmarks in the topic community closest to the query.
    Uses Louvain community detection — returns a coherent topic cluster.
    """
    try:
        q_embed = _get_embedder().embed_query(query)
        results = neo4j_store.search_by_community(q_embed, n=n)
        if not results:
            return "No community match. Try search_bookmarks instead."
        lines = [f"- [{r.get('category', '')}] {r.get('title', '')}\n  {r.get('url', '')}"
                 for r in results]
        return f"Topic community for '{query}' ({len(results)} items):\n\n" + "\n\n".join(lines)
    except Exception as e:
        return f"Community search error: {e}"


@mcp.tool
def get_knowledge_base_stats() -> str:
    """Get statistics about Ahmad's bookmark knowledge base."""
    try:
        s = neo4j_store.get_stats()
        return (
            f"OpenMark Knowledge Base (Neo4j Graph RAG)\n"
            f"  Bookmarks:   {s.get('bookmarks', 0):,}\n"
            f"  Tags:        {s.get('tags', 0):,}\n"
            f"  Categories:  {s.get('categories', 0)}\n"
            f"  Communities: {s.get('communities', 0)}\n"
            f"  Embeddings:  pplx-embed-context-v1-0.6B (1024-dim)"
        )
    except Exception as e:
        return f"Stats error: {e}"


if __name__ == "__main__":
    sys.stderr.write("OpenMark MCP server running (stdio)\n")
    mcp.run()
