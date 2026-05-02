"""
LangGraph tools for the OpenMark Graph RAG agent.
All search runs through Neo4j vector index + graph traversal.
No ChromaDB — Neo4j only.
"""

from langchain_core.tools import tool
from openmark.embeddings.factory import get_embedder
from openmark.stores import neo4j_store

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def _fmt(results: list[dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        tags = ", ".join(r.get("tags") or []) or "—"
        sim  = r.get("similarity", 0)
        lines.append(
            f"{i}. [{r.get('category', '')}] {r.get('title', '')}\n"
            f"   {r.get('url', '')}\n"
            f"   similarity: {sim:.3f}  source: {r.get('source', '')}  tags: {tags}"
        )
        similar = r.get("similar_urls") or []
        if similar:
            lines.append(f"   related: {' | '.join(similar[:2])}")
    return "\n\n".join(lines)


@tool
def search_semantic(query: str, n: int = 10) -> str:
    """
    Search bookmarks by semantic meaning using Neo4j vector index + graph context.
    Returns titles, URLs, categories, tags, and related bookmarks.
    Use this FIRST for any query.
    """
    try:
        q_embed = _get_embedder().embed_query(query)
        results = neo4j_store.vector_search(q_embed, n=n)
        return _fmt(results)
    except Exception as e:
        return f"Search error: {e}"


@tool
def search_by_category(category: str, query: str = "", n: int = 15) -> str:
    """
    Find bookmarks in a specific category, optionally narrowed by a query.
    Categories: RAG & Vector Search, Agent Development, LangChain / LangGraph,
    MCP & Tool Use, Context Engineering, LLM Fine-tuning, AI Tools & Platforms,
    GitHub Repos & OSS, Learning & Courses, YouTube & Video, Web Development,
    Cloud & Infrastructure, Data Science & ML, Knowledge Graphs & Neo4j,
    Career & Jobs, Finance & Crypto, Design & UI/UX, News & Articles, Entertainment & Other
    """
    try:
        q_embed = _get_embedder().embed_query(query or category)
        results = neo4j_store.vector_search(q_embed, n=n, category=category)
        return f"Category '{category}':\n\n" + _fmt(results)
    except Exception as e:
        return f"Search error: {e}"


@tool
def graph_expand(url: str) -> str:
    """
    Expand a bookmark in the knowledge graph.
    Returns its tags, similar bookmarks (SIMILAR_TO edges), and community members.
    Use after finding an interesting bookmark to discover related content.
    """
    try:
        return neo4j_store.graph_expand(url)
    except Exception as e:
        return f"Graph error: {e}"


@tool
def search_by_community(query: str, n: int = 20) -> str:
    """
    Find all bookmarks in the topic community closest to the query.
    Uses Louvain graph clustering — surfaces a coherent topic cluster.
    Great for discovering everything saved about a broad topic.
    """
    try:
        q_embed = _get_embedder().embed_query(query)
        results = neo4j_store.search_by_community(q_embed, n=n)
        if not results:
            return "No community found. Try search_semantic instead."
        lines = [f"- [{r.get('category', '')}] {r.get('title', '')}\n  {r.get('url', '')}"
                 for r in results]
        return f"Topic community for '{query}' ({len(results)} items):\n\n" + "\n\n".join(lines)
    except Exception as e:
        return f"Community search error: {e}"


@tool
def find_by_tag(tag: str) -> str:
    """Find bookmarks tagged with a specific tag using the knowledge graph."""
    try:
        results = neo4j_store.find_by_tag(tag, limit=20)
        if not results:
            return f"No bookmarks tagged '{tag}'. Try search_semantic('{tag}')."
        lines = [f"- {r['title']}\n  {r['url']}" for r in results]
        return f"Tag '{tag}' — {len(results)} results:\n" + "\n".join(lines)
    except Exception as e:
        return f"Tag search error: {e}"


@tool
def explore_tag_cluster(tag: str) -> str:
    """Explore connected tags 2 hops away via CO_OCCURS_WITH edges."""
    try:
        results = neo4j_store.find_tag_cluster(tag, hops=2, limit=25)
        if not results:
            return f"No cluster for tag '{tag}'."
        lines = [f"- [{r['via_tag']}] {r['title']}\n  {r['url']}" for r in results]
        return f"Knowledge cluster around '{tag}':\n" + "\n".join(lines)
    except Exception as e:
        return f"Cluster search error: {e}"


@tool
def get_stats() -> str:
    """Get statistics about the OpenMark knowledge base."""
    try:
        s = neo4j_store.get_stats()
        return (
            f"OpenMark Knowledge Base:\n"
            f"  Bookmarks:   {s.get('bookmarks', 0):,}\n"
            f"  Tags:        {s.get('tags', 0):,}\n"
            f"  Categories:  {s.get('categories', 0)}\n"
            f"  Communities: {s.get('communities', 0)}"
        )
    except Exception as e:
        return f"Stats error: {e}"


@tool
def run_cypher(cypher: str) -> str:
    """Run a raw Cypher query against the Neo4j knowledge graph."""
    try:
        rows = neo4j_store.query(cypher)
        if not rows:
            return "Query returned no results."
        return "\n".join(str(r) for r in rows[:20])
    except Exception as e:
        return f"Cypher error: {e}"


ALL_TOOLS = [
    search_semantic,
    search_by_category,
    graph_expand,
    search_by_community,
    find_by_tag,
    explore_tag_cluster,
    get_stats,
    run_cypher,
]
