"""
LangGraph tools for the OpenMark agent.
Each tool hits either ChromaDB (semantic) or Neo4j (graph) or both.
"""

from langchain_core.tools import tool
from openmark.embeddings.factory import get_embedder
from openmark.stores import chroma as chroma_store
from openmark.stores import neo4j_store

# Embedder is loaded once and reused
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


@tool
def search_semantic(query: str, n: int = 10) -> str:
    """
    Search bookmarks by semantic meaning using vector similarity.
    Use this for natural language queries like 'RAG tools', 'LangGraph tutorials', etc.
    Returns top N most relevant bookmarks.
    """
    results = chroma_store.search(query, _get_embedder(), n=n)
    if not results:
        return "No results found."
    lines = [f"{r['rank']}. [{r['category']}] {r['title']}\n   {r['url']} (similarity: {r['similarity']}, score: {r['score']})"
             for r in results]
    return "\n".join(lines)


@tool
def search_by_category(category: str, query: str = "", n: int = 15) -> str:
    """
    Find bookmarks in a specific category, optionally filtered by semantic query.
    Categories: RAG & Vector Search, Agent Development, LangChain / LangGraph,
    MCP & Tool Use, Context Engineering, AI Tools & Platforms, GitHub Repos & OSS,
    Learning & Courses, YouTube & Video, Web Development, Cloud & Infrastructure,
    Data Science & ML, Knowledge Graphs & Neo4j, Career & Jobs, LLM Fine-tuning,
    Finance & Crypto, Design & UI/UX, News & Articles, Entertainment & Other
    """
    if query:
        results = chroma_store.search(query, _get_embedder(), n=n, category=category)
    else:
        results = chroma_store.search(category, _get_embedder(), n=n, category=category)
    if not results:
        return f"No bookmarks found in category '{category}'."
    lines = [f"{r['rank']}. {r['title']}\n   {r['url']}" for r in results]
    return f"Category '{category}' — top results:\n" + "\n".join(lines)


@tool
def find_by_tag(tag: str) -> str:
    """
    Find all bookmarks tagged with a specific tag using the knowledge graph.
    Returns bookmarks ordered by quality score.
    """
    results = neo4j_store.find_by_tag(tag, limit=20)
    if not results:
        return f"No bookmarks found with tag '{tag}'."
    lines = [f"- {r['title']}\n  {r['url']} (score: {r['score']})" for r in results]
    return f"Bookmarks tagged '{tag}':\n" + "\n".join(lines)


@tool
def find_similar_bookmarks(url: str) -> str:
    """
    Find bookmarks semantically similar to a given URL.
    Uses SIMILAR_TO edges in the knowledge graph (built from embedding neighbors).
    """
    results = neo4j_store.find_similar(url, limit=10)
    if not results:
        return f"No similar bookmarks found for {url}."
    lines = [f"- {r['title']}\n  {r['url']} (similarity: {r['similarity']:.3f})" for r in results]
    return "Similar bookmarks:\n" + "\n".join(lines)


@tool
def explore_tag_cluster(tag: str) -> str:
    """
    Explore the knowledge graph around a tag — find related tags and their bookmarks.
    Traverses CO_OCCURS_WITH edges (2 hops) to discover connected topics.
    Great for discovering what else you know about a topic.
    """
    results = neo4j_store.find_tag_cluster(tag, hops=2, limit=25)
    if not results:
        return f"No cluster found for tag '{tag}'."
    lines = [f"- [{r['via_tag']}] {r['title']}\n  {r['url']}" for r in results]
    return f"Knowledge cluster around '{tag}':\n" + "\n".join(lines)


@tool
def get_stats() -> str:
    """
    Get statistics about the OpenMark knowledge base.
    Shows total bookmarks, tags, categories in both ChromaDB and Neo4j.
    """
    chroma_stats = chroma_store.get_stats()
    neo4j_stats  = neo4j_store.get_stats()
    return (
        f"OpenMark Knowledge Base Stats:\n"
        f"  ChromaDB vectors:   {chroma_stats.get('total', 0)}\n"
        f"  Neo4j bookmarks:    {neo4j_stats.get('bookmarks', 0)}\n"
        f"  Neo4j tags:         {neo4j_stats.get('tags', 0)}\n"
        f"  Neo4j categories:   {neo4j_stats.get('categories', 0)}"
    )


@tool
def run_cypher(cypher: str) -> str:
    """
    Run a raw Cypher query against the Neo4j knowledge graph.
    Use for advanced graph traversals. Example:
    MATCH (b:Bookmark)-[:TAGGED]->(t:Tag) WHERE t.name='rag' RETURN b.title, b.url LIMIT 10
    """
    try:
        rows = neo4j_store.query(cypher)
        if not rows:
            return "Query returned no results."
        lines = [str(r) for r in rows[:20]]
        return "\n".join(lines)
    except Exception as e:
        return f"Cypher error: {e}"


ALL_TOOLS = [
    search_semantic,
    search_by_category,
    find_by_tag,
    find_similar_bookmarks,
    explore_tag_cluster,
    get_stats,
    run_cypher,
]
