# Architecture

## Overview

OpenMark uses a **dual-store architecture** — two databases working together, each doing what it's best at.

```
                        User Query
                            │
                    LangGraph Agent
                    (gpt-4o-mini)
                   /              \
          ChromaDB               Neo4j
        (vector store)        (graph store)

        "find by meaning"     "find by connection"
        "what's similar?"     "how are things linked?"
```

---

## Embedding Layer

The embedding layer is **provider-agnostic** — swap between local and cloud with one env var.

```
EMBEDDING_PROVIDER=local   →  LocalEmbedder  (pplx-embed, runs on your machine)
EMBEDDING_PROVIDER=azure   →  AzureEmbedder  (Azure AI Foundry, API call)
```

**Why two pplx-embed models?**

Perplexity AI ships two variants:
- `pplx-embed-v1-0.6b` — for encoding **queries** (what the user types)
- `pplx-embed-context-v1-0.6b` — for encoding **documents** (the bookmarks, surrounding context matters)

Using the correct model for each role improves retrieval quality. Most implementations use one model for both — this is the correct production pattern.

**The compatibility patches:**

pplx-embed models ship with custom Python code (`st_quantize.py`) that has two incompatibilities with modern libraries:

1. **`sentence_transformers 4.x` removed the `Module` base class** — pplx-embed's code imports it. Fixed by aliasing `torch.nn.Module` to `sentence_transformers.models.Module` before import.

2. **`transformers 4.57` added `list_repo_templates()`** — it looks for an `additional_chat_templates` folder in every model repo. pplx-embed doesn't have one, causing a hard 404 crash. Fixed by monkey-patching the function to return an empty list on exception.

Both patches are applied in `openmark/embeddings/local.py` before any model loading.

**Why `sentence-transformers==3.3.1` specifically?**

Version 4.x removed the `Module` base class that pplx-embed depends on. Pin to 3.3.1.

---

## ChromaDB

Local, file-based vector database. No server, no API key, no cloud.

**Collection:** `openmark_bookmarks`
**Similarity metric:** cosine
**Data path:** `CHROMA_PATH` in `.env` (default: `OpenMark/data/chroma_db/`)

**What's stored per item:**
```python
{
    "id":       url,           # primary key
    "document": doc_text,      # rich text used for embedding
    "metadata": {
        "title":    str,
        "category": str,
        "source":   str,       # raindrop, linkedin, youtube_liked, edge, etc.
        "score":    float,     # quality score 1-10
        "tags":     str,       # comma-separated
        "folder":   str,
    },
    "embedding": [float x 1024]  # or 1536 for Azure
}
```

**Querying:**
```python
collection.query(
    query_embeddings=[embedder.embed_query("RAG tools")],
    n_results=10,
    where={"category": {"$eq": "RAG & Vector Search"}},  # optional filter
)
```

---

## Neo4j Graph Schema

```
(:Bookmark {url, title, score})
    -[:IN_CATEGORY]->   (:Category {name})
    -[:TAGGED]->        (:Tag {name})
    -[:FROM_SOURCE]->   (:Source {name})
    -[:FROM_DOMAIN]->   (:Domain {name})
    -[:SIMILAR_TO {score}]->  (:Bookmark)  ← from embeddings

(:Tag)-[:CO_OCCURS_WITH {count}]-(:Tag)    ← tags that appear together
```

**Useful Cypher queries:**

```cypher
// Count everything
MATCH (b:Bookmark) RETURN count(b) AS bookmarks
MATCH (t:Tag) RETURN count(t) AS tags

// Top categories
MATCH (b:Bookmark)-[:IN_CATEGORY]->(c:Category)
RETURN c.name, count(b) AS count ORDER BY count DESC

// All bookmarks tagged 'rag'
MATCH (b:Bookmark)-[:TAGGED]->(t:Tag {name: 'rag'})
RETURN b.title, b.url ORDER BY b.score DESC

// Find what connects to 'langchain' tag (2 hops)
MATCH (t:Tag {name: 'langchain'})-[:CO_OCCURS_WITH*1..2]-(related:Tag)
RETURN related.name, count(*) AS strength ORDER BY strength DESC

// Similar bookmarks to a URL
MATCH (b:Bookmark {url: 'https://...'})-[r:SIMILAR_TO]-(other)
RETURN other.title, other.url, r.score ORDER BY r.score DESC

// Most connected domains
MATCH (b:Bookmark)-[:FROM_DOMAIN]->(d:Domain)
RETURN d.name, count(b) AS saved ORDER BY saved DESC LIMIT 20
```

---

## LangGraph Agent

Built with `create_react_agent` from LangGraph 1.0.x.

**Model:** Azure gpt-4o-mini (streaming enabled)
**Memory:** `MemorySaver` — conversation history persists per `thread_id` within a session

**Tools:**

| Tool | Store | Description |
|------|-------|-------------|
| `search_semantic` | ChromaDB | Natural language vector search |
| `search_by_category` | ChromaDB | Filter by category + optional query |
| `find_by_tag` | Neo4j | Exact tag lookup |
| `find_similar_bookmarks` | Neo4j | SIMILAR_TO edge traversal |
| `explore_tag_cluster` | Neo4j | CO_OCCURS_WITH traversal (2 hops) |
| `get_stats` | Both | Count totals |
| `run_cypher` | Neo4j | Raw Cypher for power users |

**Agent routing:** The LLM decides which tool(s) to call based on the query. For "what do I know about RAG" it will call `search_semantic` + `search_by_category` + `find_by_tag`. For "how does LangGraph connect to my Neo4j saves" it will call `explore_tag_cluster` and `run_cypher`.

---

## Gradio UI

Three tabs:

| Tab | What it does |
|-----|-------------|
| Chat | Full LangGraph agent conversation. Remembers context within session. |
| Search | Direct ChromaDB search with category filter, min score slider, result count. |
| Stats | Neo4j category breakdown + top tags. Loads on startup. |

Run: `python openmark/ui/app.py` → `http://localhost:7860`

---

## Data Flow Summary

```
Source files (JSON, HTML)
        │
   merge.py → normalize.py
        │
   8,007 items with doc_text
        │
   EmbeddingProvider.embed_documents()
        │
   ┌────┴────┐
   │         │
ChromaDB   Neo4j
add()      MERGE nodes + relationships
           CO_OCCURS_WITH edges
           SIMILAR_TO edges (from ChromaDB top-5 per item)
```
