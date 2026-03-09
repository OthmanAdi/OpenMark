# Ingest Pipeline

The ingest pipeline is the heart of OpenMark. It merges all your data, embeds everything, and writes to both ChromaDB and Neo4j.

---

## Command

```bash
python scripts/ingest.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--provider local` | from `.env` | Use local pplx-embed models |
| `--provider azure` | from `.env` | Use Azure AI Foundry embeddings |
| `--fresh-raindrop` | off | Also pull live from Raindrop API during merge |
| `--skip-similar` | off | Skip SIMILAR_TO edge computation (saves ~30 min) |

---

## Pipeline Steps

### Step 1 — Merge

Loads and deduplicates all sources:
- `CATEGORIZED.json` — pre-categorized bookmarks from Edge + Raindrop + daily.dev
- `linkedin_saved.json` — LinkedIn saved posts
- `youtube_MASTER.json` — liked videos, watch later, playlists (not subscriptions)

Deduplication is URL-based (case-insensitive, trailing slash stripped). If the same URL appears in multiple sources, the first occurrence wins.

Each item gets a `doc_text` field built for embedding:
```
{title} | {category} | {tag1 tag2 tag3} | {content/excerpt/channel}
```
This rich text is what gets embedded — not just the title.

**Output:** ~8,000 normalized items in memory.

---

### Step 2 — Embedding

Loads the embedding provider specified by `EMBEDDING_PROVIDER` in `.env` (or `--provider` flag).

**Local (pplx-embed):**
- Query model: `perplexity-ai/pplx-embed-v1-0.6b` — used for user search queries
- Document model: `perplexity-ai/pplx-embed-context-v1-0.6b` — used for bookmark documents
- Output dimension: 1024
- Downloaded once to HuggingFace cache (~1.2 GB total), free on every subsequent run
- **Known compatibility issue:** pplx-embed requires `sentence-transformers==3.3.1` and two runtime patches (applied automatically in `local.py`). See [troubleshooting.md](troubleshooting.md) for details.

**Azure:**
- Uses `text-embedding-ada-002` (or configured `AZURE_DEPLOYMENT_EMBED`)
- Output dimension: 1536
- Cost: ~€0.30 for 8,000 items (as of 2026)
- Batched in groups of 100 with progress logging

---

### Step 3 — ChromaDB Ingest

Embeds all documents in batches of 100 and stores in ChromaDB.

- Skips items already in ChromaDB (resumable — safe to re-run)
- Stores: URL (as ID), embedding vector, title, category, source, score, tags
- Uses cosine similarity space (`hnsw:space: cosine`)
- Database written to disk at `CHROMA_PATH` (default: `OpenMark/data/chroma_db/`)

**Timing:**
| Provider | 8K items | Notes |
|----------|----------|-------|
| Local pplx-embed (CPU) | ~20 min | No GPU detected = CPU inference |
| Local pplx-embed (GPU) | ~3 min | NVIDIA GPU with CUDA |
| Azure AI Foundry | ~5 min | Network bound |

---

### Step 4 — Neo4j Ingest

Creates nodes and relationships in batches of 200.

**Nodes created:**
- `Bookmark` — url, title, score
- `Category` — name
- `Tag` — name
- `Source` — name (raindrop, linkedin, youtube_liked, edge, dailydev, etc.)
- `Domain` — extracted from URL (e.g. `github.com`, `medium.com`)

**Relationships created:**
- `(Bookmark)-[:IN_CATEGORY]->(Category)`
- `(Bookmark)-[:TAGGED]->(Tag)`
- `(Bookmark)-[:FROM_SOURCE]->(Source)`
- `(Bookmark)-[:FROM_DOMAIN]->(Domain)`
- `(Tag)-[:CO_OCCURS_WITH {count}]-(Tag)` — built after all nodes are written

**Timing:** ~3-5 minutes for 8K items.

**Idempotent:** Uses `MERGE` everywhere — safe to re-run, won't create duplicates.

---

### Step 5 — SIMILAR_TO Edges

This is the most powerful and most time-consuming step.

For each of the 8K bookmarks, OpenMark queries ChromaDB for its top-5 nearest semantic neighbors and writes those as `SIMILAR_TO` edges in Neo4j with a similarity score.

```
(Bookmark {url: "...langchain-docs..."})-[:SIMILAR_TO {score: 0.94}]->(Bookmark {url: "...langgraph-tutorial..."})
```

These edges encode **semantic connections you never manually created**. The knowledge graph becomes a web of meaning, not just a web of tags.

**Timing:** ~25-40 minutes on CPU for 8K items. This is the longest step.

**Skip it if you're in a hurry:**
```bash
python scripts/ingest.py --skip-similar
```
Everything else works without SIMILAR_TO edges. You only lose the `find_similar_bookmarks` tool in the agent and the graph traversal from those edges.

**Only edges with similarity > 0.5 are written.** Low-quality connections are discarded.

---

## Re-running the Pipeline

The pipeline is safe to re-run at any time:

- **ChromaDB:** skips already-ingested URLs automatically
- **Neo4j:** uses `MERGE` — no duplicates created
- **SIMILAR_TO:** edges are overwritten (not duplicated) via `MERGE`

To add new bookmarks after the first run:
1. Update your source files (fresh Raindrop pull, new LinkedIn export, etc.)
2. Run `python scripts/ingest.py` — only new items get embedded and stored

---

## Checking What's Ingested

```bash
# Quick stats
python scripts/search.py --stats

# Search to verify
python scripts/search.py "RAG tools"

# Neo4j — open browser
# http://localhost:7474
# Run: MATCH (b:Bookmark) RETURN count(b)
```
