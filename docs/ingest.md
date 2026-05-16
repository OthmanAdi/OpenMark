# Ingest Pipeline (v3 — Neo4j only)

The ingest pipeline is the heart of OpenMark. It merges every source, embeds each item, and writes a Neo4j Graph RAG store with vector index + tag co-occurrence + SIMILAR_TO neighbors + optional Louvain communities. ChromaDB was removed in v3.

---

## Command

```bash
python scripts/ingest.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--fresh-raindrop` | off | Also pull live from Raindrop API during merge |
| `--skip-similar`   | off | Skip SIMILAR_TO edge computation (saves ~25-40 min) |
| `--skip-louvain`   | off | Skip Louvain community detection (requires GDS plugin) |

Provider comes from `.env` (`EMBEDDING_PROVIDER`). No `--provider` flag.

---

## Pipeline steps

### Step 1 — Merge

Loads and deduplicates every source:
- `CATEGORIZED.json` — pre-categorized bookmarks from Edge + Raindrop + daily.dev
- `linkedin_saved.json` — LinkedIn saved posts
- `youtube_MASTER.json` — liked videos, watch-later, playlists (not subscriptions)

Deduplication is URL-based (case-insensitive, trailing slash stripped). First occurrence wins.

Each item gets a `doc_text` field built for embedding:
```
{title} | {category} | {tag1 tag2 tag3} | {content/excerpt/channel}
```

Rich text is what gets embedded, not just the title.

### Step 2 — Embedding

Loads the embedding provider specified by `EMBEDDING_PROVIDER` in `.env`.

**Local pplx-embed (recommended):**
- Query model: `pplx-embed-v1-0.6b` for user queries
- Document model: `pplx-embed-context-v1-0.6b` for bookmark documents
- Output dimension: 1024
- Downloaded once to HuggingFace cache (~1.2 GB total)
- Known compatibility patches applied in `openmark/embeddings/local.py`; see [troubleshooting.md](troubleshooting.md).

**Azure:**
- Uses `AZURE_DEPLOYMENT_EMBED` (default `text-embedding-3-large`, 1536 dim if you pick that; or 1024 dim with `text-embedding-3-small`)
- Cost: a few cents per 1K items (varies by deployment)
- Batched, with progress logging

### Step 3 — Neo4j ingest

Single-store write. Each batch of ~200 items writes nodes + edges via `MERGE` (idempotent).

**Nodes:**
- `Bookmark` — `url, title, score, source, category, created_at, embedding[1024]`
- `Category`, `Tag`, `Source`, `Domain`

**Relationships:**
- `(Bookmark)-[:IN_CATEGORY]->(Category)`
- `(Bookmark)-[:TAGGED]->(Tag)`
- `(Bookmark)-[:FROM_SOURCE]->(Source)`
- `(Bookmark)-[:FROM_DOMAIN]->(Domain)`
- `(Tag)-[:CO_OCCURS_WITH {count}]-(Tag)` — built after all bookmarks are in

Vector index `bookmark_embedding` on `b.embedding` powers semantic search.

### Step 4 — SIMILAR_TO edges (skippable)

For each Bookmark, queries the Neo4j vector index for its top-5 nearest neighbors and writes `(:Bookmark)-[:SIMILAR_TO {score}]->(:Bookmark)` edges.

| Setup | 13k items | Notes |
|---|---|---|
| Neo4j vector index | ~10-15 min | Native to Neo4j 5.13+ |
| --skip-similar | 0 min | Skip; recoverable later |

### Step 5 — Louvain communities (skippable)

If the Neo4j GDS plugin is installed, runs Louvain on a similarity sub-graph and writes `(:Bookmark)-[:IN_COMMUNITY]->(:Community)` edges. Skipped automatically if GDS is missing.

---

## Re-running

`MERGE` everywhere → safe to re-run. The pipeline:

- Skips already-embedded URLs based on the existing `b.embedding` property
- Recomputes SIMILAR_TO and CO_OCCURS_WITH from scratch (cheap)
- Refreshes Louvain communities

To force a full re-embed (e.g. after switching providers):
```bash
python scripts/fresh_reembed.py
```
