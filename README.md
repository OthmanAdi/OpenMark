# OpenMark

**Your personal knowledge graph — built from everything you've ever saved.**

OpenMark ingests your bookmarks, LinkedIn saved posts, and YouTube videos into a dual-store knowledge system: **ChromaDB** for semantic vector search and **Neo4j** for graph-based connection discovery. A LangGraph agent sits on top, letting you query everything in natural language.

Built by [Ahmad Othman Ammar Adi](https://github.com/OthmanAdi).

---

## What it does

- Pulls all your saved content from multiple sources into one place
- Embeds everything using [pplx-embed](https://huggingface.co/collections/perplexity-ai/pplx-embed) (local, free) or Azure AI Foundry (fast, cheap)
- Stores vectors in **ChromaDB** — find things by *meaning*, not keywords
- Builds a **Neo4j knowledge graph** — discover how topics connect
- Runs a **LangGraph agent** (powered by gpt-4o-mini) that searches both stores intelligently
- Serves a **Gradio UI** with Chat, Search, and Stats tabs
- Also works as a **CLI** — `python scripts/search.py "RAG tools"`

---

## Data Sources

### 1. Raindrop.io

Create a test token at [app.raindrop.io/settings/integrations](https://app.raindrop.io/settings/integrations).
OpenMark pulls **all collections** automatically via the Raindrop REST API.

### 2. Browser Bookmarks

Export your bookmarks as an HTML file from Edge, Chrome, or Firefox:
- **Edge:** `Settings → Favourites → ··· → Export favourites` → save as `favorites.html`
- **Chrome/Firefox:** `Bookmarks Manager → Export`

Point `RAINDROP_MISSION_DIR` in your `.env` to the folder containing the exported HTML files.
The pipeline parses the Netscape bookmark format automatically.

### 3. LinkedIn Saved Posts

LinkedIn does not provide a public API for saved posts. The included `linkedin_fetch.py` script uses your browser session cookie to call LinkedIn's internal Voyager GraphQL API.

**Steps:**
1. Log into LinkedIn in your browser
2. Open DevTools → Application → Cookies → copy the value of `li_at`
3. Run:
   ```bash
   python raindrop-mission/linkedin_fetch.py
   ```
   Paste your `li_at` cookie when prompted. The script fetches all saved posts and writes `linkedin_saved.json`.

> **Personal use only.** This uses LinkedIn's internal API which is not publicly documented or officially supported. Use responsibly.

### 4. YouTube

Uses the official [YouTube Data API v3](https://developers.google.com/youtube/v3) via OAuth 2.0.

**Steps:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/) → Create a project
2. Enable the **YouTube Data API v3**
3. Create OAuth 2.0 credentials → Download as `client_secret.json`
4. Add your Google account as a test user (OAuth consent screen → Test users)
5. Run:
   ```bash
   python raindrop-mission/youtube_fetch.py
   ```
   A browser window opens for auth. After that, `youtube_MASTER.json` is written with liked videos, watch later, and playlists.

---

## How it works

```
Your saved content
        │
        ▼
  normalize.py          ← clean titles, dedupe by URL, fix categories
        │
        ▼
  EmbeddingProvider     ← LOCAL: pplx-embed-context-v1-0.6b (documents)
                                  pplx-embed-v1-0.6b (queries)
                           AZURE: text-embedding-ada-002
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
    ChromaDB                            Neo4j
  (vector store)                    (knowledge graph)
  find by meaning                   find by connection

  "show me RAG tools"               "what connects LangGraph
                                     to my Neo4j saves?"
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              LangGraph Agent
              (gpt-4o-mini)
                       │
                       ▼
                  Gradio UI  /  CLI
```

### Why embeddings?

An embedding is a list of numbers that represents the *meaning* of a piece of text. Two pieces of text with similar meaning will have similar numbers — even if they use completely different words. This is how OpenMark finds "retrieval augmented generation tutorials" when you search "RAG tools."

### Why ChromaDB?

ChromaDB stores those embedding vectors locally on your disk. It's a persistent vector database — no server, no cloud, no API key. When you search, it compares your query's embedding against all stored embeddings and returns the closest matches.

### Why Neo4j?

Embeddings answer "what's similar?" — Neo4j answers "how are these connected?" Every bookmark is a node. Tags, categories, domains, and sources are also nodes. Edges connect them. After ingestion, OpenMark also writes `SIMILAR_TO` edges derived from embedding neighbors — so the graph contains semantic connections you never manually created. You can then traverse: *"start from this LangChain article, walk similar-to 2 hops, what clusters emerge?"*

---

## Requirements

- Python 3.13
- Neo4j Desktop (local) or AuraDB (cloud) — [neo4j.com/download](https://neo4j.com/download/)
- **Either** Azure AI Foundry account **or** enough disk space for local pplx-embed (~1.2 GB)

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/OthmanAdi/OpenMark.git
cd OpenMark
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
# Choose your embedding provider
EMBEDDING_PROVIDER=local        # or: azure

# Azure AI Foundry (required if EMBEDDING_PROVIDER=azure, also used for the LLM agent)
AZURE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_API_KEY=your-key
AZURE_DEPLOYMENT_LLM=gpt-4o-mini
AZURE_DEPLOYMENT_EMBED=text-embedding-ada-002

# Neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# Raindrop (get token at app.raindrop.io/settings/integrations)
RAINDROP_TOKEN=your-token

# Path to your raindrop-mission data folder
RAINDROP_MISSION_DIR=C:\path\to\raindrop-mission
```

### 3. Ingest

```bash
# Local embeddings (free, ~20 min for 8K items on CPU)
python scripts/ingest.py

# Azure embeddings (fast, ~5 min, costs ~€0.30 for 8K items)
python scripts/ingest.py --provider azure

# Also pull fresh from Raindrop API during ingest
python scripts/ingest.py --fresh-raindrop

# Skip SIMILAR_TO edge computation (saves 25-40 min, Neo4j still required)
python scripts/ingest.py --skip-similar

# ChromaDB only — skip Neo4j entirely (Neo4j not required)
python scripts/ingest.py --skip-neo4j
```

### 4. Search (CLI)

```bash
python scripts/search.py "RAG tools"
python scripts/search.py "LangGraph" --category "Agent Development"
python scripts/search.py --tag "rag"
python scripts/search.py --stats
```

### 5. Launch UI

```bash
python openmark/ui/app.py
```

Open [http://localhost:7860](http://localhost:7860)

---

## Required API Keys

| Key | Where to get it | Required? |
|-----|----------------|-----------|
| `RAINDROP_TOKEN` | [app.raindrop.io/settings/integrations](https://app.raindrop.io/settings/integrations) | Yes |
| `AZURE_API_KEY` | Azure Portal → your AI Foundry resource | Only if `EMBEDDING_PROVIDER=azure` |
| `NEO4J_PASSWORD` | Set when creating your Neo4j database | Yes |
| YouTube OAuth | Google Cloud Console → YouTube Data API v3 | Only if ingesting YouTube |

No HuggingFace token is needed for local pplx-embed. The models are open weights and download automatically. You will see a warning `"You are sending unauthenticated requests to the HF Hub"` — this is harmless and can be silenced by setting `HF_TOKEN` in your `.env` if you want higher rate limits.

---

## Project Structure

```
OpenMark/
├── openmark/
│   ├── config.py              ← all settings loaded from .env
│   ├── pipeline/
│   │   ├── raindrop.py        ← pull all Raindrop collections via API
│   │   ├── normalize.py       ← clean, dedupe, build embedding text
│   │   └── merge.py           ← combine all sources
│   ├── embeddings/
│   │   ├── base.py            ← abstract EmbeddingProvider interface
│   │   ├── local.py           ← pplx-embed (local, free)
│   │   ├── azure.py           ← Azure AI Foundry
│   │   └── factory.py         ← returns provider based on .env
│   ├── stores/
│   │   ├── chroma.py          ← ChromaDB: ingest + semantic search
│   │   └── neo4j_store.py     ← Neo4j: graph nodes, edges, traversal
│   ├── agent/
│   │   ├── tools.py           ← LangGraph tools (search, tag, graph)
│   │   └── graph.py           ← create_react_agent with gpt-4o-mini
│   └── ui/
│       └── app.py             ← Gradio UI (Chat / Search / Stats)
└── scripts/
    ├── ingest.py              ← full pipeline runner
    └── search.py              ← CLI search
```

---

## Roadmap

- [ ] OpenAI embeddings integration
- [ ] Ollama local LLM support
- [ ] Pinecone vector store option
- [ ] Web scraping — fetch full page content for richer embeddings
- [ ] Browser extension for real-time saving to OpenMark
- [ ] Comet / Arc browser bookmark import
- [ ] Automatic re-ingestion on schedule
- [ ] Export to Obsidian / Notion
- [ ] Multi-user support

---

## Documentation

| Doc | What's in it |
|-----|-------------|
| [docs/data-collection.md](docs/data-collection.md) | Full guide for each data source — Raindrop, Edge, LinkedIn cookie method, YouTube OAuth, daily.dev console script |
| [docs/ingest.md](docs/ingest.md) | All ingest flags, timing for each step, how SIMILAR_TO edges work, re-run behavior |
| [docs/architecture.md](docs/architecture.md) | Dual-store design, Neo4j graph schema, embedding patches, Cypher query examples, agent tools |
| [docs/troubleshooting.md](docs/troubleshooting.md) | pplx-embed compatibility fixes, LinkedIn queryId changes, Neo4j connection issues, Windows encoding |

---

## License

MIT
