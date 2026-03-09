"""
ChromaDB store — semantic vector search.
"""

import chromadb
from openmark import config
from openmark.embeddings.base import EmbeddingProvider

COLLECTION_NAME = "openmark_bookmarks"


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=config.CHROMA_PATH)


def get_collection(client: chromadb.PersistentClient, embedder: EmbeddingProvider):
    """Get or create the bookmarks collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest(items: list[dict], embedder: EmbeddingProvider, batch_size: int = 100):
    """Embed all items and store in ChromaDB."""
    client     = get_client()
    collection = get_collection(client, embedder)

    # Check already ingested
    existing = set(collection.get(include=[])["ids"])
    new_items = [i for i in items if i["url"] not in existing]
    print(f"ChromaDB: {len(existing)} already ingested, {len(new_items)} new")

    if not new_items:
        return

    total = 0
    for start in range(0, len(new_items), batch_size):
        batch = new_items[start:start + batch_size]

        texts = [i["doc_text"] for i in batch]
        ids   = [i["url"] for i in batch]
        metas = [
            {
                "title":    i["title"][:500],
                "category": i["category"],
                "source":   i["source"],
                "score":    float(i["score"]),
                "tags":     ",".join(i["tags"]),
                "folder":   i.get("folder", ""),
            }
            for i in batch
        ]

        embeddings = embedder.embed_documents(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )
        total += len(batch)
        print(f"  ChromaDB ingested {total}/{len(new_items)}")

    print(f"ChromaDB total: {collection.count()} items")


def search(
    query: str,
    embedder: EmbeddingProvider,
    n: int = 10,
    category: str | None = None,
    source: str | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """Semantic search with optional metadata filters."""
    client     = get_client()
    collection = get_collection(client, embedder)

    q_embedding = embedder.embed_query(query)

    # Build filters
    filters = []
    if category:
        filters.append({"category": {"$eq": category}})
    if source:
        filters.append({"source": {"$eq": source}})
    if min_score is not None:
        filters.append({"score": {"$gte": min_score}})

    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    output = []
    for i, (meta, doc, dist) in enumerate(zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    )):
        output.append({
            "rank":       i + 1,
            "url":        results["ids"][0][i],
            "title":      meta.get("title", ""),
            "category":   meta.get("category", ""),
            "source":     meta.get("source", ""),
            "score":      meta.get("score", 0),
            "tags":       meta.get("tags", "").split(","),
            "similarity": round(1 - dist, 4),
        })
    return output


def get_stats() -> dict:
    client     = get_client()
    collection = get_collection(client, None)
    return {"total": collection.count()}
