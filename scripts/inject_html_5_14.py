"""
One-shot: inject the ALREADY normalized + diffed new bookmarks
from data/html_export_5_14_26_new.json into Neo4j.

Loads embedder once, embeds all docs, MERGE-writes to Neo4j.
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.embeddings.factory import get_embedder
from openmark.stores import neo4j_store
from openmark.pipeline.normalize import build_document_text

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "html_export_5_14_26_new.json")

def main():
    with open(SRC, encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} pre-diffed new items")

    # Ensure each has doc_text (normalize_item should have populated it but check)
    for it in items:
        if not it.get("doc_text"):
            it["doc_text"] = build_document_text(it)

    print("Loading embedder (pplx-embed-context-v1-0.6b on CPU)...")
    t0 = time.time()
    embedder = get_embedder()
    print(f"Embedder ready in {time.time()-t0:.1f}s")

    texts = [it["doc_text"] for it in items]
    print(f"Embedding {len(texts)} docs...")
    t0 = time.time()
    embeddings = embedder.embed_documents(texts)
    print(f"Embedded in {time.time()-t0:.1f}s ({(time.time()-t0)/len(texts)*1000:.0f} ms/item)")

    print("Writing to Neo4j...")
    t0 = time.time()
    neo4j_store.ingest(items, embeddings=embeddings)
    print(f"Ingest done in {time.time()-t0:.1f}s")

    stats = neo4j_store.get_stats()
    print(f"\nNeo4j now has: {stats.get('bookmarks', 0):,} bookmarks, {stats.get('tags', 0):,} tags")

if __name__ == "__main__":
    main()
