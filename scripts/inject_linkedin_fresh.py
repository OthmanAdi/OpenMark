"""
Inject only NEW LinkedIn posts (not already in Neo4j) into the KB.
Uses the same load_linkedin() mapper as the full ingest pipeline,
plus the same normalize_item / dedupe steps, then diffs against Neo4j.
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.pipeline.merge import load_linkedin
from openmark.pipeline.normalize import normalize_item, dedupe, build_document_text
from openmark.embeddings.factory import get_embedder
from openmark.stores import neo4j_store


def main():
    raw = load_linkedin()
    print(f"Raw LinkedIn items: {len(raw)}")

    norm = dedupe([normalize_item(i) for i in raw])
    print(f"After normalize + local dedupe: {len(norm)}")

    # Filter out empty URLs (LinkedIn posts sometimes have no nav URL)
    norm = [i for i in norm if i.get("url")]
    print(f"With URLs: {len(norm)}")

    urls = [i["url"] for i in norm]
    existing = set()
    batch = 500
    for i in range(0, len(urls), batch):
        rows = neo4j_store.query(
            "MATCH (b:Bookmark) WHERE b.url IN $urls RETURN b.url AS url",
            {"urls": urls[i:i+batch]},
        )
        existing.update(r["url"] for r in rows)
    print(f"Already in Neo4j: {len(existing)}")

    new_items = [i for i in norm if i["url"] not in existing]
    print(f"NEW LinkedIn posts: {len(new_items)}")
    if not new_items:
        print("Nothing to inject.")
        return

    for it in new_items:
        if not it.get("doc_text"):
            it["doc_text"] = build_document_text(it)

    print("Loading embedder...")
    t0 = time.time()
    embedder = get_embedder()
    print(f"Embedder ready in {time.time()-t0:.1f}s")

    texts = [i["doc_text"] for i in new_items]
    print(f"Embedding {len(texts)} docs...")
    t0 = time.time()
    embeddings = embedder.embed_documents(texts)
    print(f"Embedded in {time.time()-t0:.1f}s")

    print("Writing to Neo4j...")
    t0 = time.time()
    neo4j_store.ingest(new_items, embeddings=embeddings)
    print(f"Ingest done in {time.time()-t0:.1f}s")

    s = neo4j_store.get_stats()
    print(f"\nKB now: {s.get('bookmarks', 0):,} bookmarks, {s.get('tags', 0):,} tags")


if __name__ == "__main__":
    main()
