"""
OpenMark Full Ingest Pipeline — Graph RAG (Neo4j only)

Steps:
  1. Merge all data sources
  2. Load pplx-embed-context-v1-0.6B embedder
  3. Embed all items locally
  4. Store in Neo4j (nodes + embeddings + relationships)
  5. Build SIMILAR_TO edges via Neo4j vector index
  6. Run Louvain community detection (requires GDS plugin)

Usage:
  python scripts/ingest.py
  python scripts/ingest.py --fresh-raindrop   # also pull live from Raindrop API
  python scripts/ingest.py --skip-similar     # skip SIMILAR_TO edge computation
  python scripts/ingest.py --skip-louvain     # skip community detection
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.pipeline.merge import merge_all
from openmark.embeddings.factory import get_embedder
from openmark.stores import neo4j_store
from openmark import config


def main():
    parser = argparse.ArgumentParser(description="OpenMark Ingest Pipeline")
    parser.add_argument("--fresh-raindrop", action="store_true", help="Pull fresh from Raindrop API")
    parser.add_argument("--skip-similar",   action="store_true", help="Skip SIMILAR_TO edge computation")
    parser.add_argument("--skip-louvain",   action="store_true", help="Skip Louvain community detection")
    args = parser.parse_args()

    print("=" * 60)
    print("OPENMARK INGEST — GRAPH RAG")
    print(f"Embedding:  {config.EMBEDDING_PROVIDER} ({config.pplx_dimension()}-dim)")
    print(f"Neo4j:      {config.NEO4J_URI} / db:{config.NEO4J_DATABASE}")
    print("=" * 60)

    print("\n[1/4] Merging data sources...")
    items = merge_all(include_fresh_raindrop=args.fresh_raindrop)

    print(f"\n[2/4] Loading {config.EMBEDDING_PROVIDER} embedder...")
    embedder = get_embedder()

    print(f"\n[3/4] Checking for new items...")
    try:
        existing_urls = set(
            r["url"] for r in neo4j_store.query("MATCH (b:Bookmark) RETURN b.url AS url")
        )
        new_items = [i for i in items if i["url"] not in existing_urls]
        print(f"  Neo4j has {len(existing_urls):,} items already. {len(new_items):,} new to embed.")
    except Exception:
        # Neo4j empty or first run — embed everything
        new_items = items
        print(f"  First run — embedding all {len(new_items):,} items.")

    if not new_items:
        print("  Nothing new. Skipping embed + ingest.")
    else:
        print(f"\n  Embedding {len(new_items)} items ({embedder.dimension}-dim)...")
        texts = [i["doc_text"] for i in new_items]
        embeddings = embedder.embed_documents(texts)
        print(f"  Done — {len(embeddings)} vectors")

        print("\n[4/4] Ingesting into Neo4j...")
        neo4j_store.ingest(new_items, embeddings=embeddings)

    if not args.skip_similar:
        print("\nBuilding SIMILAR_TO edges via vector index...")
        neo4j_store.build_similar_to_edges()

        if not args.skip_louvain:
            print("\nRunning Louvain community detection...")
            neo4j_store.setup_louvain()
    else:
        print("\nSIMILAR_TO + Louvain skipped.")

    print("\n" + "=" * 60)
    print("INGEST COMPLETE")
    try:
        stats = neo4j_store.get_stats()
        print(f"  Bookmarks:   {stats.get('bookmarks', 0):,}")
        print(f"  Tags:        {stats.get('tags', 0):,}")
        print(f"  Communities: {stats.get('communities', 0)}")
    except Exception as e:
        print(f"  Stats error: {e}")
    print("=" * 60)
    print("\nRun: python -m openmark.ui.app")


if __name__ == "__main__":
    main()
