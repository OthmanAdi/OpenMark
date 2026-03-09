"""
OpenMark Full Ingest Pipeline
Run this once (or again to update) to:
  1. Merge all data sources (CATEGORIZED.json + LinkedIn + YouTube)
  2. Embed everything with chosen provider (local pplx-embed or Azure)
  3. Store in ChromaDB (semantic search)
  4. Store in Neo4j (knowledge graph)
  5. Compute SIMILAR_TO edges (top-5 neighbors per bookmark → graph edges)

Usage:
  C:\\Python313\\python scripts/ingest.py
  C:\\Python313\\python scripts/ingest.py --provider azure
  C:\\Python313\\python scripts/ingest.py --fresh-raindrop   (also pulls live from Raindrop API)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.pipeline.merge import merge_all
from openmark.embeddings.factory import get_embedder
from openmark.stores import chroma as chroma_store
from openmark.stores import neo4j_store
from openmark import config


def build_similar_to_edges(items: list[dict], embedder, top_k: int = 5):
    """
    For each item, find its top-k nearest neighbors in ChromaDB
    and write SIMILAR_TO edges in Neo4j.
    This creates the semantic web inside the graph.
    """
    print(f"\nBuilding SIMILAR_TO edges (top-{top_k} per bookmark)...")
    pairs = []
    total = len(items)

    for i, item in enumerate(items):
        url = item["url"]
        try:
            results = chroma_store.search(
                item["doc_text"], embedder, n=top_k + 1
            )
            for r in results:
                if r["url"] != url and r["similarity"] > 0.5:
                    pairs.append((url, r["url"], r["similarity"]))
        except Exception:
            pass

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{total} for SIMILAR_TO")

    print(f"  Writing {len(pairs)} SIMILAR_TO edges to Neo4j...")
    neo4j_store.add_similar_to_edges(pairs)
    print("  SIMILAR_TO done.")


def main():
    parser = argparse.ArgumentParser(description="OpenMark Ingest Pipeline")
    parser.add_argument("--provider",        default=None, help="Embedding provider: local or azure")
    parser.add_argument("--fresh-raindrop",  action="store_true", help="Also pull fresh from Raindrop API")
    parser.add_argument("--skip-similar",    action="store_true", help="Skip SIMILAR_TO edge computation")
    args = parser.parse_args()

    if args.provider:
        os.environ["EMBEDDING_PROVIDER"] = args.provider

    print("=" * 60)
    print("OPENMARK INGEST PIPELINE")
    print(f"Embedding: {config.EMBEDDING_PROVIDER}")
    print("=" * 60)

    # Step 1: Merge all sources
    print("\n[1/4] Merging data sources...")
    items = merge_all(include_fresh_raindrop=args.fresh_raindrop)

    # Step 2: Load embedder
    print(f"\n[2/4] Loading {config.EMBEDDING_PROVIDER} embedder...")
    embedder = get_embedder()

    # Step 3: ChromaDB
    print("\n[3/4] Ingesting into ChromaDB...")
    chroma_store.ingest(items, embedder)

    # Step 4: Neo4j
    print("\n[4/4] Ingesting into Neo4j...")
    neo4j_store.ingest(items)

    # Step 5: SIMILAR_TO edges
    if not args.skip_similar:
        build_similar_to_edges(items, embedder, top_k=5)

    print("\n" + "=" * 60)
    print("INGEST COMPLETE")
    chroma = chroma_store.get_stats()
    neo4j  = neo4j_store.get_stats()
    print(f"  ChromaDB: {chroma.get('total', 0)} vectors")
    print(f"  Neo4j:    {neo4j.get('bookmarks', 0)} bookmarks, {neo4j.get('tags', 0)} tags")
    print("=" * 60)
    print("\nNow run: C:\\Python313\\python scripts/search.py \"your query\"")
    print("     or: C:\\Python313\\python -m openmark.ui.app")


if __name__ == "__main__":
    main()
