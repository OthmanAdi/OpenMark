"""
OpenMark FRESH Re-Embedding Pipeline
=====================================
- Upgrades to pplx-embed-v1-4b (6.7x larger than 0.6b)
- Enriches doc_text with more context for better search
- Filters out NSFW flagged items
- Wipes ChromaDB and rebuilds from scratch
- Pulls fresh from Raindrop API

Usage:
  C:\Python313\python scripts/fresh_reembed.py
  C:\Python313\python scripts/fresh_reembed.py --keep-06b     (use old 0.6b model instead)
  C:\Python313\python scripts/fresh_reembed.py --skip-raindrop (don't pull from Raindrop)
"""

import sys
import os
import json
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.pipeline.merge import merge_all
from openmark.pipeline.normalize import normalize_item, dedupe
from openmark import config


def load_nsfw_blocklist() -> set:
    """Load NSFW flagged URLs that should be excluded."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nsfw_flagged.json")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        flagged = json.load(f)
    # Block items where keep != true
    blocked = {item["url"] for item in flagged if not item.get("keep", False)}
    print(f"NSFW blocklist: {len(blocked)} URLs will be excluded")
    return blocked


def enrich_doc_text(item: dict) -> str:
    """
    Build a RICHER document text for embedding.
    More context = better semantic search results.
    """
    parts = []

    # Title is the strongest signal
    if item.get("title"):
        parts.append(item["title"])

    # Category provides topical grounding
    if item.get("category"):
        parts.append(f"Category: {item['category']}")

    # Tags are high-signal keywords
    if item.get("tags"):
        tag_str = ", ".join(item["tags"]) if isinstance(item["tags"], list) else item["tags"]
        parts.append(f"Tags: {tag_str}")

    # Folder path gives hierarchical context
    if item.get("folder"):
        parts.append(f"Folder: {item['folder']}")

    # Content/excerpt is the richest signal when available
    if item.get("content"):
        parts.append(item["content"][:500])  # Increased from 200 to 500
    elif item.get("excerpt"):
        parts.append(item["excerpt"][:500])
    elif item.get("description"):
        parts.append(item["description"][:500])

    # Author/channel adds attribution context
    if item.get("channel"):
        parts.append(f"Channel: {item['channel']}")
    if item.get("author"):
        parts.append(f"Author: {item['author']}")

    # Source helps distinguish origin
    if item.get("source"):
        parts.append(f"Source: {item['source']}")

    return " | ".join(p for p in parts if p)


def main():
    parser = argparse.ArgumentParser(description="OpenMark Fresh Re-Embedding")
    parser.add_argument("--keep-06b", action="store_true", help="Use 0.6B model instead of 4B")
    parser.add_argument("--skip-raindrop", action="store_true", help="Don't pull fresh from Raindrop")
    args = parser.parse_args()

    # Select model
    if args.keep_06b:
        query_model = "perplexity-ai/pplx-embed-v1-0.6b"
        doc_model = "perplexity-ai/pplx-embed-context-v1-0.6b"
        print("Using pplx-embed 0.6B (original)")
    else:
        query_model = "perplexity-ai/pplx-embed-v1-4b"
        doc_model = "perplexity-ai/pplx-embed-context-v1-4b"
        print("Using pplx-embed 4B (UPGRADED — better semantic understanding)")

    # Override config
    os.environ["PPLX_QUERY_MODEL"] = query_model
    os.environ["PPLX_DOC_MODEL"] = doc_model

    print("=" * 60)
    print("OPENMARK FRESH RE-EMBEDDING PIPELINE")
    print(f"Query model:  {query_model}")
    print(f"Doc model:    {doc_model}")
    print("=" * 60)

    # Step 1: Merge all sources
    print("\n[1/5] Merging all data sources...")
    items = merge_all(include_fresh_raindrop=not args.skip_raindrop)

    # Step 2: Load NSFW blocklist and filter
    print("\n[2/5] Filtering NSFW content...")
    blocked = load_nsfw_blocklist()
    before = len(items)
    items = [i for i in items if i["url"] not in blocked]
    print(f"  Removed {before - len(items)} NSFW items")
    print(f"  Clean items: {len(items)}")

    # Step 3: Enrich doc_text for ALL items
    print("\n[3/5] Enriching document text for better embeddings...")
    for item in items:
        item["doc_text"] = enrich_doc_text(item)

    # Step 4: Wipe and rebuild ChromaDB
    print("\n[4/5] Wiping ChromaDB for fresh rebuild...")
    chroma_path = config.CHROMA_PATH
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        os.makedirs(chroma_path)
        print(f"  Wiped: {chroma_path}")
    else:
        os.makedirs(chroma_path, exist_ok=True)

    # Step 5: Load embedder and ingest
    print(f"\n[5/5] Loading {query_model.split('/')[-1]} embedder and ingesting...")

    # Reload config to pick up new model names
    import importlib
    importlib.reload(config)

    from openmark.embeddings.factory import get_embedder
    from openmark.stores import chroma as chroma_store

    embedder = get_embedder()
    chroma_store.ingest(items, embedder)

    # Stats
    stats = chroma_store.get_stats()
    print("\n" + "=" * 60)
    print("FRESH RE-EMBEDDING COMPLETE")
    print(f"  Model:     {doc_model}")
    print(f"  Vectors:   {stats.get('total', 0)}")
    print(f"  NSFW:      {before - len(items)} blocked")
    print(f"  Doc text:  enriched (title + category + tags + folder + content + author)")
    print("=" * 60)
    print("\nNow run: C:\\Python313\\python openmark/ui/app.py")
    print("Search quality should be significantly better.")


if __name__ == "__main__":
    main()
