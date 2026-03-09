"""
OpenMark CLI Search — instant search from terminal.

Usage:
  C:\\Python313\\python scripts/search.py "RAG tools"
  C:\\Python313\\python scripts/search.py "LangGraph" --category "Agent Development"
  C:\\Python313\\python scripts/search.py "embeddings" --n 20
  C:\\Python313\\python scripts/search.py --tag "rag"
  C:\\Python313\\python scripts/search.py --stats
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.embeddings.factory import get_embedder
from openmark.stores import chroma as chroma_store
from openmark.stores import neo4j_store


def print_results(results: list[dict]):
    if not results:
        print("No results found.")
        return
    for r in results:
        title = r.get("title") or r.get("url")
        url   = r.get("url", "")
        cat   = r.get("category", "")
        sim   = r.get("similarity", "")
        score = r.get("score", "")
        tags  = ", ".join(t for t in r.get("tags", []) if t)
        print(f"\n  {r.get('rank', '-')}. {title}")
        print(f"     {url}")
        if cat:   print(f"     Category:   {cat}")
        if tags:  print(f"     Tags:       {tags}")
        if score: print(f"     Score:      {score}")
        if sim:   print(f"     Similarity: {sim}")


def main():
    parser = argparse.ArgumentParser(description="OpenMark CLI Search")
    parser.add_argument("query",      nargs="?", default=None, help="Search query")
    parser.add_argument("--category", default=None, help="Filter by category")
    parser.add_argument("--tag",      default=None, help="Search by tag (graph lookup)")
    parser.add_argument("--n",        type=int, default=10, help="Number of results")
    parser.add_argument("--stats",    action="store_true", help="Show knowledge base stats")
    args = parser.parse_args()

    if args.stats:
        chroma = chroma_store.get_stats()
        neo4j  = neo4j_store.get_stats()
        print("\nOpenMark Stats:")
        print(f"  ChromaDB vectors: {chroma.get('total', 0)}")
        print(f"  Neo4j bookmarks:  {neo4j.get('bookmarks', 0)}")
        print(f"  Neo4j tags:       {neo4j.get('tags', 0)}")
        return

    if args.tag:
        print(f"\nSearching by tag: '{args.tag}'")
        results = neo4j_store.find_by_tag(args.tag, limit=args.n)
        for r in results:
            print(f"\n  - {r.get('title', '')}")
            print(f"    {r.get('url', '')} (score: {r.get('score', '')})")
        return

    if not args.query:
        parser.print_help()
        return

    print(f"\nSearching: '{args.query}'")
    if args.category:
        print(f"Category filter: {args.category}")

    embedder = get_embedder()
    results  = chroma_store.search(
        args.query, embedder, n=args.n, category=args.category
    )
    print_results(results)


if __name__ == "__main__":
    main()
