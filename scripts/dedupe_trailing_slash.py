"""
Merge trailing-slash duplicate Bookmark nodes into their canonical (no-slash)
counterparts. Pre-existing data baggage: nodes ingested before
`normalize.canonicalize_url()` landed kept their trailing slash.

For each (dirty, clean) pair where `dirty.url = clean.url + '/'`:
  1. Copy any of dirty's TAGGED edges onto clean (via MERGE — idempotent).
  2. Copy non-null scalars from dirty into clean ONLY if clean's are NULL.
  3. DETACH DELETE the dirty node.

SIMILAR_TO edges from the dirty node are dropped — rerun `graph_hygiene.py`
afterward (it rebuilds SIMILAR_TO over the full graph in ~45s).

Run with: python scripts/dedupe_trailing_slash.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.stores import neo4j_store


def main() -> None:
    print("Scanning for trailing-slash dupe pairs...")
    pairs = neo4j_store.query(
        """
        MATCH (dirty:Bookmark) WHERE dirty.url ENDS WITH '/' AND size(dirty.url) > 8
        WITH dirty, substring(dirty.url, 0, size(dirty.url) - 1) AS clean_url
        MATCH (clean:Bookmark) WHERE clean.url = clean_url AND clean <> dirty
        RETURN count(*) AS n
        """
    )
    total = pairs[0]["n"] if pairs else 0
    print(f"Pairs to merge: {total}\n")
    if total == 0:
        print("Nothing to do.")
        return

    t0 = time.time()
    # Single-shot Cypher: per pair, MERGE tags from dirty onto clean,
    # coalesce missing scalars, then DETACH DELETE dirty.
    result = neo4j_store.query(
        """
        MATCH (dirty:Bookmark) WHERE dirty.url ENDS WITH '/' AND size(dirty.url) > 8
        WITH dirty, substring(dirty.url, 0, size(dirty.url) - 1) AS clean_url
        MATCH (clean:Bookmark) WHERE clean.url = clean_url AND clean <> dirty
        WITH dirty, clean
        OPTIONAL MATCH (dirty)-[:TAGGED]->(t:Tag)
        WITH dirty, clean, collect(DISTINCT t) AS dirty_tags
        FOREACH (tg IN dirty_tags |
            MERGE (clean)-[:TAGGED]->(tg)
        )
        SET clean.created_at = coalesce(clean.created_at, dirty.created_at),
            clean.doc_text   = coalesce(clean.doc_text,   dirty.doc_text),
            clean.activity_urn = coalesce(clean.activity_urn, dirty.activity_urn)
        WITH dirty, clean, dirty_tags
        DETACH DELETE dirty
        RETURN count(clean) AS merged
        """
    )
    merged = (result[0].get("merged") if result else 0) or 0
    print(f"Merged + deleted dirty nodes: {merged} in {time.time() - t0:.1f}s")

    # Final state
    stats = neo4j_store.get_stats()
    print(
        f"\nFinal: {stats.get('bookmarks',0):,} bookmarks · "
        f"{stats.get('tags',0):,} tags · "
        f"{stats.get('categories',0)} categories"
    )

    # Quick recheck that no trailing-slash dupes remain
    remaining = neo4j_store.query(
        """
        MATCH (dirty:Bookmark) WHERE dirty.url ENDS WITH '/' AND size(dirty.url) > 8
        WITH dirty, substring(dirty.url, 0, size(dirty.url) - 1) AS clean_url
        MATCH (clean:Bookmark) WHERE clean.url = clean_url AND clean <> dirty
        RETURN count(*) AS n
        """
    )
    print(f"Remaining trailing-slash dupes: {remaining[0]['n'] if remaining else 0}")
    print(
        "\nReminder: run `python scripts/graph_hygiene.py` to rebuild SIMILAR_TO "
        "now that dirty nodes are gone."
    )


if __name__ == "__main__":
    main()
