"""
Graph hygiene: bring the Neo4j graph back into a coherent state
after a fresh injection batch.

Steps (in order):
  1. Strip LinkedIn tracking query params from existing URLs (idempotent fix
     for the `updateEntityUrn` duplication bug) + dedupe by stripped URL.
  2. Backfill `activity_urn` on LinkedIn nodes for stable joins.
  3. Backfill `created_at` ISO timestamps from LinkedIn `urn:li:activity:N`
     IDs (the activity ID encodes the post timestamp).
  4. Rebuild SIMILAR_TO edges over the full graph (uses the Neo4j vector index).
  5. Re-run Louvain to assign community_id to every node.

Run with: python scripts/graph_hygiene.py
"""

import sys, os, re, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.stores import neo4j_store


ACTIVITY_RE = re.compile(r"urn:li:activity:(\d+)", re.IGNORECASE)
LINKEDIN_TRACKING_RE = re.compile(r"[?&]updateentityurn=[^&#]*", re.IGNORECASE)


def linkedin_timestamp_from_activity_id(activity_id: int) -> datetime | None:
    """
    LinkedIn activity IDs encode the post timestamp.
    The top 41 bits of the ID are the Unix-ms timestamp.
    """
    try:
        ms = activity_id >> 22
        if ms < 1_000_000_000_000 or ms > 4_000_000_000_000:  # sanity bound 2001..2096
            return None
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    except Exception:
        return None


def step_dedupe_linkedin_urls():
    """
    Strip the tracking query param from LinkedIn URLs and dedupe.
    Keeps the lowest-score node when duplicates exist (older entries first).
    """
    print("Step 1: dedupe LinkedIn URLs by stripped tracking params...")

    rows = neo4j_store.query("""
        MATCH (b:Bookmark)
        WHERE b.source = 'linkedin' AND toLower(b.url) CONTAINS 'updateentityurn'
        RETURN b.url AS url
    """)
    print(f"  LinkedIn URLs with tracking params: {len(rows)}")

    # Build (clean_url, old_url) pairs
    updates = []
    for r in rows:
        old = r["url"]
        clean = LINKEDIN_TRACKING_RE.sub("", old).rstrip("?").rstrip("&")
        if clean != old:
            updates.append({"old": old, "clean": clean})

    print(f"  URLs that will be rewritten: {len(updates)}")

    # Apply in batches: if the clean URL already exists as a different node,
    # we DETACH DELETE the duplicate; otherwise SET the url to the clean form.
    moved = 0
    deleted = 0
    batch = 500
    for i in range(0, len(updates), batch):
        chunk = updates[i:i+batch]
        for u in chunk:
            existing = neo4j_store.query(
                "MATCH (b:Bookmark {url: $clean}) RETURN b.url AS url LIMIT 1",
                {"clean": u["clean"]},
            )
            if existing and existing[0]["url"] != u["old"]:
                # Clean form already exists — delete the duplicate dirty node
                neo4j_store.query(
                    "MATCH (b:Bookmark {url: $old}) DETACH DELETE b",
                    {"old": u["old"]},
                )
                deleted += 1
            else:
                # Rewrite the dirty URL to its clean form
                neo4j_store.query(
                    "MATCH (b:Bookmark {url: $old}) SET b.url = $clean",
                    {"old": u["old"], "clean": u["clean"]},
                )
                moved += 1
        print(f"  Progress: {min(i+batch, len(updates))}/{len(updates)} (moved={moved}, deleted={deleted})")

    print(f"  Done. Renamed: {moved}, Deleted duplicates: {deleted}")


def step_backfill_activity_urn():
    """Set b.activity_urn on every LinkedIn node from its URL."""
    print("Step 2: backfill activity_urn on LinkedIn nodes...")
    rows = neo4j_store.query("""
        MATCH (b:Bookmark) WHERE b.source = 'linkedin' AND b.activity_urn IS NULL
        RETURN b.url AS url LIMIT 100000
    """)
    print(f"  Nodes needing activity_urn: {len(rows)}")
    n = 0
    for r in rows:
        m = ACTIVITY_RE.search(r["url"] or "")
        if not m:
            continue
        activity = int(m.group(1))
        neo4j_store.query(
            "MATCH (b:Bookmark {url: $url}) SET b.activity_urn = $a",
            {"url": r["url"], "a": activity},
        )
        n += 1
    print(f"  activity_urn set on {n} nodes")


def step_backfill_linkedin_created_at():
    """Decode timestamp from LinkedIn activity_urn and write as created_at."""
    print("Step 3: backfill created_at on LinkedIn nodes from activity_urn...")
    rows = neo4j_store.query("""
        MATCH (b:Bookmark) WHERE b.source = 'linkedin'
            AND b.activity_urn IS NOT NULL
            AND b.created_at IS NULL
        RETURN b.url AS url, b.activity_urn AS a LIMIT 100000
    """)
    print(f"  Nodes needing created_at: {len(rows)}")
    n = 0
    for r in rows:
        ts = linkedin_timestamp_from_activity_id(r["a"])
        if not ts:
            continue
        neo4j_store.query(
            "MATCH (b:Bookmark {url: $url}) SET b.created_at = datetime($ts)",
            {"url": r["url"], "ts": ts.isoformat()},
        )
        n += 1
    print(f"  created_at set on {n} nodes")


def step_build_similar_to():
    """Build SIMILAR_TO edges using vector index over full graph."""
    print("Step 4: build SIMILAR_TO edges over full graph (this takes a few min)...")
    t0 = time.time()
    neo4j_store.build_similar_to_edges()
    print(f"  Done in {time.time()-t0:.1f}s")


def step_louvain():
    """Re-run Louvain over the refreshed SIMILAR_TO graph."""
    print("Step 5: run Louvain community detection...")
    t0 = time.time()
    neo4j_store.setup_louvain()
    print(f"  Done in {time.time()-t0:.1f}s")


def main():
    start = time.time()
    step_dedupe_linkedin_urls()
    step_backfill_activity_urn()
    step_backfill_linkedin_created_at()
    step_build_similar_to()
    step_louvain()

    s = neo4j_store.get_stats()
    print(f"\nFinal stats: {s.get('bookmarks',0):,} bookmarks, "
          f"{s.get('tags',0):,} tags, "
          f"{s.get('categories',0)} categories, "
          f"{s.get('communities',0)} communities")
    print(f"Total hygiene time: {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
