"""
Backfill `created_at` on existing Bookmark nodes from every Edge HTML export
in Downloads/, plus any json files in data/ that carry a `created` field.

Idempotent: only writes when the node's current created_at is NULL. If the
same URL appears in multiple exports with different ADD_DATEs (Edge bumps
ADD_DATE on re-bookmark), the EARLIEST timestamp wins — that's the "saved at"
the user actually means.

Run with: python scripts/backfill_created_at.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.pipeline.injector import parse_html_file
from openmark.pipeline.normalize import parse_created_at
from openmark.stores import neo4j_store


DOWNLOADS = r"C:\Users\oasrvadmin\Downloads"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def collect_from_edge_exports() -> dict[str, str]:
    """Walk every favorites_*.html in Downloads, build url -> earliest ISO."""
    out: dict[str, str] = {}
    files = sorted(glob.glob(os.path.join(DOWNLOADS, "favorites_*.html"))
                   + glob.glob(os.path.join(DOWNLOADS, "bookmarks_*.html")))
    print(f"Edge exports found: {len(files)}")
    for path in files:
        try:
            items = parse_html_file(path)
        except Exception as e:
            print(f"  {os.path.basename(path)}: parse error {e}")
            continue
        kept = 0
        for item in items:
            url = (item.get("url") or "").strip().rstrip("/").lower()
            add_date = item.get("add_date")
            if not url or not add_date:
                continue
            iso = parse_created_at(add_date)
            if not iso:
                continue
            prev = out.get(url)
            if prev is None or iso < prev:
                out[url] = iso
                kept += 1
        print(f"  {os.path.basename(path)}: {len(items)} items, {kept} updated url→date map")
    print(f"Total unique URLs with ADD_DATE: {len(out)}")
    return out


def collect_from_data_json() -> dict[str, str]:
    """Read data/*.json files looking for a `created` field per item."""
    out: dict[str, str] = {}
    json_files = (
        glob.glob(os.path.join(DATA_DIR, "*.json"))
        + glob.glob(os.path.join(DATA_DIR, "html_export_*.json"))
    )
    json_files = sorted(set(json_files))
    print(f"Data JSON files: {len(json_files)}")
    for path in json_files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  {os.path.basename(path)}: read error {e}")
            continue
        items = data if isinstance(data, list) else data.get("items") if isinstance(data, dict) else None
        if not items:
            continue
        kept = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            url = (it.get("url") or it.get("link") or "").strip().rstrip("/").lower()
            raw = it.get("created_at") or it.get("created") or it.get("add_date") or it.get("publishedAt") or it.get("published_at")
            iso = parse_created_at(raw)
            if not url or not iso:
                continue
            prev = out.get(url)
            if prev is None or iso < prev:
                out[url] = iso
                kept += 1
        print(f"  {os.path.basename(path)}: {kept} entries collected")
    print(f"Total unique URLs from JSON: {len(out)}")
    return out


def apply_backfill(url_to_iso: dict[str, str]) -> None:
    """Apply created_at to Neo4j Bookmark nodes where currently NULL.

    Match strategy is forgiving: tries exact url, lower(url), rstrip('/') variants.
    """
    if not url_to_iso:
        print("No URLs to backfill.")
        return

    print(f"Applying backfill to {len(url_to_iso)} URL→date entries...")
    t0 = time.time()
    batch_size = 500
    items = list(url_to_iso.items())
    total_set = 0
    total_skipped_existing = 0
    total_no_match = 0

    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        params = [{"url": u, "alt": u + "/", "iso": iso} for u, iso in chunk]
        result = neo4j_store.query(
            """
            UNWIND $rows AS row
            OPTIONAL MATCH (b:Bookmark)
            WHERE b.url = row.url OR b.url = row.alt OR toLower(b.url) = row.url
            WITH b, row LIMIT $cap
            WHERE b IS NOT NULL
            WITH b, row,
                 CASE WHEN b.created_at IS NULL THEN 1 ELSE 0 END AS will_set
            FOREACH (_ IN CASE WHEN will_set = 1 THEN [1] ELSE [] END |
                SET b.created_at = datetime(row.iso)
            )
            RETURN sum(will_set) AS set_count, count(b) AS matched
            """,
            {"rows": params, "cap": batch_size * 4},
        )
        if result:
            sc = result[0].get("set_count") or 0
            mc = result[0].get("matched") or 0
            total_set += sc
            total_skipped_existing += (mc - sc)
        if (i + batch_size) % 2000 == 0 or (i + batch_size) >= len(items):
            print(f"  {min(i + batch_size, len(items))}/{len(items)} processed "
                  f"(set={total_set}, skipped_existing={total_skipped_existing})")

    # Count unmatched URLs separately (URLs in map that have no Bookmark node)
    matched_urls_rows = neo4j_store.query(
        """
        UNWIND $urls AS u
        OPTIONAL MATCH (b:Bookmark) WHERE b.url = u OR b.url = u + '/'
        RETURN u AS url, b IS NOT NULL AS has_node
        """,
        {"urls": [u for u, _ in items]},
    )
    no_match = sum(1 for r in matched_urls_rows if not r.get("has_node"))
    print(f"\nBackfill summary:")
    print(f"  URLs in map:        {len(items):>6}")
    print(f"  Newly set:          {total_set:>6}")
    print(f"  Already had a date: {total_skipped_existing:>6}")
    print(f"  No matching node:   {no_match:>6}")
    print(f"  Duration:           {time.time() - t0:.1f}s")


def main() -> None:
    print("=" * 60)
    print("created_at backfill — Edge HTML + data JSON")
    print("=" * 60)

    edge_map = collect_from_edge_exports()
    json_map = collect_from_data_json()

    # Merge with earliest-wins (Edge is generally authoritative)
    merged: dict[str, str] = dict(edge_map)
    for u, iso in json_map.items():
        if u not in merged or iso < merged[u]:
            merged[u] = iso
    print(f"\nMerged URL→date entries: {len(merged)}\n")

    apply_backfill(merged)

    # Final state
    stats = neo4j_store.query(
        """
        MATCH (b:Bookmark)
        RETURN count(b) AS total,
               sum(CASE WHEN b.created_at IS NOT NULL THEN 1 ELSE 0 END) AS with_date
        """
    )
    if stats:
        t = stats[0].get("total") or 0
        w = stats[0].get("with_date") or 0
        pct = (w / t * 100) if t else 0
        print(f"\nFinal: {w:,} / {t:,} nodes have created_at ({pct:.1f}%)")


if __name__ == "__main__":
    main()
