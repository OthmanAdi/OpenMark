"""
Merge ALL data sources into one clean list:
  - CATEGORIZED.json  (Edge + old Raindrop + daily.dev — already categorized)
  - edge_bookmarks.json  (parsed from Edge HTML export)
  - linkedin_saved.json  (LinkedIn saved posts)
  - youtube_MASTER.json  (liked + watch_later + playlists)
  - Fresh Raindrop pull  (new items not yet in CATEGORIZED)

Deduplicates by URL. Normalizes categories.
"""

import json
import os
from openmark import config
from openmark.pipeline.normalize import normalize_item, dedupe

EDGE_BOOKMARKS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "edge_bookmarks.json"
)


def load_categorized() -> list[dict]:
    path = os.path.join(config.RAINDROP_MISSION_DIR, "CATEGORIZED.json")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    print(f"CATEGORIZED.json: {len(items)} items")
    return items


def load_edge_bookmarks() -> list[dict]:
    path = os.path.normpath(EDGE_BOOKMARKS_PATH)
    if not os.path.exists(path):
        print("Edge bookmarks: file not found, skipping")
        return []
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    print(f"Edge bookmarks: {len(items)} items")
    return items


def load_linkedin() -> list[dict]:
    path = os.path.join(config.RAINDROP_MISSION_DIR, "linkedin_saved.json")
    if not os.path.exists(path):
        print("LinkedIn: file not found, skipping")
        return []
    with open(path, encoding="utf-8") as f:
        posts = json.load(f)
    items = []
    for p in posts:
        content = p.get("content", "")
        author  = p.get("author", "")
        items.append({
            "url":      p.get("url", ""),
            "title":    f"{author} — {content[:80]}" if author else content[:100],
            "content":  content[:300],
            "author":   author,
            "folder":   "LinkedIn Saved",
            "source":   "linkedin",
            "tags":     [],
            "category": None,  # will be assigned by normalize
            "score":    6,
        })
    print(f"LinkedIn: {len(items)} posts")
    return items


def load_youtube() -> list[dict]:
    path = os.path.join(config.RAINDROP_MISSION_DIR, "youtube_MASTER.json")
    if not os.path.exists(path):
        print("YouTube: file not found, skipping")
        return []
    with open(path, encoding="utf-8") as f:
        yt = json.load(f)
    items = []
    for section in ["liked_videos", "watch_later", "playlists"]:
        for v in yt.get(section, []):
            items.append({
                "url":      v.get("url", ""),
                "title":    v.get("title", ""),
                "channel":  v.get("channel", ""),
                "folder":   f"YouTube / {section}",
                "source":   f"youtube_{section}",
                "tags":     v.get("tags", [])[:5],
                "category": "YouTube & Video",
                "score":    7,
            })
    print(f"YouTube: {len(items)} videos (liked + watch_later + playlists)")
    return items


def merge_all(include_fresh_raindrop: bool = False) -> list[dict]:
    """
    Merge all sources. Returns deduplicated, normalized list.
    Set include_fresh_raindrop=True to also pull live from Raindrop API.
    """
    all_items = []

    all_items.extend(load_categorized())
    all_items.extend(load_edge_bookmarks())
    all_items.extend(load_linkedin())
    all_items.extend(load_youtube())

    if include_fresh_raindrop:
        from openmark.pipeline.raindrop import pull_all
        fresh = pull_all()
        all_items.extend(fresh)

    # Normalize each item
    normalized = [normalize_item(i) for i in all_items]

    # Deduplicate by URL
    unique = dedupe(normalized)
    print(f"\nTotal after merge + dedup: {len(unique)} items")
    return unique
