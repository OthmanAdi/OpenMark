"""
Fresh pull of ALL Raindrop bookmarks via API.
Fetches every collection and every raindrop inside it, paginated.
"""

import time
import requests
from openmark import config

HEADERS = {"Authorization": f"Bearer {config.RAINDROP_TOKEN}"}


def fetch_all_collections() -> list[dict]:
    """Return all collections (top-level and nested)."""
    resp = requests.get("https://api.raindrop.io/rest/v1/collections", headers=HEADERS)
    resp.raise_for_status()
    collections = resp.json().get("items", [])

    # Also fetch children
    resp2 = requests.get("https://api.raindrop.io/rest/v1/collections/childrens", headers=HEADERS)
    if resp2.status_code == 200:
        collections += resp2.json().get("items", [])

    return collections


def fetch_raindrops_for_collection(collection_id: int, title: str) -> list[dict]:
    """Fetch all raindrops in a collection, paginated."""
    items = []
    page = 0
    while True:
        resp = requests.get(
            f"https://api.raindrop.io/rest/v1/raindrops/{collection_id}",
            headers=HEADERS,
            params={"perpage": 50, "page": page},
        )
        if resp.status_code != 200:
            break
        batch = resp.json().get("items", [])
        if not batch:
            break
        for item in batch:
            items.append({
                "url":      item.get("link", ""),
                "title":    item.get("title", ""),
                "excerpt":  item.get("excerpt", "")[:200],
                "tags":     item.get("tags", [])[:5],
                "folder":   title,
                "source":   "raindrop",
            })
        if len(batch) < 50:
            break
        page += 1
        time.sleep(0.2)
    return items


def fetch_unsorted() -> list[dict]:
    """Fetch raindrops not in any collection (unsorted)."""
    return fetch_raindrops_for_collection(-1, "Unsorted")


def pull_all() -> list[dict]:
    """Pull every raindrop from every collection. Returns flat list."""
    print("Fetching Raindrop collections...")
    collections = fetch_all_collections()
    print(f"  Found {len(collections)} collections")

    all_items = []
    for col in collections:
        cid   = col["_id"]
        title = col.get("title", "Unknown")
        items = fetch_raindrops_for_collection(cid, title)
        print(f"  [{title}] {len(items)} items")
        all_items.extend(items)
        time.sleep(0.1)

    unsorted = fetch_unsorted()
    print(f"  [Unsorted] {len(unsorted)} items")
    all_items.extend(unsorted)

    print(f"Raindrop total: {len(all_items)}")
    return all_items
