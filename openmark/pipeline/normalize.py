"""
Normalize, clean, and deduplicate bookmark items.
"""

import re
from openmark import config


def clean_title(title: str) -> str:
    if not title:
        return ""
    # Strip HTML entities
    title = re.sub(r"&amp;", "&", title)
    title = re.sub(r"&lt;", "<", title)
    title = re.sub(r"&gt;", ">", title)
    title = re.sub(r"&#39;", "'", title)
    title = re.sub(r"&quot;", '"', title)
    # Strip leading/trailing whitespace and truncate
    title = title.strip()[:300]
    return title


def fix_category(cat: str | None) -> str:
    if not cat:
        return "News & Articles"
    # Apply known remapping
    cat = config.CATEGORY_MAP.get(cat, cat)
    # If still unknown, fallback
    if cat not in config.CATEGORIES:
        return "News & Articles"
    return cat


def build_document_text(item: dict) -> str:
    """
    Build a single rich text string for embedding.
    Combines title + tags + category + content/excerpt for better semantic matching.
    """
    parts = []
    if item.get("title"):
        parts.append(item["title"])
    if item.get("category"):
        parts.append(item["category"])
    if item.get("tags"):
        parts.append(" ".join(item["tags"]))
    if item.get("content"):
        parts.append(item["content"][:200])
    elif item.get("excerpt"):
        parts.append(item["excerpt"][:200])
    if item.get("channel"):
        parts.append(item["channel"])
    if item.get("author"):
        parts.append(item["author"])
    return " | ".join(p for p in parts if p)


def normalize_item(item: dict) -> dict:
    """Clean and normalize a single bookmark item."""
    url   = item.get("url", "").strip()
    title = clean_title(item.get("title", ""))
    cat   = fix_category(item.get("category"))
    tags  = [t.lower().strip() for t in item.get("tags", []) if t][:5]
    score = item.get("score", 5)
    if not isinstance(score, (int, float)):
        score = 5

    normalized = {
        "url":      url,
        "title":    title,
        "category": cat,
        "tags":     tags,
        "score":    score,
        "source":   item.get("source", "unknown"),
        "folder":   item.get("folder", ""),
    }

    # Preserve optional fields
    for field in ["content", "excerpt", "author", "channel", "description"]:
        if item.get(field):
            normalized[field] = item[field][:300]

    # Build the document text for embedding
    normalized["doc_text"] = build_document_text(normalized)

    return normalized


def dedupe(items: list[dict]) -> list[dict]:
    """Remove duplicates by URL (case-insensitive, trailing slash stripped)."""
    seen = set()
    unique = []
    for item in items:
        url = item.get("url", "").rstrip("/").lower()
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique
