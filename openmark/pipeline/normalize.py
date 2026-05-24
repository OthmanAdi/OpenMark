"""
Normalize, clean, and deduplicate bookmark items.
"""

import re
from datetime import datetime, timezone
from openmark import config


def parse_created_at(value) -> str | None:
    """Accept unix-seconds (int/str) or ISO 8601 string. Return ISO 8601 UTC string or None.

    Edge HTML uses unix-seconds (ADD_DATE), Raindrop + YouTube use ISO 8601 already.
    Neo4j's datetime() will parse the ISO 8601 result at write time.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit():
            try:
                return datetime.fromtimestamp(int(s), tz=timezone.utc).isoformat()
            except (ValueError, OSError):
                return None
        return s  # assume ISO 8601; Neo4j datetime() validates
    return None


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


_LINKEDIN_TRACKING_RE = re.compile(r"[?&]updateentityurn=[^&#]*", re.IGNORECASE)


def canonicalize_url(url: str, source: str = "") -> str:
    """URL canonicalization shared by every ingest path.

    1. Strip whitespace + lowercase.
    2. Strip trailing slashes (graphrag.com/ -> graphrag.com).
    3. For LinkedIn: drop the `updateEntityUrn=...` tracking param, which
       LinkedIn varies per fetch and which historically broke dedup.
       (`urn:li:activity:N` in the path stays — that's the stable ID.)
    """
    u = (url or "").strip()
    if not u:
        return ""
    is_linkedin = (source or "").lower() == "linkedin" or "linkedin.com" in u.lower()
    if is_linkedin:
        u = _LINKEDIN_TRACKING_RE.sub("", u)
        u = u.rstrip("?").rstrip("&")
    return u.rstrip("/").lower()


def normalize_item(item: dict) -> dict:
    """Clean and normalize a single bookmark item."""
    url   = canonicalize_url(item.get("url", ""), item.get("source", ""))
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

    # created_at: accept any of these source keys, convert to ISO 8601 string.
    # Raindrop API uses 'created', Edge HTML parser stuffs unix-secs into 'add_date',
    # other sources may pre-set 'created_at' directly.
    ca = parse_created_at(
        item.get("created_at")
        or item.get("created")
        or item.get("add_date")
        or item.get("published_at")
    )
    if ca:
        normalized["created_at"] = ca

    # Build the document text for embedding
    normalized["doc_text"] = build_document_text(normalized)

    return normalized


def dedupe(items: list[dict]) -> list[dict]:
    """Remove duplicates by URL (uses the same canonicalizer as normalize_item)."""
    seen = set()
    unique = []
    for item in items:
        url = canonicalize_url(item.get("url", ""), item.get("source", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique
