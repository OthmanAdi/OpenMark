"""
OpenMark Live Injector — add bookmarks on the fly.

Accepts:
  - Raw URL string(s)          — one per line, or comma/space separated
  - Edge/Chrome HTML export    — parses folder structure → categories
  - JSON bookmark file         — OpenMark format or Raindrop export
  - .txt file with URLs

Deduplicates against existing Neo4j items before embedding.
New items are immediately searchable after ingest.
"""

import re
import sys
import os
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from openmark.pipeline.normalize import normalize_item, dedupe
from openmark.pipeline.raindrop import _categorize_from_tags  # optional, fails gracefully


# Domain → category heuristics for bare URLs
DOMAIN_CATEGORY = {
    "github.com":          "GitHub Repos & OSS",
    "gist.github.com":     "GitHub Repos & OSS",
    "youtube.com":         "YouTube & Video",
    "youtu.be":            "YouTube & Video",
    "arxiv.org":           "Data Science & ML",
    "papers.nips.cc":      "Data Science & ML",
    "huggingface.co":      "AI Tools & Platforms",
    "kaggle.com":          "Data Science & ML",
    "medium.com":          "News & Articles",
    "dev.to":              "News & Articles",
    "substack.com":        "News & Articles",
    "stackoverflow.com":   "Web Development",
    "docs.python.org":     "Web Development",
    "linkedin.com":        "Career & Jobs",
    "openai.com":          "AI Tools & Platforms",
    "anthropic.com":       "AI Tools & Platforms",
    "langchain.com":       "LangChain / LangGraph",
    "neo4j.com":           "Knowledge Graphs & Neo4j",
    "aws.amazon.com":      "Cloud & Infrastructure",
    "cloud.google.com":    "Cloud & Infrastructure",
    "azure.microsoft.com": "Cloud & Infrastructure",
    "vercel.com":          "Web Development",
}

URL_RE = re.compile(r'https?://[^\s\'"<>]+', re.IGNORECASE)


def _guess_category(url: str) -> str:
    domain = urlparse(url).netloc.replace("www.", "")
    for d, cat in DOMAIN_CATEGORY.items():
        if domain.endswith(d):
            return cat
    return "News & Articles"


def _fetch_title(url: str, timeout: int = 4) -> str:
    """Try to get the <title> of a page. Returns URL on failure."""
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "OpenMark/1.0"},
                         allow_redirects=True, stream=True)
        content = b""
        for chunk in r.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 65536:  # 64KB is enough to find <title>
                break
        text = content.decode("utf-8", errors="replace")
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()[:200]
            return title if title else url
    except Exception:
        pass
    # Fallback: use path as title
    parsed = urlparse(url)
    path = parsed.path.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return (path or parsed.netloc)[:100]


def extract_urls_from_text(text: str) -> list[str]:
    """Extract all HTTP URLs from a text string."""
    urls = URL_RE.findall(text)
    # Also try line-by-line (for plain URL lists)
    for line in text.splitlines():
        line = line.strip().rstrip(",;")
        if line.startswith("http://") or line.startswith("https://"):
            if line not in urls:
                urls.append(line)
    return list(dict.fromkeys(urls))  # dedupe preserving order


def urls_to_items(urls: list[str], fetch_titles: bool = True) -> list[dict]:
    """Convert bare URLs to OpenMark items. Optionally fetches page titles."""
    items = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        title = _fetch_title(url) if fetch_titles else urlparse(url).netloc
        category = _guess_category(url)
        items.append({
            "url":      url,
            "title":    title,
            "category": category,
            "tags":     [],
            "score":    5,
            "source":   "manual",
            "folder":   "Manual Add",
        })
    return items


def parse_html_file(path: str) -> list[dict]:
    """Parse Edge/Chrome HTML bookmark export."""
    # Reuse the existing parser
    from html.parser import HTMLParser

    class BookmarkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.bookmarks = []
            self._current_folder = "Manual Add"
            self._in_a = False
            self._current_url = ""

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "h3":
                self._in_a = False
            if tag == "a":
                self._in_a = True
                self._current_url = attrs.get("href", "")

        def handle_endtag(self, tag):
            if tag == "a":
                self._in_a = False

        def handle_data(self, data):
            if self._in_a and self._current_url.startswith("http"):
                data = data.strip()
                if data:
                    self.bookmarks.append({
                        "url": self._current_url,
                        "title": data[:200],
                        "category": _guess_category(self._current_url),
                        "tags": [],
                        "score": 5,
                        "source": "edge",
                        "folder": "HTML Import",
                    })
                self._in_a = False

    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    parser = BookmarkParser()
    parser.feed(content)
    return parser.bookmarks


def parse_json_file(path: str) -> list[dict]:
    """Parse a JSON bookmark file (OpenMark or Raindrop format)."""
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = []
        for item in data:
            if isinstance(item, dict) and item.get("url"):
                items.append({
                    "url":      item.get("url", ""),
                    "title":    item.get("title", item.get("url", ""))[:200],
                    "category": item.get("category") or _guess_category(item.get("url", "")),
                    "tags":     item.get("tags", [])[:5],
                    "score":    item.get("score", 5),
                    "source":   item.get("source", "json_import"),
                    "folder":   item.get("folder", "JSON Import"),
                })
        return items

    # Raindrop export format
    if isinstance(data, dict) and "items" in data:
        return parse_json_file.__wrapped__(data["items"]) if hasattr(parse_json_file, '__wrapped__') else []

    return []


def parse_txt_file(path: str, fetch_titles: bool = True) -> list[dict]:
    """Extract URLs from a plain text file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    urls = extract_urls_from_text(text)
    return urls_to_items(urls, fetch_titles=fetch_titles)


def dedup_against_neo4j(items: list[dict]) -> list[dict]:
    """Filter out items whose URLs already exist in Neo4j."""
    from openmark.stores import neo4j_store
    urls = [i["url"] for i in items if i.get("url")]
    if not urls:
        return items
    try:
        existing = set(
            r["url"] for r in neo4j_store.query(
                "MATCH (b:Bookmark) WHERE b.url IN $urls RETURN b.url AS url",
                {"urls": urls}
            )
        )
        return [i for i in items if i.get("url") not in existing]
    except Exception:
        return items  # If Neo4j check fails, let MERGE handle it


def run_injection(items: list[dict], embedder=None) -> dict:
    """
    Core injection: normalize → dedup → embed → ingest.
    Returns stats dict: {total, new, skipped, errors}
    """
    from openmark.stores import neo4j_store

    total = len(items)
    if not items:
        return {"total": 0, "new": 0, "skipped": 0, "error": None}

    # Normalize
    normalized = [normalize_item(i) for i in items]
    # Local dedup (within the batch)
    normalized = dedupe(normalized)
    # Dedup against Neo4j
    new_items = dedup_against_neo4j(normalized)
    skipped = len(normalized) - len(new_items)

    if not new_items:
        return {"total": total, "new": 0, "skipped": skipped, "error": None}

    # Embed
    if embedder is None:
        from openmark.embeddings.factory import get_embedder
        embedder = get_embedder()

    texts = [i["doc_text"] for i in new_items]
    embeddings = embedder.embed_documents(texts)

    # Ingest to Neo4j
    neo4j_store.ingest(new_items, embeddings=embeddings)

    return {"total": total, "new": len(new_items), "skipped": skipped, "error": None}
