"""
Parse Edge/Chrome/Firefox exported HTML bookmarks into OpenMark's normalized JSON format.

Usage:
    python scripts/parse_edge_bookmarks.py <bookmarks.html> [output.json]

Default output: data/edge_bookmarks.json
"""

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

# Map Edge folder names → OpenMark canonical categories
FOLDER_CATEGORY_MAP = {
    # AI / Agent
    "ai_agent_development":        "Agent Development",
    "ai_development":              "Agent Development",
    "ai development":              "Agent Development",
    "ai development & tools":      "Agent Development",
    "agent development":           "Agent Development",
    "agent dev":                   "Agent Development",
    "ai frameworks & tools":       "AI Tools & Platforms",
    "open_responses_api":          "MCP & Tool Use",
    "mcp & tool use":              "MCP & Tool Use",
    "mcp":                         "MCP & Tool Use",
    "tool use":                    "MCP & Tool Use",
    "context engineering":         "Context Engineering",
    "langchain_documentation":     "LangChain / LangGraph",
    "langchain / langgraph":       "LangChain / LangGraph",
    "langchain":                   "LangChain / LangGraph",
    "langgraph":                   "LangChain / LangGraph",
    "llm fine-tuning":             "LLM Fine-tuning",
    "llm fine tuning":             "LLM Fine-tuning",
    "fine-tuning":                 "LLM Fine-tuning",
    "rag & vector search":         "RAG & Vector Search",
    "rag":                         "RAG & Vector Search",
    "vector search":               "RAG & Vector Search",
    "ai & machine learning":       "AI Tools & Platforms",
    "ai tools & platforms":        "AI Tools & Platforms",
    "ai tools":                    "AI Tools & Platforms",
    "email & productivity":        "AI Tools & Platforms",
    "debugging & tools":           "AI Tools & Platforms",
    # Code / Dev
    "web development":             "Web Development",
    "web develo":                  "Web Development",
    " web development":            "Web Development",
    "software development":        "Web Development",
    "developer resources":         "Web Development",
    "github_projects":             "GitHub Repos & OSS",
    "github repos & oss":          "GitHub Repos & OSS",
    "github":                      "GitHub Repos & OSS",
    "cloud & infrastructure":      "Cloud & Infrastructure",
    "cloud & infr":                "Cloud & Infrastructure",
    "cloud":                       "Cloud & Infrastructure",
    # Data
    "data science & ml":           "Data Science & ML",
    " data science":               "Data Science & ML",
    "data science":                "Data Science & ML",
    "knowledge graphs & neo4j":    "Knowledge Graphs & Neo4j",
    "research":                    "News & Articles",
    # Learning
    "learning & courses":          "Learning & Courses",
    "learning_resources":          "Learning & Courses",
    "learning resources":          "Learning & Courses",
    "education":                   "Learning & Courses",
    "daily.dev_articles":          "News & Articles",
    "developer blogs & articles":  "News & Articles",
    "blogs & articles":            "News & Articles",
    "news & articles":             "News & Articles",
    "news & arti":                 "News & Articles",
    # Media
    "youtube_videos":              "YouTube & Video",
    "youtube videos":              "YouTube & Video",
    "youtube & video":             "YouTube & Video",
    "youtube":                     "YouTube & Video",
    "linkedin_events":             "News & Articles",
    # Design
    "design & ui/ux":              "Design & UI/UX",
    "design & ui":                 "Design & UI/UX",
    "ui/ux design":                "Design & UI/UX",
    "ui & design":                 "Design & UI/UX",
    # Career
    "career & jobs":               "Career & Jobs",
    "career_jobs":                 "Career & Jobs",
    "career & job search":         "Career & Jobs",
    "job search & career":         "Career & Jobs",
    "career":                      "Career & Jobs",
    # Finance
    "finance & crypto":            "Finance & Crypto",
    "banking & finance":           "Finance & Crypto",
    "finance":                     "Finance & Crypto",
    "real estate":                 "Finance & Crypto",
    "real_estate":                 "Finance & Crypto",
    "prediction markets":          "Finance & Crypto",
    # Other
    "patents & legal":             "Entertainment & Other",
    "legal":                       "Entertainment & Other",
    "music":                       "Entertainment & Other",
    "gaming":                      "Entertainment & Other",
    "games & entertainment":       "Entertainment & Other",
    "communication":               "Entertainment & Other",
    "marketplaces":                "News & Articles",
    "e-commerce & marketplaces":   "News & Articles",
    "social_media":                "News & Articles",
    "migraven tools & news":       "AI Tools & Platforms",
    "migraven":                    "AI Tools & Platforms",
    "entertainment & other":       "Entertainment & Other",
    # Container/date folders — return None to inherit from sibling/child
    "tools and articles":          None,
    "opentabs":                    None,
    "2022":                        None,
    "2025 latest":                 None,
    "2026":                        None,
    "newest":                      None,
    "newnew":                      None,
    "new fav 2026/5":              None,
    "short 2026/4":                None,
    "2026/3":                      None,
    "2026/2":                      None,
    "favorites bar":               None,
    "group 1":                     None,
    "group 9":                     None,
}

# Domain → category fallback when folder gives no signal
DOMAIN_CATEGORY_MAP = {
    "github.com":          "GitHub Repos & OSS",
    "youtube.com":         "YouTube & Video",
    "youtu.be":            "YouTube & Video",
    "huggingface.co":      "AI Tools & Platforms",
    "arxiv.org":           "News & Articles",
    "linkedin.com":        "News & Articles",
    "medium.com":          "News & Articles",
    "substack.com":        "News & Articles",
    "twitter.com":         "News & Articles",
    "x.com":               "News & Articles",
    "reddit.com":          "News & Articles",
    "stackoverflow.com":   "Web Development",
    "dev.to":              "Web Development",
    "docs.python.org":     "Web Development",
    "npmjs.com":           "Web Development",
    "pypi.org":            "Web Development",
    "openai.com":          "AI Tools & Platforms",
    "anthropic.com":       "AI Tools & Platforms",
    "langchain.com":       "LangChain / LangGraph",
    "python.langchain.com":"LangChain / LangGraph",
    "neo4j.com":           "Knowledge Graphs & Neo4j",
    "udemy.com":           "Learning & Courses",
    "coursera.org":        "Learning & Courses",
    "daily.dev":           "News & Articles",
}

CANONICAL = {
    "RAG & Vector Search", "LLM Fine-tuning", "Agent Development",
    "LangChain / LangGraph", "MCP & Tool Use", "Context Engineering",
    "AI Tools & Platforms", "GitHub Repos & OSS", "Learning & Courses",
    "YouTube & Video", "Web Development", "Cloud & Infrastructure",
    "Data Science & ML", "Knowledge Graphs & Neo4j", "Career & Jobs",
    "Finance & Crypto", "Design & UI/UX", "News & Articles",
    "Entertainment & Other",
}


def folder_to_category(folder_path: list[str], url: str = "") -> str:
    """Walk folder path deepest→shallowest, then fall back to domain."""
    for folder in reversed(folder_path):
        key = folder.strip().lower()
        cat = FOLDER_CATEGORY_MAP.get(key)
        if cat and cat in CANONICAL:
            return cat
        # Prefix match (handles "Tools and Articles 5" etc)
        for k, v in FOLDER_CATEGORY_MAP.items():
            if k and len(k) >= 6 and key.startswith(k[:8]) and v and v in CANONICAL:
                return v

    # Domain-based fallback
    if url:
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower().lstrip("www.")
            for d, cat in DOMAIN_CATEGORY_MAP.items():
                if domain == d or domain.endswith("." + d):
                    return cat
        except Exception:
            pass

    return "News & Articles"


class BookmarkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.bookmarks: list[dict] = []
        self._folder_stack: list[str] = []
        self._in_link = False
        self._current_href = ""
        self._current_add_date = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h3":
            # Push folder name — we'll set it when we see the text
            self._folder_stack.append("")
        elif tag == "a":
            self._in_link = True
            self._current_href = attrs.get("href", "")
            self._current_add_date = attrs.get("add_date", "")
            self._current_title = ""

    def handle_endtag(self, tag):
        if tag == "dl":
            if self._folder_stack:
                self._folder_stack.pop()
        elif tag == "a":
            self._in_link = False
            url = self._current_href.strip()
            title = self._current_title.strip()
            if url and title and url.startswith("http"):
                folder_path = [f for f in self._folder_stack if f]
                self.bookmarks.append({
                    "url":    url,
                    "title":  title[:300],
                    "folder": " / ".join(folder_path) if folder_path else "",
                    "folder_path": list(folder_path),
                    "source": "edge",
                    "tags":   [],
                    "score":  5,
                })

    def handle_data(self, data):
        if self._in_link:
            self._current_title += data
        elif self._folder_stack and self._folder_stack[-1] == "":
            # This text is the folder name
            self._folder_stack[-1] = data.strip()


def clean_title(title: str) -> str:
    title = re.sub(r"&amp;", "&", title)
    title = re.sub(r"&lt;", "<", title)
    title = re.sub(r"&gt;", ">", title)
    title = re.sub(r"&#39;", "'", title)
    title = re.sub(r"&quot;", '"', title)
    return title.strip()


def build_doc_text(item: dict) -> str:
    parts = [item["title"], item["category"]]
    if item["folder"]:
        parts.append(item["folder"])
    return " | ".join(p for p in parts if p)


def parse(html_path: str) -> list[dict]:
    with open(html_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    parser = BookmarkParser()
    parser.feed(content)

    results = []
    seen = set()
    for bm in parser.bookmarks:
        url = bm["url"].rstrip("/").lower()
        if url in seen:
            continue
        seen.add(url)

        title = clean_title(bm["title"])
        category = folder_to_category(bm.get("folder_path", []), bm["url"])

        item = {
            "url":      bm["url"],
            "title":    title,
            "category": category,
            "tags":     [],
            "score":    5,
            "source":   "edge",
            "folder":   bm["folder"],
        }
        item["doc_text"] = build_doc_text(item)
        results.append(item)

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/parse_edge_bookmarks.py <bookmarks.html> [output.json]")
        sys.exit(1)

    html_path = sys.argv[1]
    out_path  = sys.argv[2] if len(sys.argv) > 2 else "data/edge_bookmarks.json"

    print(f"Parsing {html_path} ...")
    items = parse(html_path)
    print(f"Parsed {len(items)} unique bookmarks")

    # Category breakdown
    from collections import Counter
    cats = Counter(i["category"] for i in items)
    print("\nCategory breakdown:")
    for cat, count in cats.most_common():
        print(f"  {cat:<35} {count}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
