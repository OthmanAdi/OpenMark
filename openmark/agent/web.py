"""
Web research toolkit for the OpenMark in-app agent.

Design goals:
  - Zero required API keys. Works on a fresh clone.
  - Hardened: 30 s timeouts, exponential backoff, UA rotation, multi-provider
    fallback so one rate-limit doesn't break the chain.
  - No artificial guardrails — robots.txt ignored, User-Agent spoofed to a
    recent Chrome string. Used for personal research.
  - Optional upgrade path: set TAVILY_API_KEY or BRAVE_API_KEY in .env and
    web_search() prefers them automatically (higher quality results).

Surface (called by langchain @tool wrappers in tools.py):

    web_search(query, n=8)            → list[dict] {title, url, snippet, source}
    web_fetch(url)                    → dict {url, title, markdown, status}
    github_repo_intel(slug)           → dict {meta, readme_md, recent_commits, open_prs}
    reddit_search(query, sub=None, n) → list[dict] {title, url, subreddit, score, num_comments, body_snippet}
"""

from __future__ import annotations

import os
import re
import time
import random
import logging
from typing import Optional, Any
from urllib.parse import quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as _md

log = logging.getLogger("openmark.web")
if not log.handlers:
    import sys as _sys
    h = logging.StreamHandler(_sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [%(name)s] %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False


# ── UA pool + http client factory ─────────────────────────────────────────────
_UA_POOL = [
    # Recent Chrome + Edge + Firefox on Win11/macOS — enough variance to dodge naive blocks.
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
]


def _client(timeout: float = 30.0, follow_redirects: bool = True) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        headers={
            "User-Agent": random.choice(_UA_POOL),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )


def _retry(fn, *, attempts: int = 3, base: float = 0.6, what: str = "op"):
    """Exponential backoff. Returns fn() or raises the last exception."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            wait = base * (2 ** i)
            log.info(f"[retry] {what} attempt {i+1}/{attempts} failed ({type(e).__name__}: {str(e)[:80]}); sleep {wait:.1f}s")
            time.sleep(wait)
    raise last  # type: ignore[misc]


# ── web_search — DDG default, Tavily/Brave preferred when keys present ──────

def _tavily_key() -> Optional[str]:
    return os.getenv("TAVILY_API_KEY")


def _tavily(query: str, n: int) -> list[dict]:
    """Tavily /search advanced. Returns the LLM-friendly answer as the first
    pseudo-hit when present so the agent gets the synthesis even without
    fetching pages."""
    key = _tavily_key()
    if not key:
        return []
    with _client() as c:
        r = c.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key, "query": query, "max_results": n,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
            },
        )
        r.raise_for_status()
        data = r.json()

    out: list[dict] = []
    ans = (data.get("answer") or "").strip()
    if ans:
        out.append({"title": "Tavily synthesized answer", "url": "tavily://answer",
                    "snippet": ans[:600], "source": "tavily:answer"})
    for x in (data.get("results") or [])[:n]:
        out.append({"title": x.get("title", "") or "", "url": x.get("url", "") or "",
                    "snippet": (x.get("content") or "")[:300], "source": "tavily"})
    return out


def tavily_extract(urls: list[str], depth: str = "advanced") -> list[dict]:
    """
    Tavily /extract — turn one or more URLs into clean raw_content in one
    call. Way more reliable than httpx+markdownify for JS-heavy pages
    (LinkedIn, Reddit, GitHub gists, paywalled previews, etc).
    Returns [{url, raw_content, status}, ...].
    """
    key = _tavily_key()
    if not key or not urls:
        return []
    with _client(timeout=60.0) as c:
        r = c.post(
            "https://api.tavily.com/extract",
            json={"api_key": key, "urls": urls, "extract_depth": depth},
        )
        r.raise_for_status()
        data = r.json()
    out: list[dict] = []
    for x in (data.get("results") or []):
        out.append({"url": x.get("url"), "raw_content": x.get("raw_content") or "", "status": "ok"})
    for x in (data.get("failed_results") or []):
        out.append({"url": x.get("url"), "raw_content": "",
                    "status": f"failed: {x.get('error', 'unknown')}"})
    return out


def tavily_crawl(
    seed_url: str,
    *,
    max_depth: int = 1,
    max_breadth: int = 5,
    limit: int = 8,
    instructions: Optional[str] = None,
) -> list[dict]:
    """
    Tavily /crawl — follow links from `seed_url` up to `max_depth` hops and
    `max_breadth` links per page, collecting up to `limit` pages.
    `instructions` is a natural-language steer ("focus on community plugins
    and workflow templates") that Tavily uses to score which links to follow.

    Note: /crawl uses Bearer auth header (different from /search and /extract).
    """
    key = _tavily_key()
    if not key or not seed_url:
        return []
    payload: dict = {"url": seed_url, "max_depth": max_depth,
                     "max_breadth": max_breadth, "limit": limit}
    if instructions:
        payload["instructions"] = instructions
    with _client(timeout=120.0) as c:
        c.headers["Authorization"] = f"Bearer {key}"
        r = c.post("https://api.tavily.com/crawl", json=payload)
        r.raise_for_status()
        data = r.json()
    out = []
    for x in (data.get("results") or []):
        out.append({"url": x.get("url"), "raw_content": x.get("raw_content") or "",
                    "status": "ok"})
    return out


def _brave(query: str, n: int) -> list[dict]:
    key = os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY")
    if not key:
        return []
    with _client() as c:
        r = c.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(n, 20)},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
    web = (data.get("web") or {}).get("results", [])
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""),
         "snippet": (x.get("description") or "")[:280], "source": "brave"}
        for x in web[:n]
    ]


def _ddg(query: str, n: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except Exception as e:
        log.info(f"[ddg] not installed: {e}")
        return []
    out: list[dict] = []
    with DDGS() as d:
        for hit in d.text(query, max_results=n, safesearch="off", region="wt-wt"):
            out.append({
                "title": hit.get("title", "") or "",
                "url":   hit.get("href", "") or hit.get("url", "") or "",
                "snippet": (hit.get("body", "") or "")[:280],
                "source": "duckduckgo",
            })
    return out


def web_search(query: str, n: int = 8) -> list[dict]:
    """Search the open web. Tavily > Brave > DuckDuckGo, falling back per provider."""
    query = (query or "").strip()
    if not query:
        return []
    t0 = time.time()
    for fn, name in ((_tavily, "tavily"), (_brave, "brave"), (_ddg, "duckduckgo")):
        try:
            hits = _retry(lambda: fn(query, n), attempts=2, what=f"search:{name}")
            if hits:
                log.info(f"[web_search] {name} returned {len(hits)} hits in {round((time.time()-t0)*1000)}ms")
                return hits
        except Exception as e:
            log.info(f"[web_search] {name} failed: {e}")
    log.info(f"[web_search] ALL providers empty for query={query!r}")
    return []


# ── web_fetch — fetch + readability + markdown ────────────────────────────────

_BOILERPLATE = re.compile(
    r"(cookie|consent|signin|sign in|sign up|subscribe|newsletter|navigation|footer|"
    r"copyright|all rights reserved|privacy policy)",
    re.IGNORECASE,
)


def _extract_main(html: str) -> tuple[str, str]:
    """Return (title, main_text_markdown)."""
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and not title:
        title = h1.get_text(strip=True)

    # Drop noise
    for tag in soup(["script", "style", "noscript", "nav", "aside", "form",
                     "header", "footer", "iframe", "svg"]):
        tag.decompose()

    # Prefer <article>, <main>, then largest <div>
    candidate = soup.find("article") or soup.find("main")
    if not candidate:
        # Heuristic: pick the <div> with the most paragraph text.
        best, best_len = None, 0
        for div in soup.find_all("div"):
            p_text = " ".join(p.get_text(" ", strip=True) for p in div.find_all("p", recursive=False))
            if len(p_text) > best_len:
                best, best_len = div, len(p_text)
        candidate = best or soup.body or soup

    md = _md(str(candidate), heading_style="ATX", strip=["a-img"])
    # Squash whitespace
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return title, md


def web_fetch(url: str, max_chars: int = 12000) -> dict:
    """
    Fetch one URL, return cleaned-up markdown of the main content.
    Path:
      1. If TAVILY_API_KEY set → use Tavily /extract (advanced depth). Reliable
         on JS-heavy pages, paywall previews, LinkedIn, GitHub blob views.
      2. Otherwise fall back to httpx + BeautifulSoup + markdownify.
    Caps output at max_chars so it doesn't blow agent context.
    """
    url = (url or "").strip()
    if not url:
        return {"url": url, "title": "", "markdown": "", "status": "empty_url"}

    # --- Path 1: Tavily extract ---
    if _tavily_key():
        try:
            res = tavily_extract([url], depth="advanced")
            if res and res[0].get("status") == "ok" and res[0].get("raw_content"):
                md = res[0]["raw_content"]
                # Tavily already gives clean text; just take first H1 or first line as title.
                first = md.lstrip().split("\n", 1)[0].strip("# ").strip()[:120] or url
                if len(md) > max_chars:
                    md = md[:max_chars].rstrip() + "\n\n…(truncated)"
                log.info(f"[web_fetch:tavily] OK {url} chars={len(md)}")
                return {"url": url, "title": first, "markdown": md, "status": "ok"}
            log.info(f"[web_fetch:tavily] empty, falling back to httpx for {url}")
        except Exception as e:
            log.info(f"[web_fetch:tavily] {url} :: {e}; falling back")

    # --- Path 2: httpx + readability fallback ---
    def _do():
        with _client(timeout=30.0) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.text

    try:
        html = _retry(_do, attempts=3, what=f"fetch:{urlparse(url).netloc}")
    except Exception as e:
        log.info(f"[web_fetch] FAIL {url} :: {e}")
        return {"url": url, "title": "", "markdown": "", "status": f"error: {e}"}

    title, md = _extract_main(html)
    if len(md) > max_chars:
        md = md[:max_chars].rstrip() + "\n\n…(truncated)"
    log.info(f"[web_fetch:httpx] OK {url} title={title[:60]!r} chars={len(md)}")
    return {"url": url, "title": title, "markdown": md, "status": "ok"}


# ── github_repo_intel — GitHub API (unauth, 60/hr is enough for one report) ──

_REPO_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)", re.IGNORECASE)


def _gh_headers() -> dict:
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28",
         "User-Agent": random.choice(_UA_POOL)}
    tok = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _parse_slug(slug_or_url: str) -> Optional[str]:
    s = (slug_or_url or "").strip().rstrip("/")
    m = _REPO_RE.search(s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if "/" in s and " " not in s and len(s.split("/")) == 2:
        return s
    return None


def github_repo_intel(slug_or_url: str, days: int = 30) -> dict:
    """
    Pull a quick intel snapshot of a public repo:
      - core meta (name, description, stars, forks, language, license, default_branch)
      - README markdown (decoded)
      - recent commits to default branch over the last `days`
      - open PRs (top 15)
      - open issues count
    """
    slug = _parse_slug(slug_or_url)
    if not slug:
        return {"status": f"error: could not parse repo slug from {slug_or_url!r}"}

    log.info(f"[github] intel slug={slug} days={days}")
    base = f"https://api.github.com/repos/{slug}"
    out: dict[str, Any] = {"slug": slug, "status": "ok"}

    with _client(timeout=30.0) as c:
        c.headers.update(_gh_headers())

        # Meta
        r = c.get(base)
        if r.status_code == 404:
            return {"slug": slug, "status": "error: repo not found (private or wrong slug?)"}
        r.raise_for_status()
        meta = r.json()
        out["meta"] = {
            "name": meta.get("full_name"),
            "description": meta.get("description"),
            "stars": meta.get("stargazers_count"),
            "forks": meta.get("forks_count"),
            "watchers": meta.get("subscribers_count"),
            "language": meta.get("language"),
            "license": (meta.get("license") or {}).get("spdx_id"),
            "default_branch": meta.get("default_branch"),
            "open_issues": meta.get("open_issues_count"),
            "topics": meta.get("topics", []),
            "homepage": meta.get("homepage"),
            "html_url": meta.get("html_url"),
            "created_at": meta.get("created_at"),
            "pushed_at": meta.get("pushed_at"),
        }

        # README (try main contents endpoint)
        try:
            r = c.get(f"{base}/readme")
            if r.status_code == 200:
                import base64 as _b64
                data = r.json()
                content = _b64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
                if len(content) > 16000:
                    content = content[:16000].rstrip() + "\n\n…(truncated)"
                out["readme"] = content
        except Exception as e:
            log.info(f"[github] readme fetch failed: {e}")

        # Recent commits on default branch
        try:
            from datetime import datetime, timezone, timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            branch = out["meta"]["default_branch"] or "main"
            r = c.get(f"{base}/commits", params={"sha": branch, "since": since, "per_page": 50})
            if r.status_code == 200:
                out["recent_commits"] = [
                    {
                        "sha": ci["sha"][:7],
                        "author": (ci.get("author") or {}).get("login") or
                                  ((ci.get("commit") or {}).get("author") or {}).get("name", ""),
                        "date": ((ci.get("commit") or {}).get("author") or {}).get("date", ""),
                        "message": ((ci.get("commit") or {}).get("message") or "").split("\n")[0][:160],
                        "url": ci.get("html_url"),
                    }
                    for ci in r.json()[:50]
                ]
                out["recent_commits_window_days"] = days
        except Exception as e:
            log.info(f"[github] commits fetch failed: {e}")

        # Open PRs
        try:
            r = c.get(f"{base}/pulls", params={"state": "open", "per_page": 15, "sort": "updated"})
            if r.status_code == 200:
                out["open_prs"] = [
                    {"number": p["number"], "title": p["title"], "user": (p.get("user") or {}).get("login"),
                     "updated_at": p.get("updated_at"), "url": p.get("html_url")}
                    for p in r.json()
                ]
        except Exception as e:
            log.info(f"[github] PRs fetch failed: {e}")

    return out


# ── reddit_search — public JSON API (no auth) ─────────────────────────────────

def reddit_search(query: str, subreddit: Optional[str] = None, n: int = 15) -> list[dict]:
    """
    Search Reddit for a query. Optional subreddit ('muapi', 'LocalLLaMA', etc).
    Returns up to n posts with title, url, score, num_comments, subreddit,
    and a short body snippet.
    """
    query = (query or "").strip()
    if not query:
        return []
    base = "https://www.reddit.com"
    if subreddit:
        url = f"{base}/r/{quote_plus(subreddit)}/search.json"
        params = {"q": query, "restrict_sr": "on", "sort": "relevance", "limit": min(n, 25),
                  "t": "all"}
    else:
        url = f"{base}/search.json"
        params = {"q": query, "sort": "relevance", "limit": min(n, 25), "t": "all"}

    def _do():
        with _client(timeout=20.0) as c:
            c.headers["User-Agent"] = "openmark-research/0.1 (by /u/anon)"
            r = c.get(url, params=params)
            r.raise_for_status()
            return r.json()

    try:
        data = _retry(_do, attempts=3, what="reddit")
    except Exception as e:
        log.info(f"[reddit] FAIL: {e}")
        return []

    children = (data.get("data") or {}).get("children", []) or []
    out = []
    for ch in children[:n]:
        d = ch.get("data", {})
        body = d.get("selftext", "") or ""
        out.append({
            "title": d.get("title", ""),
            "url": "https://reddit.com" + d.get("permalink", ""),
            "external_url": d.get("url_overridden_by_dest") or d.get("url"),
            "subreddit": d.get("subreddit"),
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "created_utc": d.get("created_utc"),
            "body_snippet": (body[:280] + "…") if len(body) > 280 else body,
        })
    log.info(f"[reddit] q={query!r} sub={subreddit} hits={len(out)}")
    return out
