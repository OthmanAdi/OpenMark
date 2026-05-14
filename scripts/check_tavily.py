"""Verify Tavily key + advanced endpoints (search / extract / crawl)."""
import os, sys, httpx, json
sys.path.insert(0, r"C:\Users\oasrvadmin\Documents\OpenMark")
sys.stdout.reconfigure(encoding="utf-8")

# Force-load .env (override existing env so the fresh key wins)
from dotenv import load_dotenv
load_dotenv(r"C:\Users\oasrvadmin\Documents\OpenMark\.env", override=True)

KEY = os.getenv("TAVILY_API_KEY")
print("key loaded:", bool(KEY), "prefix:", (KEY or "")[:8] + "...")
H = {"Content-Type": "application/json"}

print("\n=== /search advanced ===")
r = httpx.post("https://api.tavily.com/search",
               json={"api_key": KEY, "query": "Open-Generative-AI Anil-matcha",
                     "max_results": 5, "search_depth": "advanced", "include_answer": True,
                     "include_raw_content": False},
               headers=H, timeout=30)
print("status:", r.status_code)
if r.status_code == 200:
    d = r.json()
    print("answer:", (d.get("answer") or "")[:200])
    for x in d.get("results", [])[:3]:
        print(f"  - {x.get('title','')[:70]}\n    {x.get('url','')}")
else:
    print(r.text[:300])

print("\n=== /extract (single URL) ===")
r = httpx.post("https://api.tavily.com/extract",
               json={"api_key": KEY,
                     "urls": ["https://github.com/Anil-matcha/Open-Generative-AI"],
                     "extract_depth": "advanced"},
               headers=H, timeout=60)
print("status:", r.status_code)
if r.status_code == 200:
    d = r.json()
    for x in d.get("results", []):
        rc = x.get("raw_content") or ""
        print(f"  url={x.get('url')}\n  chars={len(rc)}\n  head: {rc[:240]}")
else:
    print(r.text[:300])

print("\n=== /crawl (1 hop, 3 pages) ===")
r = httpx.post("https://api.tavily.com/crawl",
               json={"api_key": KEY,
                     "url": "https://github.com/Anil-matcha/Open-Generative-AI",
                     "max_depth": 1, "max_breadth": 3, "limit": 3},
               headers=H, timeout=90)
print("status:", r.status_code)
if r.status_code == 200:
    d = r.json()
    print("pages:", len(d.get("results", [])))
    for x in d.get("results", [])[:3]:
        rc = x.get("raw_content") or ""
        print(f"  - {x.get('url','')} ({len(rc)} chars)")
else:
    print(r.text[:300])
