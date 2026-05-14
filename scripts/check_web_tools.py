"""Smoke-test web.py: 4 functions, ~30s total."""
import sys, json
sys.path.insert(0, r"C:\Users\oasrvadmin\Documents\OpenMark")
sys.stdout.reconfigure(encoding="utf-8")

from openmark.agent import web

print("=== web_search('Open-Generative-AI Anil-matcha') ===")
hits = web.web_search("Open-Generative-AI Anil-matcha github", n=5)
print(f"{len(hits)} hits via", hits[0]["source"] if hits else "(none)")
for h in hits[:3]:
    print(f"  - {h['title'][:70]}\n    {h['url']}")

print("\n=== web_fetch(github readme) ===")
doc = web.web_fetch("https://github.com/Anil-matcha/Open-Generative-AI", max_chars=600)
print(f"status={doc['status']} title={doc['title'][:60]!r} chars={len(doc['markdown'])}")
print(doc["markdown"][:400])

print("\n=== github_repo_intel('Anil-matcha/Open-Generative-AI', days=14) ===")
intel = web.github_repo_intel("Anil-matcha/Open-Generative-AI", days=14)
print(f"status={intel.get('status')}")
if intel.get("meta"):
    m = intel["meta"]
    print(f"  {m['name']} stars={m['stars']} forks={m['forks']} default_branch={m['default_branch']}")
print(f"  recent_commits: {len(intel.get('recent_commits') or [])}")
print(f"  open_prs: {len(intel.get('open_prs') or [])}")
print(f"  readme bytes: {len(intel.get('readme') or '')}")

print("\n=== reddit_search('Open-Generative-AI', sub='muapi') ===")
posts = web.reddit_search("open generative ai", subreddit="muapi", n=3)
print(f"{len(posts)} posts")
for p in posts[:3]:
    print(f"  [{p['score']}↑ {p['num_comments']}💬] r/{p['subreddit']} — {p['title'][:80]}")
