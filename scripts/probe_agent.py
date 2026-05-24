"""
Agent probe harness — drives build_agent() through representative queries
and captures the full tool-call trace.

Output:
  research/mission_2026_05_24/E1_probe.log   — text trace per query

Run with: python scripts/probe_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "research", "mission_2026_05_24")
OUT_PATH = os.path.join(OUT_DIR, "E1_probe.log")


PROBES = [
    {
        "id": "killer-1",
        "name": "Killer query (exact wording)",
        "query": "get me my bookmarks from a specific date FULLY ALL OF THEM. The date is 2026-05-24.",
        "expect": "find_all_in_range tool call with from_iso=2026-05-24, returns 200+ hits",
    },
    {
        "id": "killer-2",
        "name": "Killer query (calendar week)",
        "query": "show me every bookmark I saved this week — fully all of them",
        "expect": "find_all_in_range over the current week with no LIMIT",
    },
    {
        "id": "exact-1",
        "name": "Exact-keyword retrieval",
        "query": "find anything about muapi in my bookmarks",
        "expect": "search_hybrid finds muapi.ai (BM25-only match)",
    },
    {
        "id": "topic-1",
        "name": "Topic semantic retrieval",
        "query": "what did I save about RAG with knowledge graphs",
        "expect": "search_semantic or search_hybrid returns graphrag.com + Neo4j posts",
    },
    {
        "id": "dive-1",
        "name": "Single-URL dive",
        "query": "dive into https://graphrag.com and tell me related bookmarks",
        "expect": "graph_expand + get_bookmark_full on graphrag.com",
    },
]


def _serialize_tool_calls(messages: list) -> list[dict]:
    calls = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            calls.append({
                "name": tc.get("name", "?") if isinstance(tc, dict) else str(tc),
                "args": tc.get("args", {}) if isinstance(tc, dict) else {},
            })
    return calls


def _final_text(messages: list) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai":
            c = getattr(m, "content", "")
            if isinstance(c, list):
                parts = []
                for b in c:
                    if isinstance(b, dict) and b.get("type", "") in ("text", "output_text", ""):
                        t = (b.get("text") or "").strip()
                        if t:
                            parts.append(t)
                return "\n\n".join(parts)
            return str(c)
    return ""


def main() -> None:
    print("=" * 70)
    print("Agent probe harness — building fresh agent in this process")
    print("=" * 70)
    os.makedirs(OUT_DIR, exist_ok=True)

    from openmark.agent.graph import build_agent
    t0 = time.time()
    agent = build_agent()
    print(f"\nAgent compiled in {time.time() - t0:.1f}s\n")

    log_fh = open(OUT_PATH, "w", encoding="utf-8")
    log_fh.write(f"# E1 — Agent probe — {datetime.now(timezone.utc).isoformat()}\n\n")

    summary: list[dict] = []
    for p in PROBES:
        log_fh.write(f"\n{'=' * 70}\n")
        log_fh.write(f"PROBE {p['id']} — {p['name']}\n")
        log_fh.write(f"  Query   : {p['query']!r}\n")
        log_fh.write(f"  Expect  : {p['expect']}\n")
        log_fh.flush()

        print(f"\n>>> {p['id']}  ({p['name']})")
        print(f"    {p['query']}")

        cfg = {"configurable": {"thread_id": f"probe-{p['id']}"}}
        t0 = time.time()
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": p["query"]}]},
                config=cfg,
            )
            duration = time.time() - t0
            messages = result.get("messages", [])
            tool_calls = _serialize_tool_calls(messages)
            final = _final_text(messages)
            intent = result.get("intent")
            named = result.get("named_skill")
            err = None
        except Exception as e:
            duration = time.time() - t0
            messages = []
            tool_calls = []
            final = ""
            intent = None
            named = None
            err = repr(e)

        log_fh.write(f"  Duration: {duration:.1f}s\n")
        log_fh.write(f"  Intent  : {intent}\n")
        log_fh.write(f"  NamedSkl: {named}\n")
        if err:
            log_fh.write(f"  ERROR   : {err}\n")
        log_fh.write(f"  Tools called ({len(tool_calls)}):\n")
        for tc in tool_calls:
            args_str = json.dumps(tc["args"], default=str)[:200]
            log_fh.write(f"    - {tc['name']}({args_str})\n")
        log_fh.write(f"\n  Final answer ({len(final)} chars):\n")
        log_fh.write(f"  {'-' * 60}\n")
        for line in (final or "(empty)").split("\n"):
            log_fh.write(f"  | {line}\n")
        log_fh.write(f"  {'-' * 60}\n")
        log_fh.flush()

        print(f"    intent={intent} tools={len(tool_calls)} duration={duration:.1f}s err={err}")
        for tc in tool_calls[:6]:
            print(f"      - {tc['name']}({json.dumps(tc['args'], default=str)[:80]})")

        summary.append({
            "id": p["id"],
            "intent": intent,
            "tool_count": len(tool_calls),
            "tool_names": [tc["name"] for tc in tool_calls],
            "duration_s": round(duration, 1),
            "answer_chars": len(final),
            "error": err,
        })

    log_fh.write(f"\n\n{'=' * 70}\nSUMMARY\n{'=' * 70}\n")
    log_fh.write(json.dumps(summary, indent=2))
    log_fh.write("\n")
    log_fh.close()

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for s in summary:
        flag = "[OK]" if not s["error"] and s["tool_count"] > 0 else "[!! ]"
        tools = ",".join(sorted(set(s["tool_names"]))) or "(none)"
        print(f"  {flag} {s['id']:<10} intent={s['intent']:<10} tools=[{tools}] dur={s['duration_s']}s")
        if s["error"]:
            print(f"       ERROR: {s['error'][:120]}")
    print(f"\nFull log: {OUT_PATH}")


if __name__ == "__main__":
    main()
