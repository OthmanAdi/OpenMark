"""
Focused re-probe — drives the 2 queries that failed in E1 through the agent
chain, capturing the researcher's INTERNAL tool calls via the event bus.

This is the user's acceptance gate:
  - killer-2: "show me every bookmark I saved this week ... fully all of them"
              MUST resolve via find_all_in_range with a window that includes today.
  - exact-1:  "find anything about muapi in my bookmarks"
              MUST surface muapi.ai in the final answer.

Output: research/mission_2026_05_24/E1_reprobe.log
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
OUT_PATH = os.path.join(OUT_DIR, "E1_reprobe.log")


PROBES = [
    {
        "id": "killer-2-v2",
        "query": "show me every bookmark I saved this week — fully all of them",
        "expect": "find_all_in_range tool called with from <= 2026-05-18 and to >= 2026-05-25 (covers 2026-05-24's 268 nodes). Final answer should mention ~268 items.",
    },
    {
        "id": "exact-1-v2",
        "query": "find anything about muapi in my bookmarks",
        "expect": "search_hybrid called with query=muapi. Final answer mentions muapi.ai.",
    },
]


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
    print("Re-probe — drives the 2 previously failing queries")
    print("=" * 70)
    os.makedirs(OUT_DIR, exist_ok=True)

    from openmark.agent.graph import build_agent
    from openmark.agent.middleware import drain_events

    t0 = time.time()
    agent = build_agent()
    print(f"\nAgent compiled in {time.time() - t0:.1f}s\n")

    log = open(OUT_PATH, "w", encoding="utf-8")
    log.write(f"# E1 RE-PROBE — {datetime.now(timezone.utc).isoformat()}\n\n")

    pass_count = 0
    for p in PROBES:
        log.write(f"\n{'=' * 70}\n")
        log.write(f"{p['id']}\n")
        log.write(f"  Query   : {p['query']!r}\n")
        log.write(f"  Expect  : {p['expect']}\n")

        thread = f"reprobe-{p['id']}-{int(time.time())}"
        drain_events(thread)  # clear

        print(f"\n>>> {p['id']}: {p['query']}")
        cfg = {"configurable": {"thread_id": thread}}
        t0 = time.time()
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": p["query"]}]},
                config=cfg,
            )
            dur = time.time() - t0
            messages = result.get("messages", [])
            intent = result.get("intent")
            final = _final_text(messages)
            err = None
        except Exception as e:
            dur = time.time() - t0
            messages = []
            intent = None
            final = ""
            err = repr(e)

        # Drain ALL tool events for this thread (includes researcher's internal calls)
        events = drain_events(thread)

        log.write(f"  Duration: {dur:.1f}s  intent={intent}  err={err}\n")
        log.write(f"  Tool events ({len(events)}):\n")
        for e in events:
            phase = e.get("phase")
            name = e.get("tool")
            if phase == "start":
                args = json.dumps(e.get("args", {}), default=str)
                # Don't truncate — we need to see the actual args
                log.write(f"    > {name}({args})\n")
            elif phase == "end":
                preview = (e.get("result_preview") or "")[:180].replace("\n", " ")
                ms = e.get("duration_ms", 0)
                log.write(f"    < {name}  {ms} ms  | {preview}\n")
            elif phase == "error":
                log.write(f"    ! {name}  ERROR: {e.get('error', '')[:200]}\n")
        log.write("\n  Final answer:\n")
        log.write("  " + "-" * 60 + "\n")
        for line in (final or "(empty)").split("\n"):
            log.write(f"  | {line}\n")
        log.write("  " + "-" * 60 + "\n")

        # Acceptance test
        passed = False
        if p["id"] == "killer-2-v2":
            tool_names = [e.get("tool") for e in events if e.get("phase") == "start"]
            used_find_all = "find_all_in_range" in tool_names
            mentions_268 = "268" in final or "all" in final.lower() and len(final) > 500
            passed = used_find_all and (mentions_268 or "268" in (final or ""))
            log.write(f"  Acceptance: find_all_in_range called={used_find_all}, "
                      f"answer mentions full set={('268' in (final or ''))}\n")
        elif p["id"] == "exact-1-v2":
            mentions_muapi = "muapi.ai" in (final or "").lower() or "muapi" in (final or "").lower()
            passed = mentions_muapi
            log.write(f"  Acceptance: final mentions muapi={mentions_muapi}\n")
        log.write(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
        log.flush()

        print(f"    duration={dur:.1f}s  intent={intent}  events={len(events)}  "
              f"result={'PASS' if passed else 'FAIL'}")
        for e in events:
            if e.get("phase") == "start":
                args = json.dumps(e.get("args", {}), default=str)[:120]
                print(f"      > {e.get('tool')}({args})")
        if passed:
            pass_count += 1

    log.write(f"\n\n{'=' * 70}\nRESULT: {pass_count}/{len(PROBES)} PASS\n")
    log.close()
    print(f"\nRESULT: {pass_count}/{len(PROBES)} PASS")
    print(f"Full log: {OUT_PATH}")


if __name__ == "__main__":
    main()
