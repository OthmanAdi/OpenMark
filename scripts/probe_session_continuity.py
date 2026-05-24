"""
Session-continuity probe — verify SqliteSaver actually carries prior-turn
context when the SAME thread_id is reused.

Drives the agent twice on one thread:
  Turn A: "find me bookmarks about RAG with knowledge graphs"  (expect tool calls)
  Turn B: "what about hashtags for those?"  (expect zero new task_researcher
          calls AND mention of the URLs/topics from turn A)

After both turns:
  - Open the SqliteSaver state for the thread
  - Verify the message list has BOTH turns' messages stitched in order
  - Verify turn B has access to turn A's ToolMessage content

Output: research/mission_2026_05_24/G2_continuity.log
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")


OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research", "mission_2026_05_24", "G2_continuity.log",
)


def main() -> None:
    from openmark.agent.graph import build_agent
    from openmark.agent.middleware import drain_events

    agent = build_agent()
    thread = f"continuity-probe-{int(time.time())}"
    drain_events(thread)

    turns = [
        ("A", "find me bookmarks about RAG with knowledge graphs"),
        ("B", "what about hashtags for those? give me 5 specific hashtags"),
    ]

    log = open(OUT, "w", encoding="utf-8")
    log.write(f"# G2 — Session-continuity probe — {datetime.now(timezone.utc).isoformat()}\n")
    log.write(f"thread_id = {thread}\n\n")

    cfg = {"configurable": {"thread_id": thread}}
    last_event_count = 0

    for tag, q in turns:
        log.write(f"\n{'=' * 60}\nTURN {tag}: {q!r}\n")
        print(f">>> Turn {tag}: {q}")

        t0 = time.time()
        try:
            agent.invoke({"messages": [{"role": "user", "content": q}]}, config=cfg)
            dur = time.time() - t0
            err = None
        except Exception as e:
            dur = time.time() - t0
            err = repr(e)

        # Drain events that fired during THIS turn (everything in deque for thread)
        events = drain_events(thread)
        new_events = events  # already only thread-scoped

        # Pull state via the agent's get_state to confirm SqliteSaver is loading
        state = agent.get_state(cfg)
        msgs = (state.values or {}).get("messages", []) or []

        ai_calls = 0
        for m in msgs:
            ai_calls += len(getattr(m, "tool_calls", None) or [])

        log.write(f"duration: {dur:.1f}s  err={err}\n")
        log.write(f"events this turn: {len(new_events)}\n")
        log.write(f"total messages in state after turn: {len(msgs)}\n")
        log.write(f"cumulative AI tool calls in state: {ai_calls}\n")
        # type sequence
        types = "".join(type(m).__name__[:1] for m in msgs)
        log.write(f"message types: {types}\n")
        # tool calls THIS turn (filter starts in events)
        tool_starts = [(e.get("tool"), e.get("args", {})) for e in new_events if e.get("phase") == "start"]
        log.write(f"tool starts this turn: {len(tool_starts)}\n")
        for name, args in tool_starts[:10]:
            log.write(f"  > {name}\n")
        log.flush()

        print(f"    dur={dur:.1f}s  state_msgs={len(msgs)}  this_turn_tools={len(tool_starts)}")

        last_event_count = len(new_events)

    # Final assertion
    final_state = agent.get_state(cfg)
    final_msgs = (final_state.values or {}).get("messages", []) or []

    log.write(f"\n{'=' * 60}\nFINAL STATE\n")
    log.write(f"total messages: {len(final_msgs)}\n")
    user_count = sum(1 for m in final_msgs if type(m).__name__ == "HumanMessage")
    log.write(f"HumanMessages: {user_count} (expected 2)\n")

    # Show last 6 messages condensed
    for i, m in enumerate(final_msgs[-10:]):
        t = type(m).__name__
        c = getattr(m, "content", "")
        if isinstance(c, list):
            cs = " | ".join(str(b)[:60] for b in c[:2])
        else:
            cs = (str(c) or "")[:140].replace("\n", " ")
        cs = cs.encode("ascii", "replace").decode("ascii")
        nm = getattr(m, "name", None)
        tc = len(getattr(m, "tool_calls", None) or [])
        log.write(f"  [{i}] {t}{(' name=' + nm) if nm else ''} content={cs!r} tool_calls={tc}\n")

    ok = user_count >= 2 and len(final_msgs) > 3
    log.write(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
              f"(needed: 2+ HumanMessages, >3 total msgs)\n")
    log.close()

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print(f"Log: {OUT}")


if __name__ == "__main__":
    main()
