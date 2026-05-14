"""Sanity-test the SQLite chat history module."""
import sys
sys.path.insert(0, r"C:\Users\oasrvadmin\Documents\OpenMark")
sys.stdout.reconfigure(encoding="utf-8")

from openmark import history

history.init_db()

# Create a session, write 2 messages, list, fetch, delete.
sid = history.create_session(title="smoke test session")
print(f"Created session id={sid}")

history.append_message(sid, "user", "what did I save about RAG?")
history.append_message(
    sid, "assistant",
    "Here are 3 RAG bookmarks ...",
    thinking="step 1: searched semantic. step 2: ranked. step 3: rendered.",
    tool_calls=[{"tool": "search_semantic", "args": {"query": "RAG"}, "duration_ms": 230}],
)

# Auto-title test
title = history.auto_title("/newsletter on RAG patterns")
print(f"Auto-title: {title!r}")
history.update_session_title(sid, title)

# List sessions
print("\nSessions:")
for s in history.list_sessions(limit=5):
    print(" ", history.session_label(s))

# Fetch messages back
print(f"\nMessages in session #{sid}:")
for m in history.get_messages(sid):
    print(f"  [{m['role']}] {m['content'][:60]}")
    if m["thinking"]:
        print(f"    thinking: {m['thinking'][:60]}")
    if m["tool_calls"]:
        print(f"    {len(m['tool_calls'])} tool call(s)")

# Cleanup
history.delete_session(sid)
print(f"\nDeleted session #{sid}; remaining sessions: {len(history.list_sessions())}")
