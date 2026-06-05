"""Small long-term preference memory for the OpenMark agent.

This is deliberately narrow. Neo4j remains the long-term knowledge store for
bookmarks and research content. This module only stores stable user preferences
that the user explicitly asks OpenMark to remember, such as default language,
preferred newsletter format, or citation style.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool


PREF_NAMESPACE = ("openmark", "ahmad", "preferences")
_KEY_RE = re.compile(r"^[a-zA-Z0-9_.:-]{2,64}$")
_STORE = None


def _memory_enabled() -> bool:
    return os.getenv("OPENMARK_AGENT_MEMORY", "1").strip().lower() not in {"0", "false", "no"}


def _store_path() -> str:
    return os.path.normpath(
        os.getenv(
            "OPENMARK_AGENT_MEMORY_DB",
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "openmark_agent_memory.db"),
        )
    )


def get_store():
    """Return the process-wide LangGraph store for cross-thread preferences."""
    global _STORE
    if _STORE is not None:
        return _STORE

    if not _memory_enabled():
        return None

    try:
        from langgraph.store.sqlite import SqliteStore

        db_path = _store_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        store = SqliteStore(conn)
        store.setup()
        _STORE = store
        return _STORE
    except Exception:
        return None


def list_preferences(limit: int = 20) -> dict[str, str]:
    """Load remembered preferences as a flat key to value mapping."""
    store = get_store()
    if store is None:
        return {}
    try:
        items = store.search(PREF_NAMESPACE, limit=limit)
    except Exception:
        return {}

    prefs: dict[str, str] = {}
    for item in items:
        key = getattr(item, "key", "")
        value = getattr(item, "value", {}) or {}
        if key and isinstance(value, dict):
            pref = value.get("value")
            if pref is not None:
                prefs[str(key)] = str(pref)
    return dict(sorted(prefs.items()))


@tool
def remember_preference(key: str, value: str) -> str:
    """Remember an explicit stable user preference for future chats.

    Use only when the user clearly asks OpenMark to remember a preference, for
    example "remember that my default newsletter format is analytical". Do not
    infer preferences from ordinary chat. Do not store secrets, tokens, cookies,
    passwords, API keys, or private credentials.

    Args:
        key: Short stable key like default_language or newsletter_format.
        value: Preference value to remember, without secrets or credentials.
    """
    key = (key or "").strip()
    value = (value or "").strip()
    if not _memory_enabled():
        return "Preference memory is disabled by OPENMARK_AGENT_MEMORY=0."
    if not _KEY_RE.match(key):
        return "BLOCKED: key must be 2-64 chars using letters, numbers, dot, colon, underscore, or hyphen."
    if not value:
        return "BLOCKED: value is empty."
    if len(value) > 500:
        return "BLOCKED: value is too long for preference memory."
    if re.search(r"(api[_-]?key|token|cookie|password|secret|li_at|jsessionid)", key + " " + value, re.I):
        return "BLOCKED: preference memory must not store secrets, tokens, cookies, or credentials."

    store = get_store()
    if store is None:
        return "Preference memory store is unavailable."
    store.put(
        PREF_NAMESPACE,
        key,
        {"value": value, "updated_at": time.time(), "source": "user_explicit"},
        index=False,
    )
    return f"Remembered preference {key}={value!r}."


class PreferenceMemoryMiddleware(AgentMiddleware):
    """Inject stable remembered preferences into the system prompt."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        prefs = list_preferences()
        if not prefs:
            return handler(request)

        lines = [f"- {k}: {v}" for k, v in prefs.items()]
        addendum = (
            "\n\n## Remembered User Preferences\n\n"
            + "\n".join(lines)
            + "\n\nThese are stable preferences the user explicitly asked OpenMark to remember. "
              "Use them when relevant, but user instructions in the current turn override them.\n"
        )

        sys_msg = request.system_message
        if sys_msg is None:
            new_system_message = SystemMessage(content=addendum)
        else:
            existing = sys_msg.content
            if isinstance(existing, str):
                new_system_message = SystemMessage(content=existing + addendum)
            else:
                blocks = list(existing or []) + [{"type": "text", "text": addendum}]
                new_system_message = SystemMessage(content=blocks)
        return handler(request.override(system_message=new_system_message))
