"""
Skill loader for the in-app Gradio agent.

A skill is a .claude/skills/openmark-<name>/SKILL.md file with YAML frontmatter:

    ---
    name: openmark-fast-search
    description: ...
    metadata:
      type: search
    ---

    # body markdown

list_skills()       → [{name, description, type}, ...]   for the slash-command dropdown
load_skill(name)    → full text of SKILL.md body         for system-prompt injection
parse_slash(text)   → (skill_name|None, remaining_text)  for /skill-name routing
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

SKILLS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".claude", "skills")
)
# The composer agent uses three skill families:
#   openmark-*        — OpenMark-curated retrieval / composition recipes
#   humanizer-*       — humanizer-semitic (Arabic + Hebrew) drop-ins
#   agent-generated-* — skills the agent writes itself via the write_skill tool
# Anything outside these prefixes belongs to Claude Code or another plugin
# and must be ignored here.
SKILL_PREFIXES: tuple[str, ...] = ("openmark-", "humanizer-", "agent-generated-")
SKILL_PREFIX = "openmark-"  # kept for backward-compat with slash dispatch

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)
_FIELD_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.+?)\s*$", re.MULTILINE)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body). Frontmatter parser is intentionally tiny —
    we only need `name`, `description`, and `metadata.type` for the dropdown."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    head, body = m.group(1), m.group(2)
    fm: dict = {}
    # Flatten nested metadata: lines like "  type: search" are metadata.<key>
    in_metadata = False
    for raw in head.splitlines():
        if raw.strip() == "metadata:":
            in_metadata = True
            fm["metadata"] = {}
            continue
        if in_metadata and raw.startswith(" "):
            k, _, v = raw.strip().partition(":")
            fm["metadata"][k.strip()] = v.strip()
            continue
        in_metadata = False
        fm_match = _FIELD_RE.match(raw)
        if fm_match:
            fm[fm_match.group(1)] = fm_match.group(2)
    return fm, body


def _strip_known_prefix(entry: str) -> str:
    """Strip the FIRST matching skill prefix; preserves the rest verbatim."""
    for p in SKILL_PREFIXES:
        if entry.startswith(p):
            return entry[len(p):]
    return entry


@lru_cache(maxsize=1)
def _scan() -> list[dict]:
    """Discover skills with one of SKILL_PREFIXES on disk. Cached for the process lifetime."""
    out: list[dict] = []
    if not os.path.isdir(SKILLS_DIR):
        return out
    for entry in sorted(os.listdir(SKILLS_DIR)):
        if not any(entry.startswith(p) for p in SKILL_PREFIXES):
            continue
        skill_md = os.path.join(SKILLS_DIR, entry, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        try:
            with open(skill_md, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue
        fm, body = _parse_frontmatter(text)
        out.append({
            "name": fm.get("name", entry),
            # short_name strips the family prefix so the slash + UI labels stay clean.
            "short_name": _strip_known_prefix(entry),
            "description": fm.get("description", "")[:240],
            "type": (fm.get("metadata") or {}).get("type", "skill"),
            "body": body.strip(),
            "path": skill_md,
            "family": next((p.rstrip("-") for p in SKILL_PREFIXES if entry.startswith(p)), "skill"),
        })
    return out


def list_skills() -> list[dict]:
    """All openmark-* skills, fresh each call (cache invalidated by reload_skills)."""
    return _scan()


def reload_skills() -> None:
    """Drop the cache so newly-added SKILL.md files appear without restart."""
    _scan.cache_clear()


def load_skill(name: str) -> Optional[dict]:
    """
    Find a skill by full name (`openmark-fast-search`), short name (`fast-search`),
    or even just the suffix (`fast`). Returns the skill dict or None.
    """
    if not name:
        return None
    name = name.strip().lower().lstrip("/")
    skills = list_skills()
    # Exact full match first
    for s in skills:
        if s["name"].lower() == name:
            return s
    # Short name match
    for s in skills:
        if s["short_name"].lower() == name:
            return s
    # Prefix match (fast → openmark-fast-search)
    for s in skills:
        if s["short_name"].lower().startswith(name):
            return s
    return None


_SLASH_RE = re.compile(r"^/([A-Za-z0-9_-]+)\s*", re.IGNORECASE)


def parse_slash(text: str) -> tuple[Optional[str], str]:
    """
    If `text` starts with `/<name>`, return (matched_skill_name, remaining_text).
    Otherwise return (None, text) unchanged.

    Examples:
        "/fast-search RAG tools"      → ("fast-search", "RAG tools")
        "/openmark-newsletter agents" → ("openmark-newsletter", "agents")
        "what did I save"             → (None, "what did I save")
    """
    if not text:
        return None, text
    m = _SLASH_RE.match(text.strip())
    if not m:
        return None, text
    return m.group(1), text.strip()[m.end():].strip()


def autocomplete_choices() -> list[tuple[str, str]]:
    """
    For UI dropdowns: list of (display_label, slash_command_to_insert).
    Display: "fast-search — Quick one-shot lookup over Ahmad's OpenMark…"
    Insert: "/fast-search "  (with trailing space so user can type the query)
    """
    out = []
    for s in list_skills():
        desc = s["description"][:90] + ("…" if len(s["description"]) > 90 else "")
        out.append((f"/{s['short_name']} — {desc}", f"/{s['short_name']} "))
    return out
