"""
OpenMark Gradio UI — 3 tabs:
  1. Search  — instant semantic search with rich card results
  2. Chat    — talk to the LangGraph agent (requires Azure API key)
  3. Stats   — knowledge base overview
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr
import json as _json
import base64 as _b64
from openmark import config
from openmark import history as _history

# Chat history DB — survives refresh, restart, and crashes. Persists every
# message exactly as the user saw it.
_history.init_db()

print("Loading OpenMark...")

# ── Load embedder + Neo4j ────────────────────────────────────────────────────
_embedder = None
_embedder_error = None
try:
    from openmark.embeddings.factory import get_embedder
    from openmark.stores import neo4j_store
    _embedder = get_embedder()
    print(f"Embedder ready ({config.EMBEDDING_PROVIDER} / {config.pplx_dimension()}-dim).")
except Exception as e:
    _embedder_error = str(e)
    print(f"Embedder failed: {e}")

# ── Load agent (needs Azure key — optional) ───────────────────────────────────
_agent = None
_agent_error = None
try:
    from openmark.agent.graph import build_agent, ask
    _agent = build_agent()
    print(f"Agent ready ({config.AZURE_DEPLOYMENT_LLM}).")
except Exception as e:
    _agent_error = str(e)
    print(f"Agent unavailable (no API key?): {e}")

print("OpenMark UI starting...")


# ── Category colour map ───────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Agent Development":         "#6366f1",
    "RAG & Vector Search":       "#8b5cf6",
    "LangChain / LangGraph":     "#a855f7",
    "MCP & Tool Use":            "#ec4899",
    "Context Engineering":       "#f43f5e",
    "LLM Fine-tuning":           "#ef4444",
    "AI Tools & Platforms":      "#f97316",
    "GitHub Repos & OSS":        "#22c55e",
    "Learning & Courses":        "#10b981",
    "YouTube & Video":           "#e11d48",
    "Web Development":           "#3b82f6",
    "Cloud & Infrastructure":    "#0ea5e9",
    "Data Science & ML":         "#06b6d4",
    "Knowledge Graphs & Neo4j":  "#14b8a6",
    "Career & Jobs":             "#84cc16",
    "Finance & Crypto":          "#eab308",
    "Design & UI/UX":            "#f59e0b",
    "News & Articles":           "#6b7280",
    "Entertainment & Other":     "#9ca3af",
}

SOURCE_ICONS = {
    "raindrop":  "🌧",
    "linkedin":  "💼",
    "youtube":   "▶",
}


def _similarity_bar(sim: float) -> str:
    """Visual percentage bar for similarity score."""
    pct = int(sim * 100)
    color = "#22c55e" if pct >= 75 else "#f97316" if pct >= 50 else "#6b7280"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px">'
        f'<span style="display:inline-block;width:80px;height:6px;background:#1e293b;border-radius:3px;overflow:hidden">'
        f'<span style="display:block;width:{pct}%;height:100%;background:{color};border-radius:3px"></span></span>'
        f'<span style="font-size:0.82em;color:{color};font-weight:600">{pct}%</span>'
        f'</span>'
    )


def _result_card(r: dict) -> str:
    """Render a single result as an HTML card."""
    cat   = r.get("category", "")
    color = CATEGORY_COLORS.get(cat, "#6b7280")
    src   = r.get("source", "")
    icon  = SOURCE_ICONS.get(src, "🔖")
    title = r.get("title") or r.get("url", "Untitled")
    url   = r.get("url", "")
    tags  = [t for t in r.get("tags", []) if t]
    score = int(r.get("score", 0))
    sim   = r.get("similarity", 0)

    tag_chips = " ".join(
        f'<span style="background:#334155;color:#cbd5e1;padding:3px 10px;'
        f'border-radius:9999px;font-size:0.8em;margin-right:4px">{t}</span>'
        for t in tags[:6]
    )

    stars = "★" * min(score, 5) if score else ""

    return f"""
<div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
            padding:16px 20px;margin-bottom:12px;font-family:sans-serif">

  <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:8px">
    <span style="font-size:1.2em;margin-top:1px">{icon}</span>
    <a href="{url}" target="_blank"
       style="color:#f1f5f9;font-weight:700;font-size:1.05em;
              text-decoration:none;flex:1;line-height:1.4">
      {r['rank']}. {title}
    </a>
    <span style="background:{color};color:#fff;padding:3px 12px;
                 border-radius:9999px;font-size:0.78em;white-space:nowrap;font-weight:500">
      {cat}
    </span>
  </div>

  <div style="font-size:0.85em;margin-bottom:10px;word-break:break-all">
    <a href="{url}" target="_blank" style="color:#7dd3fc;text-decoration:none">{url[:100]}{"…" if len(url)>100 else ""}</a>
  </div>

  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <div>{_similarity_bar(sim)}</div>
    {"<span style='color:#fbbf24;font-size:0.9em;letter-spacing:2px'>" + stars + "</span>" if stars else ""}
    <div style="flex:1">{tag_chips}</div>
  </div>
</div>
"""


# ── Search ────────────────────────────────────────────────────────────────────

def search_fn(query: str, category: str, min_score: float, n_results: int):
    if _embedder is None:
        return f"<p style='color:#ef4444'>Embedder not loaded: {_embedder_error}</p>"
    if not query.strip():
        return "<p style='color:#94a3b8'>Enter a query above and hit Search.</p>"

    cat = category if category != "All" else None
    ms  = float(min_score) if min_score > 0 else None

    try:
        q_embed = _embedder.embed_query(query)
        results = neo4j_store.vector_search(q_embed, n=int(n_results), category=cat)
        if ms is not None:
            results = [r for r in results if float(r.get("bm_score", 0)) >= ms]
        # Add rank + remap bm_score → score for card renderer
        for i, r in enumerate(results, 1):
            r["rank"] = i
            r["score"] = r.get("bm_score", 0)
    except Exception as e:
        return f"<p style='color:#ef4444'>Search error: {e}</p>"

    if not results:
        return "<p style='color:#f97316'>No results found. Try a broader query or remove filters.</p>"

    header = (
        f'<div style="color:#64748b;font-size:0.82em;margin-bottom:10px">'
        f'Found <strong style="color:#e2e8f0">{len(results)}</strong> results '
        f'for "<strong style="color:#e2e8f0">{query}</strong>"'
        + (f' · Category: <strong style="color:#e2e8f0">{cat}</strong>' if cat else "")
        + f'</div>'
    )
    cards = "".join(_result_card(r) for r in results)
    return f"<div style='overflow-y:auto;max-height:650px;padding-right:4px'>{header + cards}</div>"


# ── Chat ──────────────────────────────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    """One-line preview of tool args for the live card."""
    if not args:
        return ""
    pairs = []
    for k, v in list(args.items())[:3]:
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "…"
        pairs.append(f"{k}={s!r}" if isinstance(v, str) else f"{k}={s}")
    return ", ".join(pairs)


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tool_card(ev: dict) -> str:
    """Render one tool event as a chat card. End cards show the FULL tool output
    in an expandable <details> block so the user can see every result, not a
    truncated crumb."""
    name = ev.get("tool", "?")
    args = _fmt_args(ev.get("args", {}))
    phase = ev.get("phase")
    if phase == "start":
        return (f"<div style='background:#0f172a;border-left:3px solid #6366f1;"
                f"padding:6px 10px;margin:6px 0;border-radius:4px;font-size:0.85em;color:#94a3b8'>"
                f"🔧 <b style='color:#a5b4fc'>{name}</b>({_esc(args)}) "
                f"<i style='color:#475569'>running…</i></div>")
    if phase == "end":
        dur = ev.get("duration_ms", 0)
        full = ev.get("result_preview", "") or ""
        n_chars = len(full)
        # First 1500 chars rendered inline; full content in a <details> if longer.
        head = full[:1500]
        tail = full[1500:] if n_chars > 1500 else ""
        head_html = _esc(head).replace("\n", "<br>")
        body = (
            f"<div style='color:#cbd5e1;font-size:0.85em;margin-top:4px;"
            f"max-height:280px;overflow:auto;background:#0b1220;padding:6px 8px;"
            f"border-radius:4px;font-family:ui-monospace,Menlo,monospace;line-height:1.45'>"
            f"{head_html}"
            f"</div>"
        )
        if tail:
            body += (
                f"<details style='margin-top:4px;font-size:0.78em;color:#64748b'>"
                f"<summary style='cursor:pointer'>show {n_chars - 1500} more chars</summary>"
                f"<pre style='white-space:pre-wrap;background:#0b1220;padding:6px 8px;"
                f"border-radius:4px;max-height:420px;overflow:auto;color:#cbd5e1'>"
                f"{_esc(tail)}</pre></details>"
            )
        return (
            f"<div style='background:#0f172a;border-left:3px solid #22c55e;"
            f"padding:6px 10px;margin:6px 0;border-radius:4px;font-size:0.85em'>"
            f"<div style='color:#86efac'>✓ <b>{name}</b>({_esc(args)}) "
            f"<span style='color:#64748b'>· {dur} ms · {n_chars:,} chars</span></div>"
            f"{body}"
            f"</div>"
        )
    if phase == "error":
        err = ev.get("error", "")
        return (f"<div style='background:#0f172a;border-left:3px solid #ef4444;"
                f"padding:6px 10px;margin:6px 0;border-radius:4px;font-size:0.85em;color:#fca5a5'>"
                f"✗ <b>{name}</b>({_esc(args)}) · {_esc(err)}</div>")
    return ""


DRAFTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "drafts")
)
os.makedirs(DRAFTS_DIR, exist_ok=True)


def _slugify(s: str, n: int = 50) -> str:
    """Lowercase, alphanumeric + hyphen only, capped at n chars."""
    import re as _re
    s = _re.sub(r"[^a-zA-Z0-9]+", "-", (s or "").lower()).strip("-")
    return (s or "report")[:n]


_URL_RE = __import__("re").compile(r"https?://[^\s\)\]'\"<>]+", __import__("re").IGNORECASE)


def _looks_like_report(md: str) -> bool:
    """
    Heuristic — fires when any of these is true:
      (a) Starts with `# H1` AND contains a Sources / Citations / TL;DR marker.
      (b) Starts with `# H1` AND has 4+ URLs (any newsletter-shaped doc).
      (c) Has 5+ URLs AND >= 500 chars (numbered recommendation list or
          'Sources cited above:' block — even when the agent forgets the H1).
      (d) Contains an explicit 'Sources cited above:' line — matches the
          openmark-newsletter skill template footer.
    Anything that fires (a)-(d) is worth saving as a draft so Ahmad can
    re-read or edit it later.
    """
    if not md:
        return False
    stripped = md.lstrip()
    low = md.lower()
    n_urls = len(_URL_RE.findall(md))
    starts_h1 = stripped.startswith("# ")
    markers = (
        "## sources", "## citations", "**tl;dr.**", "**tl;dr**", "## tl;dr",
        "## what i'm reading", "## what i am reading",
        "sources cited above", "sources cited:",
    )
    has_marker = any(t in low for t in markers)

    if starts_h1 and has_marker:
        return True                            # (a)
    if starts_h1 and n_urls >= 4:
        return True                            # (b)
    if n_urls >= 5 and len(md) >= 500:
        return True                            # (c)
    if has_marker and n_urls >= 3:
        return True                            # (d)
    return False


def _maybe_export_report(md: str, first_line_title: str = "") -> str:
    """If the agent's text response looks like a Report, save to drafts/ and
    append a download footer. Otherwise return the markdown unchanged."""
    if not _looks_like_report(md):
        return md
    # Title = first H1 line, slugified
    title = first_line_title
    if not title:
        for line in md.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    import time as _time
    ts = _time.strftime("%Y-%m-%d-%H%M")
    slug = _slugify(title or "report", 40)
    fname = f"{ts}-{slug}.md"
    path = os.path.join(DRAFTS_DIR, fname)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception as e:
        print(f"[drafts] failed to save: {e}")
        return md
    url_path = path.replace("\\", "/")
    return md + (
        f"\n\n---\n\n"
        f"<div style='background:#0f172a;border:1px solid #1e293b;"
        f"border-radius:8px;padding:10px 14px;margin-top:10px;font-family:sans-serif'>"
        f"<span style='color:#94a3b8;font-size:0.85em'>📥 Saved to </span>"
        f"<code style='background:#1e293b;color:#a5b4fc;padding:2px 6px;"
        f"border-radius:4px;font-size:0.85em'>drafts/{fname}</code>"
        f" &nbsp;·&nbsp; "
        f"<a href='/gradio_api/file={url_path}' target='_blank' "
        f"style='color:#7dd3fc;text-decoration:none;font-weight:600'>"
        f"⬇ Download Markdown</a>"
        f"</div>"
    )


def chat_fn(message: str, history: list, session_id: int | None):
    """
    Streaming generator for gr.ChatInterface.

    Returns assistant markdown chunks AND persists the full exchange to SQLite
    so the chat survives refresh / restart. The session_id state is created
    lazily on first message of a fresh session.
    """
    if _agent is None:
        yield (
            "**Agent unavailable** — Azure API key not configured in `.env`.\n\n"
            "Use the **Search** tab — it works fully offline with local pplx-embed.\n\n"
            f"*Error: {_agent_error}*"
        )
        return

    from openmark.agent.graph import ask_stream

    # Ensure we have a session — create one on the first message.
    if not session_id:
        session_id = _history.create_session(title=_history.auto_title(message))
    # Persist the user message NOW so a mid-stream refresh still shows it.
    _history.append_message(session_id, "user", message)

    # Thread the agent off the session id so MemorySaver groups turns correctly.
    thread_id = f"sess-{session_id}"

    parts: list[str] = []
    thinking_text = ""        # batched fallback (only if per-turn didn't stream)
    n_calls = 0
    turn_counter = 0          # one inline thought bubble per AIMessage emitted
    tool_events: list[dict] = []
    per_turn_thinking: list[str] = []

    try:
        for ev in ask_stream(_agent, message, thread_id=thread_id):
            kind = ev.get("kind")
            if kind == "user":
                continue
            elif kind == "turn_thinking":
                # Per-turn reasoning lands BEFORE the tool calls for that turn.
                # Render as a collapsed bubble so the chat stays scannable.
                turn_counter += 1
                t = (ev.get("text", "") or "").strip()
                if t:
                    per_turn_thinking.append(t)
                    parts.append(
                        f"<details style='background:#0f172a;border-left:3px solid #a855f7;"
                        f"padding:6px 10px;margin:6px 0;border-radius:4px;font-size:0.85em'>"
                        f"<summary style='color:#c4b5fd;cursor:pointer'>"
                        f"🧠 Thinking · turn {turn_counter} · {len(t):,} chars</summary>"
                        f"<pre style='white-space:pre-wrap;background:#0b1220;padding:6px 8px;"
                        f"border-radius:4px;max-height:340px;overflow:auto;color:#cbd5e1;"
                        f"margin-top:4px;font-size:0.92em;line-height:1.45'>{_esc(t)}</pre>"
                        f"</details>"
                    )
            elif kind == "tool_start":
                parts.append(_tool_card({"phase": "start", "tool": ev["tool"], "args": ev.get("args", {})}))
                n_calls += 1
                tool_events.append({"phase": "start", "tool": ev["tool"], "args": ev.get("args", {})})
            elif kind == "tool_end":
                parts.append(_tool_card({
                    "phase": "end",
                    "tool": ev["tool"],
                    "args": {},  # already shown on the start card
                    "duration_ms": ev.get("duration_ms"),
                    "result_preview": ev.get("preview", ""),
                }))
                tool_events.append({"phase": "end", "tool": ev["tool"],
                                    "duration_ms": ev.get("duration_ms"),
                                    "preview_len": len(ev.get("preview", "") or "")})
            elif kind == "tool_error":
                parts.append(_tool_card({
                    "phase": "error",
                    "tool": ev.get("tool", "?"),
                    "error": ev.get("error", ""),
                }))
                tool_events.append({"phase": "error", "tool": ev.get("tool", "?"),
                                    "error": ev.get("error", "")})
            elif kind == "thinking":
                thinking_text = ev.get("text", "")
            elif kind == "final":
                raw = (ev.get("text", "") or "").strip()
                if not raw:
                    final = "_(no response)_"
                else:
                    # Auto-export when the agent's markdown looks like a Report.
                    final = _maybe_export_report(raw)
                # Fallback: only show batched thinking if per-turn didn't stream.
                if thinking_text:
                    parts.append(
                        "<details><summary>🧠 <b>Thinking (full trace)</b> · "
                        f"{n_calls} tool call(s) · codex 5.3 reasoning=high</summary>\n\n"
                        f"```\n{thinking_text}\n```\n\n</details>"
                    )
                parts.append(final)
            yield "\n".join(parts)
    except Exception as e:
        parts.append(f"\n\n❌ **Agent error:** `{e}`")
        yield "\n".join(parts)
    finally:
        # Persist the assistant message exactly as the user saw it, plus the
        # thinking trace and tool events for full replay fidelity.
        try:
            assistant_md = "\n".join(parts) if parts else "_(no response)_"
            full_thinking = (
                "\n\n---\n\n".join(per_turn_thinking)
                if per_turn_thinking else thinking_text
            )
            _history.append_message(
                session_id, "assistant", assistant_md,
                thinking=full_thinking, tool_calls=tool_events,
            )
        except Exception as e:
            print(f"[history] failed to persist assistant message: {e}")


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats_fn():
    if _embedder is None:
        return f"<p style='color:#ef4444'>Embedder not loaded.</p>"

    try:
        neo4j  = neo4j_store.get_stats()
        cat_rows = neo4j_store.query("""
            MATCH (b:Bookmark)-[:IN_CATEGORY]->(c:Category)
            RETURN c.name AS category, count(b) AS count
            ORDER BY count DESC
        """)
        tag_rows = neo4j_store.query("""
            MATCH (b:Bookmark)-[:TAGGED]->(t:Tag)
            RETURN t.name AS tag, count(b) AS count
            ORDER BY count DESC LIMIT 20
        """)
    except Exception as e:
        return f"<p style='color:#f97316'>Neo4j not available: {e}</p>"

    return f"""
<div style="font-family:sans-serif;color:#e2e8f0">

  <div style="margin-bottom:16px">
    <div style="color:#94a3b8;font-size:0.85em;margin-bottom:8px">KNOWLEDGE GRAPH</div>
    <div style="display:flex;gap:20px;flex-wrap:wrap">
      {"".join(
          f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px 20px;text-align:center">'
          f'<div style="font-size:1.6em;font-weight:700;color:#e2e8f0">{v:,}</div>'
          f'<div style="font-size:0.75em;color:#64748b">{k}</div></div>'
          for k, v in [
              ("Bookmarks", neo4j.get("bookmarks", 0)),
              ("Tags",      neo4j.get("tags", 0)),
              ("Categories", neo4j.get("categories", 0)),
              ("Communities", neo4j.get("communities", 0)),
          ]
      )}
    </div>
  </div>

  <div style="margin-top:20px">
    <div style="color:#94a3b8;font-size:0.85em;margin-bottom:8px">BY CATEGORY</div>
    {"".join(
      f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
      f'<span style="background:{CATEGORY_COLORS.get(r["category"],"#6b7280")};width:10px;height:10px;border-radius:50%;display:inline-block"></span>'
      f'<span style="color:#e2e8f0;font-size:0.85em;flex:1">{r["category"]}</span>'
      f'<span style="color:#94a3b8;font-size:0.85em">{r["count"]}</span>'
      f'</div>'
      for r in cat_rows
    )}
  </div>

  <div style="margin-top:20px">
    <div style="color:#94a3b8;font-size:0.85em;margin-bottom:8px">TOP TAGS</div>
    <div style="line-height:2">
    {"".join(
      f'<span style="background:#1e293b;color:#94a3b8;padding:3px 10px;border-radius:9999px;font-size:0.78em;margin:2px">'
      f'{r["tag"]} <strong style="color:#e2e8f0">{r["count"]}</strong></span>'
      for r in tag_rows
    )}
    </div>
  </div>

</div>
"""


# ── Build UI ──────────────────────────────────────────────────────────────────

DARK_CSS = """
body, .gradio-container { background: #020617 !important; color: #e2e8f0 !important; }
.gr-button-primary { background: #6366f1 !important; border: none !important; }
.gr-button-primary:hover { background: #4f46e5 !important; }
footer { display: none !important; }
#search-list { overflow-y: auto; max-height: 680px; }
#search-graph iframe { border-radius: 12px; }
"""



def _get_graph_data(limit: int = 400, min_sim: float = 0.82) -> dict:
    """Fetch nodes + links from Neo4j for the force graph."""
    rows = neo4j_store.query("""
        MATCH (b:Bookmark)
        WHERE b.embedding IS NOT NULL
        WITH b ORDER BY b.score DESC LIMIT $limit

        OPTIONAL MATCH (b)-[:IN_CATEGORY]->(c:Category)
        OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
        OPTIONAL MATCH (b)-[r:SIMILAR_TO]-(b2:Bookmark)
        WHERE r.score >= $min_sim

        RETURN b.url AS url, b.title AS title,
               b.score AS score, b.source AS source,
               b.category AS category,
               collect(DISTINCT t.name)[..4]                  AS tags,
               collect(DISTINCT {target: b2.url, score: r.score})[..3] AS similar
    """, {"limit": limit, "min_sim": min_sim})

    nodes = {}
    links = []

    for r in rows:
        bid = "b:" + (r["url"] or "")
        cat  = r["category"] or "Entertainment & Other"
        color = CATEGORY_COLORS.get(cat, "#6b7280")
        val   = max(2, min(10, int((r["score"] or 0) * 1.0)))

        nodes[bid] = {
            "id": bid, "label": "Bookmark",
            "name": (r["title"] or r["url"] or "")[:70],
            "url": r["url"] or "",
            "category": cat, "source": r["source"] or "",
            "color": color, "val": val,
        }

        # Category nodes
        cid = "c:" + cat
        if cid not in nodes:
            nodes[cid] = {"id": cid, "label": "Category", "name": cat,
                          "color": "#f97316", "val": 14}
        links.append({"source": bid, "target": cid, "type": "CAT"})

        # Tag nodes (top 4)
        for tag in (r["tags"] or [])[:4]:
            tid = "t:" + tag
            if tid not in nodes:
                nodes[tid] = {"id": tid, "label": "Tag", "name": tag,
                              "color": "#22c55e", "val": 4}
            links.append({"source": bid, "target": tid, "type": "TAG"})

        # SIMILAR_TO edges — only if both endpoints are in nodes set
        for s in (r["similar"] or []):
            if s and s.get("target"):
                tid = "b:" + s["target"]
                if tid in nodes:
                    links.append({"source": bid, "target": tid,
                                  "type": "SIM", "val": float(s.get("score", 0))})

    return {"nodes": list(nodes.values()), "links": links}


def _get_search_graph_data(query: str, n: int = 30) -> dict:
    """Vector search → expand each result → build focused subgraph."""
    q_embed = _embedder.embed_query(query)
    results = neo4j_store.vector_search(q_embed, n=n)
    if not results:
        return {"nodes": [], "links": []}

    # Similarity score per URL (for node sizing)
    sim_scores = {r["url"]: r.get("similarity", 0) for r in results if r.get("url")}
    urls = list(sim_scores.keys())

    # Fetch tags + categories for each result (separate scalars, no nested maps)
    rows = neo4j_store.query("""
        MATCH (b:Bookmark) WHERE b.url IN $urls
        OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
        RETURN b.url AS url, b.title AS title, b.score AS score,
               b.category AS category, b.source AS source,
               collect(t.name)[..6] AS tags
    """, {"urls": urls})

    nodes = {}
    links = []

    for r in rows:
        bid  = "b:" + (r["url"] or "")
        cat  = r["category"] or "Entertainment & Other"
        sim  = sim_scores.get(r["url"] or "", 0)
        color = CATEGORY_COLORS.get(cat, "#6b7280")
        val   = max(4, min(22, int(sim * 22)))

        nodes[bid] = {
            "id": bid, "label": "Bookmark",
            "name": (r["title"] or r["url"] or "")[:80],
            "url": r["url"] or "",
            "category": cat,
            "similarity": round(sim, 3),
            "color": color, "val": val,
        }

        # Category node
        cid = "c:" + cat
        if cid not in nodes:
            nodes[cid] = {"id": cid, "label": "Category", "name": cat,
                          "color": "#f97316", "val": 12}
        links.append({"source": bid, "target": cid, "type": "CAT"})

        # Tag nodes
        for tag in (r["tags"] or [])[:5]:
            if not tag:
                continue
            tid = "t:" + tag
            if tid not in nodes:
                nodes[tid] = {"id": tid, "label": "Tag", "name": tag,
                              "color": "#22c55e", "val": 4}
            links.append({"source": bid, "target": tid, "type": "TAG"})

    # Add SIMILAR_TO edges between results that are already in the node set
    sim_rows = neo4j_store.query("""
        MATCH (a:Bookmark)-[r:SIMILAR_TO]-(b:Bookmark)
        WHERE a.url IN $urls AND b.url IN $urls AND r.score > 0.82
        RETURN a.url AS src, b.url AS tgt, r.score AS score
    """, {"urls": urls})

    seen_edges = set()
    for sr in sim_rows:
        key = tuple(sorted([sr["src"], sr["tgt"]]))
        if key not in seen_edges:
            seen_edges.add(key)
            links.append({"source": "b:" + sr["src"], "target": "b:" + sr["tgt"],
                          "type": "SIM", "val": float(sr.get("score", 0))})

    return {"nodes": list(nodes.values()), "links": links}


def graph_search_fn(query: str):
    """Render graph focused on search results."""
    if _embedder is None:
        return "<p style='color:#ef4444'>Embedder not loaded.</p>"
    if not query.strip():
        return ""
    try:
        data = _get_search_graph_data(query.strip(), n=30)
    except Exception as e:
        return f"<p style='color:#ef4444'>Graph error: {e}</p>"

    if not data["nodes"]:
        return f"<p style='color:#f97316'>No results for '{query}'.</p>"

    return _build_graph_html(data, title=f'Search: "{query}"')


def graph_fn(limit: int = 400):
    """Render full explore graph."""
    if _embedder is None:
        return "<p style='color:#ef4444'>Embedder not loaded.</p>"
    try:
        data = _get_graph_data(limit=limit)
    except Exception as e:
        return f"<p style='color:#ef4444'>Graph error: {e}</p>"
    return _build_graph_html(data, title=f"Top {limit} bookmarks by score")


def _build_graph_html(data: dict, title: str = "") -> str:
    """Shared renderer — builds the iframe HTML for any graph data."""

    data_json = _json.dumps(data)
    n_nodes = len(data["nodes"])
    n_links = len(data["links"])

    legend_items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{color}"></span>'
        f'<span style="font-size:12px;color:#94a3b8">{cat}</span></span>'
        for cat, color in list(CATEGORY_COLORS.items())[:8]
    )

    # Full HTML document in an iframe (bypasses Gradio CSP for external scripts)
    iframe_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin:0; padding:0; background:#020617; overflow:hidden; font-family:sans-serif; }}
  #graph {{ width:100vw; height:100vh; }}
  #info {{
    position:absolute; bottom:0; left:0; right:0;
    background:rgba(15,23,42,0.92); padding:8px 14px;
    font-size:13px; color:#94a3b8; min-height:32px;
    border-top:1px solid #1e293b; word-break:break-all;
  }}
  #info a {{ color:#7dd3fc; text-decoration:none; }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="info">Hover or click a node to see details.</div>
<script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1/dist/3d-force-graph.min.js"></script>
<script>
var gData = {data_json};
var linkColors = {{ CAT:'#f97316', TAG:'#22c55e', SIM:'#6366f1' }};
var Graph = ForceGraph3D()(document.getElementById('graph'))
  .backgroundColor('#020617')
  .graphData(gData)
  .nodeLabel(n => '<div style="background:#0f172a;padding:4px 8px;border-radius:6px;font-size:12px;color:#e2e8f0;max-width:280px;word-break:break-word">[' + n.label + '] ' + n.name + '</div>')
  .nodeColor(n => n.color || '#fff')
  .nodeVal(n => n.val || 3)
  .nodeOpacity(0.92)
  .linkColor(l => linkColors[l.type] || '#334155')
  .linkOpacity(0.2)
  .linkWidth(l => l.type === 'SIM' ? (l.val || 0.5) * 2 : 0.4)
  .linkDirectionalParticles(l => l.type === 'SIM' ? 2 : 0)
  .linkDirectionalParticleColor(() => '#818cf8')
  .linkDirectionalParticleWidth(1.8)
  .linkDirectionalParticleSpeed(0.004)
  .onNodeHover(node => {{
    document.getElementById('graph').style.cursor = node ? 'pointer' : 'default';
    if (node) {{
      document.getElementById('info').innerHTML =
        '<span style="color:#94a3b8">[' + node.label + ']</span> '
        + '<strong style="color:#f1f5f9">' + node.name + '</strong>'
        + (node.category ? ' <span style="background:' + (node.color||'#6b7280') + ';color:#fff;padding:2px 8px;border-radius:9999px;font-size:11px;margin-left:8px">' + node.category + '</span>' : '')
        + (node.url ? '<br><a href="' + node.url + '" target="_blank">' + node.url.slice(0,100) + '</a>' : '');
    }}
  }})
  .onNodeClick(node => {{
    if (node.url) window.open(node.url, '_blank');
    var dist = 120;
    var hypot = Math.hypot(node.x || 1, node.y || 1, node.z || 1);
    var ratio = 1 + dist / hypot;
    Graph.cameraPosition(
      {{x: node.x * ratio, y: node.y * ratio, z: node.z * ratio}},
      node, 1000
    );
  }})
  .d3AlphaDecay(0.02)
  .d3VelocityDecay(0.3);
</script>
</body>
</html>"""

    # Base64-encode the full HTML document — avoids all srcdoc quote-escaping issues
    iframe_b64 = _b64.b64encode(iframe_html.encode("utf-8")).decode("ascii")

    return f"""
<div style="font-family:sans-serif">
  <div style="display:flex;align-items:center;gap:16px;padding:6px 0 8px;flex-wrap:wrap">
    <span style="color:#64748b;font-size:0.82em">{n_nodes:,} nodes · {n_links:,} edges</span>
    <span style="color:#475569;font-size:0.75em">Click node → opens URL · Drag to rotate · Scroll to zoom</span>
    {legend_items}
  </div>
  <iframe src="data:text/html;base64,{iframe_b64}"
          style="width:100%;height:660px;border:none;border-radius:12px;background:#020617"
          sandbox="allow-scripts allow-popups">
  </iframe>
</div>
"""


def build_ui():
    categories = ["All"] + config.CATEGORIES

    with gr.Blocks(title="OpenMark") as app:

        try:
            _s = neo4j_store.get_stats() if _embedder else {}
            _bm_count = f"{_s.get('bookmarks', 0):,}" if _s else "?"
        except Exception:
            _bm_count = "?"

        gr.HTML(f"""
        <div style="padding:20px 0 10px;font-family:sans-serif">
          <h1 style="margin:0;font-size:1.6em;color:#e2e8f0">
            🔖 <strong>OpenMark</strong>
            <span style="font-weight:300;color:#64748b"> — Personal Knowledge Graph</span>
          </h1>
          <p style="margin:4px 0 0;color:#475569;font-size:0.85em">
            {_bm_count} bookmarks · pplx-embed-context-v1-0.6B · Neo4j Graph RAG
          </p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Search (default) ────────────────────────────────────
            with gr.Tab("Search"):
                with gr.Row():
                    q_input = gr.Textbox(
                        placeholder="e.g. 'LangGraph agent memory', 'fine-tuning LLMs', 'vector db comparison'...",
                        label="",
                        scale=5,
                        container=False,
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1, min_width=100)

                with gr.Row():
                    cat_input   = gr.Dropdown(categories, value="All", label="Category", scale=2)
                    score_input = gr.Slider(0, 10, value=0, step=1, label="Min Quality Score", scale=2)
                    n_input     = gr.Slider(5, 50, value=10, step=5, label="Results", scale=1)

                with gr.Row():
                    search_output = gr.HTML(
                        value="<p style='color:#475569;font-size:0.9em;padding:20px 0'>Results will appear here.</p>",
                        min_height=620,
                        elem_id="search-list",
                    )
                    search_graph_output = gr.HTML(value="", min_height=620, elem_id="search-graph")
                search_btn.click(
                    search_fn,
                    inputs=[q_input, cat_input, score_input, n_input],
                    outputs=search_output,
                    show_progress="minimal",
                )
                q_input.submit(
                    search_fn,
                    inputs=[q_input, cat_input, score_input, n_input],
                    outputs=search_output,
                    show_progress="minimal",
                )
                search_btn.click(
                    graph_search_fn, inputs=[q_input], outputs=search_graph_output,
                    show_progress="minimal",
                )
                q_input.submit(
                    graph_search_fn, inputs=[q_input], outputs=search_graph_output,
                    show_progress="minimal",
                )

            # ── Tab 2: Chat ────────────────────────────────────────────────
            with gr.Tab("Chat" + (" ✓" if _agent else " (no key)")):
                # Skill catalogue for the slash-command picker
                try:
                    from openmark.agent.skills import list_skills as _list_skills
                    _skill_rows = _list_skills()
                except Exception:
                    _skill_rows = []
                _skill_radio_choices = [
                    (f"/{s['short_name']} — {s['description'][:80]}", f"/{s['short_name']} ")
                    for s in _skill_rows
                ]

                # Helpers for the history dropdown ------------------------------
                def _session_choices() -> list[tuple[str, int]]:
                    return [(_history.session_label(s), s["id"])
                            for s in _history.list_sessions(limit=100)]

                def _load_session(sid_choice: int | None):
                    """Return chatbot messages list + the int session id, given a dropdown value."""
                    if not sid_choice:
                        return [], None
                    try:
                        sid = int(sid_choice)
                    except (TypeError, ValueError):
                        return [], None
                    rows = _history.get_messages(sid)
                    msgs = [{"role": r["role"], "content": r["content"]} for r in rows]
                    return msgs, sid

                def _new_chat():
                    """Clear chatbot + session state. Refresh the dropdown."""
                    return [], None, gr.update(choices=_session_choices(), value=None)

                def _delete_chat(sid_choice):
                    if not sid_choice:
                        return [], None, gr.update(choices=_session_choices(), value=None)
                    try:
                        _history.delete_session(int(sid_choice))
                    except Exception as e:
                        print(f"[history] delete failed: {e}")
                    return [], None, gr.update(choices=_session_choices(), value=None)

                def _refresh_sessions():
                    return gr.update(choices=_session_choices())

                # State: int session id, lazily created on first message
                session_id_state = gr.State(value=None)

                # Top bar — history controls
                with gr.Row():
                    new_chat_btn = gr.Button("✨ New chat", variant="primary", scale=1, min_width=110)
                    session_dd = gr.Dropdown(
                        choices=_session_choices(),
                        value=None,
                        label="Previous chats (persisted)",
                        info="Pick a past chat to reload it. SQLite-backed, survives refresh.",
                        scale=5,
                        interactive=True,
                    )
                    refresh_btn = gr.Button("↻", scale=0, min_width=40)
                    delete_btn = gr.Button("🗑", variant="stop", scale=0, min_width=40)

                with gr.Accordion("Skills (click to prefill /command)", open=False):
                    if _skill_rows:
                        skill_picker = gr.Radio(
                            choices=_skill_radio_choices,
                            label="",
                            value=None,
                            info="Click a skill to insert its slash command. "
                                 "Or type / at the start of any message.",
                        )
                    else:
                        gr.HTML("<i style='color:#64748b'>No skills installed yet.</i>")
                        skill_picker = None
                    gr.HTML(
                        "<div style='font-size:0.8em;color:#64748b;line-height:1.5;margin-top:6px'>"
                        "<b>Slash commands:</b> <code>/newsletter</code>, <code>/newsletter-roundup</code>, "
                        "<code>/newsletter-essay</code>, <code>/newsletter-comparison</code>, "
                        "<code>/newsletter-thread</code>, <code>/deep-research</code>, "
                        "<code>/weekly-digest</code>, <code>/bookmark-dive &lt;URL&gt;</code>, "
                        "<code>/fast-search</code>.<br>"
                        "Or just ask in plain English — the agent picks a skill itself."
                        "</div>"
                    )

                chat_ui = gr.ChatInterface(
                    fn=chat_fn,
                    additional_inputs=[session_id_state],
                    chatbot=gr.Chatbot(
                        height=620,
                        placeholder=(
                            "<div style='text-align:center;color:#475569;padding:40px'>"
                            "<div style='font-size:2em'>🔖</div>"
                            "<div style='margin-top:6px'>Ask anything about your "
                            f"{_bm_count if isinstance(_bm_count, str) else 'saved'} bookmarks</div>"
                            "<div style='font-size:0.8em;margin-top:10px;color:#64748b'>"
                            "Try: &nbsp;<i>What did I save this week?</i> &nbsp;·&nbsp; "
                            "<i>/deep-research RAG patterns</i> &nbsp;·&nbsp; "
                            "<i>/newsletter on context engineering</i> &nbsp;·&nbsp; "
                            "<i>/bookmark-dive https://...</i>"
                            "</div></div>"
                        ),
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$",  "right": "$",  "display": False},
                            {"left": "\\(", "right": "\\)", "display": False},
                            {"left": "\\[", "right": "\\]", "display": True},
                        ],
                        sanitize_html=False,
                        render_markdown=True,
                    ),
                    submit_btn="Send",
                    stop_btn="Stop",
                    fill_height=False,
                )

                # Wire history controls — load / new / delete / refresh
                session_dd.change(
                    fn=_load_session,
                    inputs=[session_dd],
                    outputs=[chat_ui.chatbot, session_id_state],
                    show_progress="hidden",
                )
                new_chat_btn.click(
                    fn=_new_chat,
                    inputs=None,
                    outputs=[chat_ui.chatbot, session_id_state, session_dd],
                    show_progress="hidden",
                )
                delete_btn.click(
                    fn=_delete_chat,
                    inputs=[session_dd],
                    outputs=[chat_ui.chatbot, session_id_state, session_dd],
                    show_progress="hidden",
                )
                refresh_btn.click(
                    fn=_refresh_sessions,
                    inputs=None,
                    outputs=[session_dd],
                    show_progress="hidden",
                )

                # After every assistant reply, refresh the dropdown so the
                # (auto-titled) current chat moves to the top of the list.
                # chat_ui exposes .chatbot for change events.
                chat_ui.chatbot.change(
                    fn=_refresh_sessions,
                    inputs=None,
                    outputs=[session_dd],
                    show_progress="hidden",
                )

                # Skill-picker prefills the chat textbox.
                if skill_picker is not None:
                    skill_picker.change(
                        fn=lambda v: v or "",
                        inputs=[skill_picker],
                        outputs=[chat_ui.textbox],
                        show_progress="hidden",
                    )

            # ── Tab 3: Stats ───────────────────────────────────────────────
            with gr.Tab("Stats"):
                refresh_btn  = gr.Button("Refresh", variant="secondary")
                stats_output = gr.HTML()
                refresh_btn.click(stats_fn, outputs=stats_output)
                app.load(stats_fn, outputs=stats_output)

            # ── Tab 4: Graph ───────────────────────────────────────────────
            with gr.Tab("Graph 3D"):
                gr.HTML("<div style='color:#64748b;font-size:0.83em;padding:4px 0 8px'>Two modes: <b style='color:#e2e8f0'>Search</b> — visualise search results as a graph &nbsp;|&nbsp; <b style='color:#e2e8f0'>Explore</b> — browse top bookmarks by score</div>")

                with gr.Tabs():

                    # Search mode — query → show those results as graph
                    with gr.Tab("Search Graph"):
                        with gr.Row():
                            gsearch_input = gr.Textbox(
                                placeholder="e.g. 'LangGraph agent memory', 'RAG tools'...",
                                label="", scale=5, container=False,
                            )
                            gsearch_btn = gr.Button("Visualise", variant="primary", scale=1, min_width=120)
                        gsearch_output = gr.HTML(
                            value="<p style='color:#475569;padding:20px'>Enter a search query to see its results as an interactive 3D graph.</p>"
                        )
                        gsearch_btn.click(graph_search_fn, inputs=[gsearch_input], outputs=gsearch_output)
                        gsearch_input.submit(graph_search_fn, inputs=[gsearch_input], outputs=gsearch_output)

                    # Explore mode — full top-N by score
                    with gr.Tab("Explore Graph"):
                        with gr.Row():
                            graph_limit = gr.Slider(100, 1000, value=400, step=100,
                                                    label="Nodes (bookmarks)", scale=3)
                            graph_btn   = gr.Button("Render Graph", variant="secondary", scale=1)
                        graph_output = gr.HTML(
                            value="<p style='color:#475569;padding:20px'>Click Render Graph to explore your full knowledge graph in 3D.</p>"
                        )
                        graph_btn.click(graph_fn, inputs=[graph_limit], outputs=graph_output)

            # ── Tab 5: Add Bookmarks ───────────────────────────────────────
            with gr.Tab("+ Add"):
                gr.HTML("""
                <div style='color:#64748b;font-size:0.83em;padding:4px 0 10px'>
                  Drop URLs or a file — parsed, deduped, embedded and searchable immediately.
                  Supports: plain URLs · Edge/Chrome HTML export · JSON · .txt
                </div>""")
                with gr.Row():
                    add_urls = gr.Textbox(
                        placeholder="Paste one or more URLs here, one per line...\nhttps://github.com/something\nhttps://arxiv.org/paper...",
                        label="Paste URLs",
                        lines=6,
                        scale=1,
                    )
                    add_file = gr.File(
                        label="Or upload a file (HTML / JSON / TXT)",
                        file_types=[".html", ".htm", ".json", ".txt"],
                        scale=1,
                    )
                with gr.Row():
                    add_fetch = gr.Checkbox(value=True, label="Fetch page titles (slower but better search)")
                    add_btn = gr.Button("Add to Knowledge Base", variant="primary", scale=2)
                add_output = gr.HTML(value="")

                def add_fn(url_text: str, file_obj, fetch_titles: bool):
                    from openmark.pipeline.injector import (
                        extract_urls_from_text, urls_to_items,
                        parse_html_file, parse_json_file, parse_txt_file,
                        run_injection,
                    )
                    items = []
                    errors = []

                    # Parse file if provided
                    if file_obj is not None:
                        path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
                        ext = os.path.splitext(path)[1].lower()
                        try:
                            if ext in (".html", ".htm"):
                                items.extend(parse_html_file(path))
                            elif ext == ".json":
                                items.extend(parse_json_file(path))
                            else:
                                items.extend(parse_txt_file(path, fetch_titles=fetch_titles))
                        except Exception as e:
                            errors.append(f"File parse error: {e}")

                    # Parse pasted URLs
                    if url_text and url_text.strip():
                        urls = extract_urls_from_text(url_text)
                        if urls:
                            items.extend(urls_to_items(urls, fetch_titles=fetch_titles))

                    if not items:
                        return "<p style='color:#f97316;padding:8px'>No URLs found. Paste URLs or upload a file.</p>"

                    try:
                        stats = run_injection(items, embedder=_embedder)
                        color = "#22c55e" if stats["new"] > 0 else "#64748b"
                        msg = (
                            f"<div style='padding:12px;background:#0f172a;border-radius:8px;border:1px solid #1e293b;font-family:sans-serif'>"
                            f"<div style='color:{color};font-size:1.1em;font-weight:700;margin-bottom:6px'>"
                            f"{'✅' if stats['new'] > 0 else '—'} "
                            f"{'Added ' + str(stats['new']) + ' new bookmarks' if stats['new'] > 0 else 'No new bookmarks added'}"
                            f"</div>"
                            f"<div style='color:#64748b;font-size:0.85em'>"
                            f"Parsed: {stats['total']} &nbsp;·&nbsp; New: {stats['new']} &nbsp;·&nbsp; Duplicates skipped: {stats['skipped']}"
                            f"</div>"
                            + (f"<div style='color:#94a3b8;font-size:0.82em;margin-top:6px'>Immediately searchable in the Search tab.</div>" if stats["new"] > 0 else "")
                            + (f"<div style='color:#ef4444;font-size:0.82em;margin-top:4px'>Errors: {'; '.join(errors)}</div>" if errors else "")
                            + "</div>"
                        )
                        return msg
                    except Exception as e:
                        return f"<p style='color:#ef4444'>Injection error: {e}</p>"

                add_btn.click(
                    add_fn,
                    inputs=[add_urls, add_file, add_fetch],
                    outputs=add_output,
                    show_progress="full",
                )

            # ── Tab 6: Agent Tools (inspection panel) ─────────────────────
            with gr.Tab("Agent Tools"):
                gr.HTML("""
                <div style='color:#64748b;font-size:0.83em;padding:4px 0 8px'>
                  Live registry of every tool the orchestrator can call. The
                  task_researcher sub-agent gets the heavy retrieval slice;
                  task_compose_* / task_humanize / task_polish / task_verify
                  / task_author_skill get zero retrieval tools by design.
                </div>""")
                tools_html = gr.HTML()

                def _render_agent_tools_panel():
                    from openmark.agent.subagents import ALL_SUBAGENT_TOOLS as _ORCH_TOOLS
                    try:
                        from openmark.agent.subagents.researcher import (
                            RESEARCHER_TOOLS as _RES_TOOLS,
                        )
                    except Exception as e:
                        return f"<div style='color:#ef4444;padding:8px'>Subagent module import failed: {e}</div>"

                    def _short_desc(t):
                        d = (getattr(t, "description", "") or "").strip()
                        d = d.split("\n", 1)[0]
                        return (d[:200] + "...") if len(d) > 200 else d

                    def _row(tool, owner):
                        name = getattr(tool, "name", str(tool))
                        return (
                            f"<tr>"
                            f"<td style='padding:6px 10px;color:#a78bfa;font-family:monospace'>{name}</td>"
                            f"<td style='padding:6px 10px;color:#64748b;font-size:0.85em'>{owner}</td>"
                            f"<td style='padding:6px 10px;color:#cbd5e1;font-size:0.85em'>{_short_desc(tool)}</td>"
                            f"</tr>"
                        )

                    rows: list[str] = []
                    for t in _ORCH_TOOLS:
                        rows.append(_row(t, "orchestrator (graph.py)"))
                    res_names = {getattr(t, "name", None) for t in _RES_TOOLS}
                    rows.append(_row(type("FakeTool", (), {"name": "(researcher slice)", "description": f"{len(_RES_TOOLS)} retrieval/web tools assigned to the researcher sub-agent: {sorted(filter(None, res_names))}"})(), "researcher sub-agent"))
                    return (
                        f"<div style='color:#94a3b8;padding:2px 0 8px'>"
                        f"<b style='color:#e2e8f0'>{len(_ORCH_TOOLS)}</b> task_* sub-agent tools on the orchestrator · "
                        f"<b style='color:#e2e8f0'>{len(_RES_TOOLS)}</b> retrieval tools inside the researcher sub-agent</div>"
                        "<table style='width:100%;border-collapse:collapse;background:#0f172a;border:1px solid #1e293b;border-radius:8px'>"
                        "<thead><tr style='background:#1e293b'>"
                        "<th style='padding:8px 10px;text-align:left;color:#e2e8f0;font-size:0.85em'>Tool</th>"
                        "<th style='padding:8px 10px;text-align:left;color:#e2e8f0;font-size:0.85em'>Owner</th>"
                        "<th style='padding:8px 10px;text-align:left;color:#e2e8f0;font-size:0.85em'>Description</th>"
                        "</tr></thead><tbody>"
                        + "".join(rows)
                        + "</tbody></table>"
                    )

                tools_refresh = gr.Button("Refresh", variant="secondary")
                tools_refresh.click(_render_agent_tools_panel, outputs=tools_html)
                app.load(_render_agent_tools_panel, outputs=tools_html)

            # ── Tab 8: Agent Skills (inspection panel) ────────────────────
            with gr.Tab("Agent Skills"):
                gr.HTML("""
                <div style='color:#64748b;font-size:0.83em;padding:4px 0 8px'>
                  Live registry of every SKILL.md the agent can load. Three
                  families: <b style='color:#e2e8f0'>openmark-*</b> (curated),
                  <b style='color:#e2e8f0'>humanizer-*</b> (Arabic / Hebrew),
                  and <b style='color:#e2e8f0'>agent-generated-*</b> (skills
                  the agent wrote itself this session via <code>write_skill</code>).
                </div>""")
                skills_html = gr.HTML()

                _FAMILY_COLORS = {
                    "openmark":        "#22c55e",
                    "humanizer":       "#a78bfa",
                    "agent-generated": "#f59e0b",
                    "skill":           "#64748b",
                }

                def _render_agent_skills_panel():
                    from openmark.agent import skills as _sk
                    try:
                        _sk.reload_skills()
                        ls = _sk.list_skills()
                    except Exception as e:
                        return f"<div style='color:#ef4444;padding:8px'>Skill scan failed: {e}</div>"

                    by_family: dict[str, list[dict]] = {}
                    for s in ls:
                        by_family.setdefault(s.get("family", "skill"), []).append(s)

                    parts: list[str] = []
                    parts.append(
                        f"<div style='color:#94a3b8;padding:2px 0 12px'>"
                        f"<b style='color:#e2e8f0'>{len(ls)}</b> skills discovered across "
                        f"<b style='color:#e2e8f0'>{len(by_family)}</b> families</div>"
                    )

                    family_order = ["openmark", "humanizer", "agent-generated", "skill"]
                    for fam in family_order:
                        items = by_family.get(fam) or []
                        if not items:
                            continue
                        color = _FAMILY_COLORS.get(fam, "#64748b")
                        label = "Created this session" if fam == "agent-generated" else f"{fam.title()} family"
                        parts.append(
                            f"<h3 style='color:{color};margin:18px 0 6px;font-size:1em'>"
                            f"{label} <span style='color:#475569;font-size:0.85em'>({len(items)})</span></h3>"
                        )
                        parts.append(
                            "<table style='width:100%;border-collapse:collapse;background:#0f172a;border:1px solid #1e293b;border-radius:8px;margin-bottom:8px'>"
                            "<thead><tr style='background:#1e293b'>"
                            "<th style='padding:6px 10px;text-align:left;color:#e2e8f0;font-size:0.82em'>Short name</th>"
                            "<th style='padding:6px 10px;text-align:left;color:#e2e8f0;font-size:0.82em'>Type</th>"
                            "<th style='padding:6px 10px;text-align:left;color:#e2e8f0;font-size:0.82em'>Description</th>"
                            "<th style='padding:6px 10px;text-align:left;color:#e2e8f0;font-size:0.82em'>Body size</th>"
                            "</tr></thead><tbody>"
                        )
                        for s in items:
                            desc = (s.get("description") or "").strip()
                            if len(desc) > 200:
                                desc = desc[:197] + "..."
                            parts.append(
                                "<tr>"
                                f"<td style='padding:6px 10px;color:{color};font-family:monospace'>/{s['short_name']}</td>"
                                f"<td style='padding:6px 10px;color:#64748b;font-size:0.85em'>{s.get('type','skill')}</td>"
                                f"<td style='padding:6px 10px;color:#cbd5e1;font-size:0.85em'>{desc}</td>"
                                f"<td style='padding:6px 10px;color:#64748b;font-size:0.85em'>{len(s.get('body','')):,} chars</td>"
                                "</tr>"
                            )
                        parts.append("</tbody></table>")
                    return "".join(parts)

                skills_refresh = gr.Button("Refresh", variant="secondary")
                skills_refresh.click(_render_agent_skills_panel, outputs=skills_html)
                app.load(_render_agent_skills_panel, outputs=skills_html)

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("OPENMARK_PORT", "7860")),
        share=False,
        inbrowser=True,
        theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
        css=DARK_CSS,
        # Allow Gradio to serve files from drafts/ so the auto-saved reports
        # are downloadable via /gradio_api/file=drafts/*.md links.
        allowed_paths=[DRAFTS_DIR],
    )
