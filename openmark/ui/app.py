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
from openmark import config

print("Loading OpenMark...")

# ── Load embedder (always local — no API key needed) ──────────────────────────
_embedder = None
_embedder_error = None
try:
    from openmark.embeddings.factory import get_embedder
    from openmark.stores import chroma as chroma_store
    from openmark.stores import neo4j_store
    _embedder = get_embedder()
    print("Embedder ready (local pplx-embed).")
except Exception as e:
    _embedder_error = str(e)
    print(f"Embedder failed: {e}")

# ── Load agent (needs Azure key — optional) ───────────────────────────────────
_agent = None
_agent_error = None
try:
    from openmark.agent.graph import build_agent, ask
    _agent = build_agent()
    print("Agent ready (Azure GPT-4o-mini).")
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
    filled = int(pct / 5)  # 20 segments
    bar = "█" * filled + "░" * (20 - filled)
    color = "#22c55e" if pct >= 75 else "#f97316" if pct >= 50 else "#6b7280"
    return (
        f'<span style="font-family:monospace;color:{color};font-size:0.8em">'
        f'{bar} {pct}%</span>'
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
        f'<span style="background:#1e293b;color:#94a3b8;padding:2px 8px;'
        f'border-radius:9999px;font-size:0.72em;margin-right:3px">{t}</span>'
        for t in tags[:6]
    )

    stars = "★" * score + "☆" * (10 - score) if score else ""

    return f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
            padding:14px 18px;margin-bottom:10px;font-family:sans-serif">

  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
    <span style="font-size:1.1em">{icon}</span>
    <a href="{url}" target="_blank"
       style="color:#e2e8f0;font-weight:600;font-size:1em;
              text-decoration:none;flex:1;line-height:1.3">
      {r['rank']}. {title}
    </a>
    <span style="background:{color};color:#fff;padding:2px 10px;
                 border-radius:9999px;font-size:0.72em;white-space:nowrap">
      {cat}
    </span>
  </div>

  <div style="font-size:0.78em;color:#64748b;margin-bottom:6px;
              word-break:break-all">
    <a href="{url}" target="_blank" style="color:#3b82f6">{url[:90]}{"…" if len(url)>90 else ""}</a>
  </div>

  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <div>{_similarity_bar(sim)}</div>
    {"<div style='color:#fbbf24;font-size:0.8em'>" + stars + "</div>" if stars else ""}
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

    results = chroma_store.search(
        query, _embedder, n=int(n_results),
        category=cat, min_score=ms,
    )

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
    return header + cards


# ── Chat ──────────────────────────────────────────────────────────────────────

def chat_fn(message: str, history: list, thread_id: str):
    """Used by gr.ChatInterface — returns the assistant reply string."""
    if _agent is None:
        return (
            "**Agent unavailable** — Azure API key not configured in `.env`.\n\n"
            "Use the **Search** tab — it works fully offline with local pplx-embed.\n\n"
            f"*Error: {_agent_error}*"
        )
    from openmark.agent.graph import ask
    return ask(_agent, message, thread_id=thread_id or "default")


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats_fn():
    if _embedder is None:
        return f"<p style='color:#ef4444'>Embedder not loaded.</p>"

    chroma = chroma_store.get_stats()

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
        neo4j_section = f"""
<div style="margin-top:16px">
  <div style="color:#94a3b8;font-size:0.85em;margin-bottom:8px">NEO4J GRAPH</div>
  <div style="display:flex;gap:20px;flex-wrap:wrap">
    {"".join(
        f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px 20px;text-align:center">'
        f'<div style="font-size:1.6em;font-weight:700;color:#e2e8f0">{v}</div>'
        f'<div style="font-size:0.75em;color:#64748b">{k}</div></div>'
        for k, v in [
            ("Bookmarks", neo4j.get("bookmarks", 0)),
            ("Tags", neo4j.get("tags", 0)),
            ("Categories", neo4j.get("categories", 0)),
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
    f'<span style="background:#1e293b;color:#94a3b8;padding:3px 10px;border-radius:9999px;font-size:0.78em;margin:2px">{r["tag"]} <strong style=color:#e2e8f0>{r["count"]}</strong></span>'
    for r in tag_rows
  )}
  </div>
</div>
"""
    except Exception as e:
        neo4j_section = f"<p style='color:#f97316'>Neo4j not available: {e}</p>"

    return f"""
<div style="font-family:sans-serif;color:#e2e8f0">

  <div style="margin-bottom:16px">
    <div style="color:#94a3b8;font-size:0.85em;margin-bottom:8px">CHROMADB VECTORS</div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px 20px;display:inline-block">
      <div style="font-size:2em;font-weight:700;color:#6366f1">{chroma.get("total", 0):,}</div>
      <div style="font-size:0.75em;color:#64748b">embedded documents</div>
    </div>
  </div>

  {neo4j_section}
</div>
"""


# ── Build UI ──────────────────────────────────────────────────────────────────

DARK_CSS = """
body, .gradio-container { background: #020617 !important; color: #e2e8f0 !important; }
.gr-button-primary { background: #6366f1 !important; border: none !important; }
.gr-button-primary:hover { background: #4f46e5 !important; }
footer { display: none !important; }
"""


def build_ui():
    categories = ["All"] + config.CATEGORIES

    with gr.Blocks(title="OpenMark", css=DARK_CSS, theme=gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="slate",
    )) as app:

        gr.HTML("""
        <div style="padding:20px 0 10px;font-family:sans-serif">
          <h1 style="margin:0;font-size:1.6em;color:#e2e8f0">
            🔖 <strong>OpenMark</strong>
            <span style="font-weight:300;color:#64748b"> — Personal Knowledge Graph</span>
          </h1>
          <p style="margin:4px 0 0;color:#475569;font-size:0.85em">
            10,000+ bookmarks · pplx-embed-v1-4b · ChromaDB + Neo4j
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

                search_output = gr.HTML(
                    value="<p style='color:#475569;font-size:0.9em;padding:20px 0'>Results will appear here.</p>",
                    min_height=200,
                )

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

            # ── Tab 2: Chat ────────────────────────────────────────────────
            with gr.Tab("Chat" + (" ✓" if _agent else " (no key)")):
                thread_input = gr.Textbox(
                    value="default", label="Session ID",
                    info="Change to start a fresh conversation thread",
                    scale=1,
                )
                gr.ChatInterface(
                    fn=chat_fn,
                    additional_inputs=[thread_input],
                    chatbot=gr.Chatbot(height=480, placeholder=(
                        "<div style='text-align:center;color:#475569;padding:40px'>"
                        "<div style='font-size:2em'>🔖</div>"
                        "<div>Ask anything about your 10,000+ saved bookmarks</div>"
                        "<div style='font-size:0.8em;margin-top:8px'>e.g. \"What did I save about LangGraph agents?\"</div>"
                        "</div>"
                    )),
                    examples=[
                        "What did I save about LangGraph?",
                        "Find my bookmarks on fine-tuning LLMs",
                        "Show me everything about vector databases",
                        "What YouTube videos did I save about AI agents?",
                        "Find resources on context engineering",
                    ],
                    submit_btn="Send",
                    stop_btn="Stop",
                    fill_height=False,
                )

            # ── Tab 3: Stats ───────────────────────────────────────────────
            with gr.Tab("Stats"):
                refresh_btn  = gr.Button("Refresh", variant="secondary")
                stats_output = gr.HTML()
                refresh_btn.click(stats_fn, outputs=stats_output)
                app.load(stats_fn, outputs=stats_output)

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
