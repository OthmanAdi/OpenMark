"""
OpenMark Gradio UI — 3 tabs:
  1. Chat    — talk to the LangGraph agent
  2. Search  — instant semantic search with filters
  3. Stats   — knowledge base overview
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr
from openmark.agent.graph import build_agent, ask
from openmark.embeddings.factory import get_embedder
from openmark.stores import chroma as chroma_store
from openmark.stores import neo4j_store
from openmark import config

# Load once at startup
print("Loading OpenMark...")
_embedder = get_embedder()
_agent    = build_agent()
print("OpenMark ready.")


# ── Chat tab ──────────────────────────────────────────────────

def chat_fn(message: str, history: list, thread_id: str):
    if not message.strip():
        return history, ""
    response = ask(_agent, message, thread_id=thread_id or "default")
    history.append((message, response))
    return history, ""


# ── Search tab ────────────────────────────────────────────────

def search_fn(query: str, category: str, min_score: float, n_results: int):
    if not query.strip():
        return "Enter a search query."

    cat = category if category != "All" else None
    ms  = min_score if min_score > 0 else None

    results = chroma_store.search(
        query, _embedder, n=int(n_results),
        category=cat, min_score=ms,
    )

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(
            f"**{r['rank']}. {r['title'] or r['url']}**\n"
            f"🔗 {r['url']}\n"
            f"📁 {r['category']} | 📌 {', '.join(t for t in r['tags'] if t)} | "
            f"⭐ {r['score']} | 🎯 {r['similarity']:.3f} similarity\n"
        )
    return "\n---\n".join(lines)


# ── Stats tab ─────────────────────────────────────────────────

def stats_fn():
    chroma = chroma_store.get_stats()
    neo4j  = neo4j_store.get_stats()

    # Category breakdown from Neo4j
    cat_rows = neo4j_store.query("""
        MATCH (b:Bookmark)-[:IN_CATEGORY]->(c:Category)
        RETURN c.name AS category, count(b) AS count
        ORDER BY count DESC
    """)
    cat_lines = "\n".join(f"  {r['category']:<35} {r['count']:>5}" for r in cat_rows)

    # Top tags
    tag_rows = neo4j_store.query("""
        MATCH (b:Bookmark)-[:TAGGED]->(t:Tag)
        RETURN t.name AS tag, count(b) AS count
        ORDER BY count DESC LIMIT 20
    """)
    tag_lines = ", ".join(f"{r['tag']} ({r['count']})" for r in tag_rows)

    return (
        f"## OpenMark Knowledge Base\n\n"
        f"**ChromaDB vectors:** {chroma.get('total', 0)}\n"
        f"**Neo4j bookmarks:** {neo4j.get('bookmarks', 0)}\n"
        f"**Neo4j tags:** {neo4j.get('tags', 0)}\n"
        f"**Neo4j categories:** {neo4j.get('categories', 0)}\n\n"
        f"### By Category\n```\n{cat_lines}\n```\n\n"
        f"### Top Tags\n{tag_lines}"
    )


# ── Build UI ──────────────────────────────────────────────────

def build_ui():
    categories = ["All"] + config.CATEGORIES

    with gr.Blocks(title="OpenMark") as app:
        gr.Markdown("# OpenMark — Your Personal Knowledge Graph")

        with gr.Tabs():

            # Tab 1: Chat
            with gr.Tab("Chat"):
                thread = gr.Textbox(value="default", label="Session ID", scale=1)
                chatbot = gr.Chatbot(height=500)
                msg_box = gr.Textbox(
                    placeholder="Ask anything about your saved bookmarks...",
                    label="Message", lines=2,
                )
                send_btn = gr.Button("Send", variant="primary")

                send_btn.click(
                    chat_fn,
                    inputs=[msg_box, chatbot, thread],
                    outputs=[chatbot, msg_box],
                )
                msg_box.submit(
                    chat_fn,
                    inputs=[msg_box, chatbot, thread],
                    outputs=[chatbot, msg_box],
                )

            # Tab 2: Search
            with gr.Tab("Search"):
                with gr.Row():
                    q_input   = gr.Textbox(placeholder="Search your knowledge base...", label="Query", scale=3)
                    cat_input = gr.Dropdown(categories, value="All", label="Category")
                with gr.Row():
                    score_input = gr.Slider(0, 10, value=0, step=1, label="Min Quality Score")
                    n_input     = gr.Slider(5, 50, value=10, step=5, label="Results")
                search_btn    = gr.Button("Search", variant="primary")
                search_output = gr.Markdown()

                search_btn.click(
                    search_fn,
                    inputs=[q_input, cat_input, score_input, n_input],
                    outputs=search_output,
                )
                q_input.submit(
                    search_fn,
                    inputs=[q_input, cat_input, score_input, n_input],
                    outputs=search_output,
                )

            # Tab 3: Stats
            with gr.Tab("Stats"):
                refresh_btn  = gr.Button("Refresh Stats")
                stats_output = gr.Markdown()

                refresh_btn.click(stats_fn, outputs=stats_output)
                app.load(stats_fn, outputs=stats_output)

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
