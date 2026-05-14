"""Smoke-test Hermes 3 8B via Ollama's OpenAI-compatible endpoint with a tool call.
Confirms the agent can drive tools locally."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="local")

tools = [{
    "type": "function",
    "function": {
        "name": "search_semantic",
        "description": "Search Ahmad's bookmark knowledge base by meaning. Returns top N hits.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, 1-12 words."},
                "n": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
}]

resp = client.chat.completions.create(
    model="hermes3:8b",
    messages=[
        {"role": "system", "content": "You are an assistant with access to a bookmark search tool. ALWAYS call search_semantic when asked to find bookmarks."},
        {"role": "user", "content": "Find my bookmarks on LangGraph agents."},
    ],
    tools=tools,
    tool_choice="auto",
    temperature=0,
)

m = resp.choices[0].message
print("finish_reason:", resp.choices[0].finish_reason)
print("content      :", (m.content or "")[:200])
print("tool_calls   :", m.tool_calls)
if m.tool_calls:
    tc = m.tool_calls[0]
    print(f"  -> name = {tc.function.name}")
    print(f"  -> args = {tc.function.arguments}")
    print("\nVERDICT: tool calling WORKS on this model.")
else:
    print("\nVERDICT: model did NOT emit a tool call.")
