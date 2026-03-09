"""
LangGraph ReAct agent for OpenMark.
Uses Azure gpt-4o-mini as the LLM.
Has access to all OpenMark tools (ChromaDB + Neo4j).
"""

from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from openmark import config
from openmark.agent.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are OpenMark — Ahmad's personal AI knowledge assistant.

You have access to his entire curated knowledge base of 7,000+ saved bookmarks,
LinkedIn posts, and YouTube videos — all categorized, tagged, and connected in a
knowledge graph.

Your job:
- Help Ahmad find exactly what he saved and can't remember
- Discover connections between topics he didn't know existed
- Answer questions by searching his real saved content (not your training data)
- Be direct and useful — no filler

When answering:
- Always use tools to search first before responding
- Show the actual URLs and titles from results
- Group results by relevance
- If one search doesn't find enough, try a different angle (by tag, by category, by similarity)

Available search modes:
- search_semantic: natural language search (most useful for general queries)
- search_by_category: filter by topic category
- find_by_tag: exact tag lookup in the knowledge graph
- find_similar_bookmarks: find related content to a specific URL
- explore_tag_cluster: discover what else connects to a topic
- get_stats: see what's in the knowledge base
- run_cypher: advanced graph queries (for power users)
"""


def build_agent():
    llm = AzureChatOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        azure_deployment=config.AZURE_DEPLOYMENT_LLM,
        api_version=config.AZURE_API_VERSION,
        temperature=0,
        streaming=True,
    )

    checkpointer = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


def ask(agent, question: str, thread_id: str = "default") -> str:
    """Run a question through the agent and return the final text response."""
    config_run = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config_run,
    )
    return result["messages"][-1].content
