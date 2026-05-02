"""
LangGraph ReAct agent for OpenMark — Graph RAG edition.

Primary LLM: Bonsai 1.7B via local llama-server (AGENT_PROVIDER=local)
Fallback:    Azure gpt-5-mini (AGENT_PROVIDER=azure)
"""

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from openmark import config
from openmark.agent.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are OpenMark — Ahmad's personal AI knowledge assistant.

You have access to his entire curated knowledge base of 8,831+ saved bookmarks,
LinkedIn posts, and YouTube videos. All items are stored in a knowledge graph (Neo4j)
with semantic embeddings (pplx-embed 1024-dim) and Louvain community structure.

Sources: Edge browser (4,359) + Raindrop (2,094) + LinkedIn (1,260) + daily.dev (430) + YouTube (189)

Your job:
- Help Ahmad find exactly what he saved and can't remember
- Surface hidden connections between topics via the knowledge graph
- Answer by searching his real saved content — never make up resources

Search strategy (in order):
1. search_semantic — vector + graph search, use this first, always
2. search_by_category — when the topic maps to a known category
3. search_by_community — when you want to find everything in a topic cluster
4. graph_expand — when you have a specific URL and want related items
5. get_stats — to see what's available in the knowledge base

Rules:
- Always call a search tool before answering
- Show real URLs and titles from results
- If one search angle fails, try a different one
- Be direct, no filler

Categories available: RAG & Vector Search, Agent Development, LangChain / LangGraph,
MCP & Tool Use, Context Engineering, LLM Fine-tuning, AI Tools & Platforms,
GitHub Repos & OSS, Learning & Courses, YouTube & Video, Web Development,
Cloud & Infrastructure, Data Science & ML, Knowledge Graphs & Neo4j,
Career & Jobs, Finance & Crypto, Design & UI/UX, News & Articles, Entertainment & Other
"""


def _build_azure_llm():
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        azure_deployment=config.AZURE_DEPLOYMENT_LLM,
        api_version=config.AZURE_API_VERSION,
        streaming=True,
    )


def _build_local_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=config.BONSAI_URL,
        api_key="none",
        model=config.BONSAI_MODEL,
        streaming=True,
        # Qwen3 thinking tokens: disable for cleaner tool-use responses
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )


def build_agent():
    provider = config.AGENT_PROVIDER.lower()

    if provider == "local":
        try:
            llm = _build_local_llm()
            print(f"Agent ready (local Bonsai @ {config.BONSAI_URL})")
        except Exception as e:
            print(f"Local agent failed ({e}), falling back to Azure")
            llm = _build_azure_llm()
            print(f"Agent ready ({config.AZURE_DEPLOYMENT_LLM} via Azure)")
    else:
        llm = _build_azure_llm()
        print(f"Agent ready ({config.AZURE_DEPLOYMENT_LLM} via Azure)")

    checkpointer = MemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


def ask(agent, question: str, thread_id: str = "default") -> str:
    config_run = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config_run,
    )
    return result["messages"][-1].content
