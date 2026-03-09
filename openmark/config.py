import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Embedding
EMBEDDING_PROVIDER     = os.getenv("EMBEDDING_PROVIDER", "local")
PPLX_QUERY_MODEL       = os.getenv("PPLX_QUERY_MODEL", "perplexity-ai/pplx-embed-v1-0.6b")
PPLX_DOC_MODEL         = os.getenv("PPLX_DOC_MODEL", "perplexity-ai/pplx-embed-context-v1-0.6b")

# Azure
AZURE_ENDPOINT         = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY          = os.getenv("AZURE_API_KEY")
AZURE_DEPLOYMENT_LLM   = os.getenv("AZURE_DEPLOYMENT_LLM", "gpt-4o-mini")
AZURE_DEPLOYMENT_EMBED = os.getenv("AZURE_DEPLOYMENT_EMBED", "text-embedding-ada-002")
AZURE_API_VERSION      = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

# Neo4j
NEO4J_URI              = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER             = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD         = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE         = os.getenv("NEO4J_DATABASE", "neo4j")

# Raindrop
RAINDROP_TOKEN         = os.getenv("RAINDROP_TOKEN")

# Paths
RAINDROP_MISSION_DIR   = os.getenv("RAINDROP_MISSION_DIR", r"C:\Users\oasrvadmin\Documents\raindrop-mission")
CHROMA_PATH            = os.getenv("CHROMA_PATH", r"C:\Users\oasrvadmin\Documents\OpenMark\data\chroma_db")

# Canonical categories
CATEGORIES = [
    "RAG & Vector Search",
    "LLM Fine-tuning",
    "Agent Development",
    "LangChain / LangGraph",
    "MCP & Tool Use",
    "Context Engineering",
    "AI Tools & Platforms",
    "GitHub Repos & OSS",
    "Learning & Courses",
    "YouTube & Video",
    "Web Development",
    "Cloud & Infrastructure",
    "Data Science & ML",
    "Knowledge Graphs & Neo4j",
    "Career & Jobs",
    "Finance & Crypto",
    "Design & UI/UX",
    "News & Articles",
    "Entertainment & Other",
]

CATEGORY_MAP = {
    "UI/UX Design":               "Design & UI/UX",
    "UI/UX":                      "Design & UI/UX",
    "Real_Estate":                "Finance & Crypto",
    "Real Estate":                "Finance & Crypto",
    "Social_Media":               "News & Articles",
    "Social/Community":           "News & Articles",
    "Social":                     "News & Articles",
    "E-commerce & Marketplaces":  "News & Articles",
    "Research & Articles":        "News & Articles",
    "Blogs & Articles":           "News & Articles",
    "Research":                   "News & Articles",
    "AI Thought Leaders & Media": "News & Articles",
    "Debugging & Tools":          "AI Tools & Platforms",
    "Health & Wellness":          "Entertainment & Other",
    "Email & Productivity":       "AI Tools & Platforms",
    "Legal":                      "Entertainment & Other",
    "NoCode - LowCode":           "AI Tools & Platforms",
    "Security":                   "AI Tools & Platforms",
}
