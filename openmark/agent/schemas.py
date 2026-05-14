"""
Pydantic schemas for OpenMark agent — typed tool returns + grounded answers.

Why: returning Pydantic instead of markdown forces URLs to live in named
fields. The LLM can't fabricate them at synthesis time because the
validator can check every cited URL against state["seen_urls"].
"""

from typing import Literal
from pydantic import BaseModel, Field


Strategy = Literal[
    "semantic", "category", "tag", "community",
    "graph_expand", "domain", "source", "cypher",
]


class BookmarkHit(BaseModel):
    """One retrieved bookmark — every field auditable."""
    url: str
    title: str = ""
    similarity: float = 0.0
    bm_score: float = 0.0
    source: str = ""           # raindrop, linkedin, youtube_*, edge, manual
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    community_id: int | None = None


class ToolResult(BaseModel):
    """Standard envelope for every retrieval tool."""
    hits: list[BookmarkHit] = Field(default_factory=list)
    strategy: Strategy
    query_echo: str = ""
    total_found: int = 0
    note: str = ""             # free-text for agent (e.g. "No results — try X")

    def to_compact_markdown(self) -> str:
        """Render for the LLM tool message. URLs preserved verbatim."""
        if not self.hits:
            return f"[strategy={self.strategy}] No results for '{self.query_echo}'. {self.note}".strip()

        lines = [f"[strategy={self.strategy}] {self.total_found} hits for '{self.query_echo}':"]
        for i, h in enumerate(self.hits, 1):
            tags = ", ".join(h.tags[:5]) if h.tags else "—"
            lines.append(
                f"{i}. {h.title}\n"
                f"   URL: {h.url}\n"
                f"   sim={h.similarity:.3f} cat={h.category or '—'} src={h.source} tags={tags}"
            )
        if self.note:
            lines.append(f"NOTE: {self.note}")
        return "\n".join(lines)


class Citation(BaseModel):
    url: str
    title: str
    why_relevant: str = Field(..., description="One short sentence connecting this bookmark to the user's question.")


class Answer(BaseModel):
    """Final grounded answer — citations validated against retrieved URLs."""
    summary: str = Field(..., description="Direct answer in 2-4 sentences. No filler.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Only URLs that appeared in tool results. NEVER fabricate.",
    )
    confidence: Literal["high", "medium", "low"] = "medium"
    suggested_followups: list[str] = Field(default_factory=list)
