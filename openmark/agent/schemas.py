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


# ── Two-mode response_format ──────────────────────────────────────────────────
# The agent picks one based on user intent. LangChain's ToolStrategy(Union[...])
# validates the chosen shape and stores it on state["structured_response"].

class QuickAnswerHit(BaseModel):
    """A single cited bookmark for a quick answer. URL must come from a tool result."""
    title: str
    url: str
    why: str = Field(default="", description="One short phrase. Max 12 words. May be empty.")


class QuickAnswer(BaseModel):
    """
    Use this for short, single-fact, or one-shot lookup questions.

    Triggers: 'find my X', 'what did I save about Y', 'do I have Z',
              fast-search slash, single-URL dive lookups.

    Format: one tight summary (1-3 sentences) + up to 8 cited hits.
    """
    summary: str = Field(description="1-3 sentences. No preamble. No 'I found...'.")
    hits: list[QuickAnswerHit] = Field(
        default_factory=list,
        description="Up to 8 bookmarks the user should look at, ranked by relevance.",
    )


class ReportSection(BaseModel):
    """One section of a long-form report."""
    heading: str = Field(description="Section title. Headline-style, no punctuation at end.")
    body_markdown: str = Field(
        description=(
            "Section body as markdown. May include sub-headings, bullet lists, "
            "inline code, bold/italic, and inline links to OpenMark URLs. "
            "Cite URLs inline as [descriptive phrase](url)."
        ),
    )


class ReportTable(BaseModel):
    """An optional comparison table to render inside the report."""
    caption: str = Field(default="", description="One-line table caption (optional).")
    headers: list[str] = Field(description="Column headers in order.")
    rows: list[list[str]] = Field(
        description="Row values aligned to headers. Each row's length must equal headers length.",
    )


class Report(BaseModel):
    """
    Use this for research, comparison, newsletter material, weekly digest,
    landscape overviews, anything multi-faceted or requiring sections.

    Triggers: 'research X', 'compose newsletter', 'compare A vs B', 'what's
              the landscape of...', 'weekly digest', 'deep dive'.

    Format: title + tldr + 3-6 sections + optional table + flat citations list.
    """
    title: str = Field(description="Punchy, specific title. Max 9 words.")
    tldr: str = Field(description="2-3 sentence headline finding. The user's takeaway.")
    sections: list[ReportSection] = Field(
        description="3-6 sections covering the question from different angles.",
    )
    table: ReportTable | None = Field(
        default=None,
        description=(
            "Optional comparison or summary table. ONLY include when the content "
            "genuinely benefits from tabular layout (e.g. comparing tools, "
            "side-by-side feature sets, ranked lists with multiple columns)."
        ),
    )
    citations: list[Citation] = Field(
        description=(
            "Every URL referenced in sections above. Flat list, in order of first "
            "mention. URLs MUST appear in a tool result during this turn."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
    suggested_followups: list[str] = Field(
        default_factory=list,
        description="0-3 concrete follow-up questions the user could ask next.",
    )
