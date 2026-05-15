"""
Pydantic schemas for OpenMark agent — typed tool returns + grounded answers.

Why: returning Pydantic instead of markdown forces URLs to live in named
fields. The LLM can't fabricate them at synthesis time because the
validator can check every cited URL against state["seen_urls"].

Two layers:
  - retrieval shapes (BookmarkHit, ToolResult) — what tools return to the agent
  - composer shapes (LinkedInPost, NewsletterEssay, ...) — what the composer
    sub-agent emits via response_format=ToolStrategy(<Model>). These are the
    contracts the verifier sub-agent grades and the export layer renders.
"""

from typing import Literal
from pydantic import BaseModel, Field

# Languages humanizer-semitic + polisher pipeline supports today.
PostLanguage = Literal["en", "ar-msa", "ar-egt", "ar-shami", "he"]


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


# ── Composer output shapes (v2 newsletter mission) ───────────────────────────
#
# Each shape carries the SAME `sources` rule: every URL must have appeared
# in a researcher tool result this turn. The verifier sub-agent grades that.

class PostSource(BaseModel):
    """A source attached to a composer output. URL must come from a tool result."""
    url: str = Field(description="MUST appear verbatim in a researcher tool result this turn.")
    title: str = Field(description="Title from the tool result.")
    note: str = Field(default="", description="One short phrase on what this source contributes.")


class LinkedInPost(BaseModel):
    """
    Phone-readable LinkedIn / X / short-post output.
    250-400 words, ONE link in body, no hashtags.
    Matches openmark-newsletter-thread skill rules.
    """
    hook: str = Field(
        min_length=20, max_length=140,
        description="Opening claim, 6-10 words. NO question mark.",
    )
    body_paragraphs: list[str] = Field(
        min_length=4, max_length=6,
        description="4-6 short paragraphs. Each 1-3 sentences. NO bullet lists.",
    )
    closer: str = Field(
        min_length=10, max_length=180,
        description="One sentence. Quotable. NOT a question.",
    )
    anchor_url: str = Field(
        description="The single inline link in the body. MUST match the one entry in `sources`.",
    )
    sources: list[PostSource] = Field(
        min_length=1, max_length=1,
        description="Exactly one anchor source.",
    )
    word_count: int = Field(ge=180, le=420)
    language: PostLanguage = "en"
    humanizer_applied: bool = False
    voice_check: Literal["pass", "warn"] = "pass"


class EssaySection(BaseModel):
    heading: str = Field(
        description="Sub-claim phrasing, NOT a topic name. Headline style, no trailing punctuation.",
    )
    body_markdown: str = Field(
        description="3-6 sentences. Inline links allowed. NO bullet lists in the body.",
    )


class NewsletterEssay(BaseModel):
    """
    Single-thesis long-form essay. 600-900 words.
    Matches openmark-newsletter-essay skill rules.
    `counter` section is mandatory.
    """
    title: str = Field(max_length=64)
    thesis: str = Field(
        min_length=20, max_length=240,
        description="One sentence. The whole essay defends or develops this.",
    )
    opening_paragraph: str
    sections: list[EssaySection] = Field(min_length=3, max_length=5)
    counter: str = Field(
        min_length=80,
        description="Mandatory: the strongest objection, then a sharpened response.",
    )
    closing_paragraph: str
    sources: list[PostSource] = Field(min_length=5, max_length=8)
    word_count: int = Field(ge=550, le=950)
    language: PostLanguage = "en"
    humanizer_applied: bool = False


class RoundupItem(BaseModel):
    title: str
    url: str
    domain: str = Field(description="Just the domain, no scheme, no path. e.g. 'github.com'.")
    so_what: str = Field(
        min_length=10, max_length=200,
        description="One sentence on why this matters. No 'this could be useful'.",
    )


class RoundupBucket(BaseModel):
    name: str = Field(
        description="Bucket name from the skill list (Models & research, Tooling & open source, ...).",
    )
    items: list[RoundupItem] = Field(min_length=2, max_length=5)


class NewsletterRoundup(BaseModel):
    """
    Categorical news-roundup. Bucketed. 3-6 buckets, 2-5 items per bucket.
    Matches openmark-newsletter-roundup skill rules.
    """
    title: str = Field(max_length=80)
    pulse: str = Field(
        min_length=20, max_length=200,
        description="One sentence pulse on the week. The hook.",
    )
    buckets: list[RoundupBucket] = Field(min_length=3, max_length=6)
    sources: list[PostSource] = Field(min_length=6, max_length=24)
    item_count: int = Field(ge=6, le=30)
    window_label: str = Field(
        description="Human-readable window. e.g. 'last 7 days', 'May 8-14'.",
    )
    language: PostLanguage = "en"


class ComparisonRow(BaseModel):
    dimension: str = Field(
        description="What this row compares (License, Cost, DX, Lock-in, Best at, Worst at, ...).",
    )
    values: list[str] = Field(
        min_length=2, max_length=5,
        description="Values for each item in `items`. Must match items length.",
    )


class ComparisonPick(BaseModel):
    item_name: str
    condition: str = Field(description="The single condition that picks this item.")
    rationale: str = Field(description="1-2 sentences with one inline cite.")


class NewsletterComparison(BaseModel):
    """
    Side-by-side comparison newsletter.
    Matches openmark-newsletter-comparison skill rules.
    Table is mandatory.
    """
    title: str = Field(max_length=80)
    recommendation: str = Field(
        min_length=20, max_length=240,
        description="Blockquote line: who wins for what.",
    )
    items: list[str] = Field(min_length=2, max_length=5)
    rows: list[ComparisonRow] = Field(min_length=4, max_length=8)
    how_to_read: str
    picks: list[ComparisonPick] = Field(min_length=2, max_length=5)
    sources: list[PostSource] = Field(min_length=3, max_length=12)
    language: PostLanguage = "en"


class NewsletterAnalytical(BaseModel):
    """
    Punchy analytical newsletter, 600-900 words, with a 'What I'm reading' spine.
    Matches openmark-newsletter (primary) skill rules.
    """
    title: str = Field(max_length=80)
    hook: str = Field(
        min_length=20, max_length=240,
        description="1-2 sentence hook stating the thesis.",
    )
    what_happened_paragraphs: list[str] = Field(
        min_length=3, max_length=5,
        description="Each paragraph cites at least one URL inline.",
    )
    why_it_matters: str = Field(
        min_length=80,
        description="1-2 paragraphs. The take.",
    )
    what_im_reading: list[RoundupItem] = Field(min_length=5, max_length=7)
    one_more_thing: str | None = Field(
        default=None,
        description="Optional weird/funny bookmark that doesn't fit the thesis.",
    )
    sources: list[PostSource] = Field(min_length=5, max_length=15)
    word_count: int = Field(ge=550, le=950)
    language: PostLanguage = "en"


# ── Verifier output shape ─────────────────────────────────────────────────────

VerifierCheck = Literal["pass", "fail"]


class VerificationReport(BaseModel):
    """
    Verifier sub-agent output. The orchestrator branches on `overall_passed`.
    Score = (count of 'pass') / 4 across the four checks. >=0.9 is the bar.
    """
    cite_check: VerifierCheck
    cite_fail_reason: str = ""
    voice_check: VerifierCheck
    voice_fail_reason: str = ""
    word_count_check: VerifierCheck
    word_count_fail_reason: str = ""
    schema_check: VerifierCheck
    schema_fail_reason: str = ""
    overall_passed: bool
    score: float = Field(
        ge=0.0, le=1.0,
        description="(count_of_pass) / 4. >= 0.90 is the project target.",
    )
    fix_instructions: str = Field(
        default="",
        description="If overall_passed=False, exact instructions for the composer to retry.",
    )


# Convenient registry for the composer factory + the export layer.
ComposerFormat = Literal["linkedin", "thread", "essay", "roundup", "comparison", "analytical"]

COMPOSER_SCHEMAS: dict[ComposerFormat, type[BaseModel]] = {
    "linkedin":   LinkedInPost,
    "thread":     LinkedInPost,        # alias — the thread skill IS a LinkedIn post
    "essay":      NewsletterEssay,
    "roundup":    NewsletterRoundup,
    "comparison": NewsletterComparison,
    "analytical": NewsletterAnalytical,
}
