"""
SubAgent registry for the OpenMark newsletter orchestrator.

Each sub-agent is a `deepagents.SubAgent` TypedDict, registered with the
orchestrator via `create_deep_agent(subagents=...)`. Sub-agents inherit:
  - the orchestrator's model (via `build_executor()`, so AGENT_PROVIDER stays in charge)
  - the orchestrator's tool registry (least-privilege per role — see researcher)
  - the orchestrator's skill catalogue via OpenMarkSkillMiddleware

Six sub-agent families:

  researcher          — heavy tools, no schema, gathers anchors and citations
  composer-{format}   — NO tools, response_format=ToolStrategy(<FormatModel>),
                        one per format in COMPOSER_SCHEMAS
  humanizer           — NO tools, loads humanizer-* skill bodies by language
  polisher            — NO tools, loads openmark-polisher (English AI-tell scrub)
  verifier            — NO tools, response_format=ToolStrategy(VerificationReport)
  skill-author        — `write_skill` tool only; runs when orchestrator
                        decides to bake a recurring prompt into a reusable skill

Sub-agent system prompts intentionally short — the real recipe comes from the
matching SKILL.md, which the sub-agent's first turn loads via load_skill().
This is the same progressive-disclosure pattern OpenMarkSkillMiddleware uses
for the chat agent.
"""

from __future__ import annotations

from typing import Any

from deepagents import SubAgent
from langchain.agents.structured_output import ToolStrategy

from openmark.agent.schemas import (
    COMPOSER_SCHEMAS,
    ComposerFormat,
    VerificationReport,
)
from openmark.agent.tools import (
    explore_tag_cluster,
    find_by_domain,
    find_by_source,
    find_by_tag,
    find_recent,
    get_bookmark_full,
    get_stats,
    github_repo_intel,
    graph_expand,
    reddit_search,
    run_cypher,
    search_by_category,
    search_by_community,
    search_by_date_range,
    search_linkedin,
    search_semantic,
    search_youtube,
    web_crawl,
    web_extract,
    web_fetch,
    web_search,
    write_skill,
)


# ── Researcher tool slice ────────────────────────────────────────────────────
# 17 tools: 11 OpenMark graph + 6 web research. NO write tools.
RESEARCHER_TOOLS: list[Any] = [
    search_semantic,
    search_by_category,
    search_by_community,
    find_by_tag,
    explore_tag_cluster,
    graph_expand,
    find_by_domain,
    find_by_source,
    search_linkedin,
    search_youtube,
    find_recent,
    search_by_date_range,
    get_bookmark_full,
    get_stats,
    run_cypher,                # read-only enforced inside the tool
    web_search,
    web_fetch,
    web_extract,
    web_crawl,
    github_repo_intel,
    reddit_search,
]


# ── Prompts (intentionally short — real recipe lives in matching SKILL.md) ──

_RESEARCHER_PROMPT = """You are the OpenMark Researcher sub-agent.

Your one job: gather anchors and citations for the composer.

CALL `load_skill('deep-research')` on your first turn for the full recipe.
Then follow it: parallel search (semantic + community OR category + linkedin),
graph_expand winners, optional web_search for fresh angles.

When done, return a JSON summary of your findings:
  {
    "anchors": [{"url":..., "title":..., "why":..., "source":...}, ...],
    "secondary": [{...}, ...],
    "notes":   "anything the composer should know"
  }

Cite ONLY URLs you actually retrieved. Never invent.
"""

_COMPOSER_PROMPT_TEMPLATE = """You are the OpenMark Composer sub-agent for format=`{format_name}`.

You receive: a topic, an angle, a language, AND the researcher's anchor list.
You do NOT research. You compose.

CALL `load_skill('{skill_short_name}')` on your first turn for voice / structure /
format rules. Follow them exactly — they encode Ahmad's house style.

OUTPUT: a single `{schema_name}` Pydantic instance. The harness will validate
and retry once on schema error with the validation message. If it fails twice,
return the closest valid shape with `voice_check='warn'` so the verifier sees it.

Citation rule (NON-NEGOTIABLE): every URL in your output MUST come from the
researcher's anchors. If you cannot back a claim, drop the claim — never
invent a URL.

Language: emit body text in the language the user asked for ({lang_note}).
"""

_HUMANIZER_PROMPT = """You are the OpenMark Humanizer sub-agent.

You receive: a composer draft + a target language (one of ar-msa, ar-egt,
ar-shami, he). Run ONE pass:

  1. Pick the matching skill: humanizer-ar-msa, humanizer-ar-egt,
     humanizer-ar-shami, or humanizer-he. CALL `load_skill('<short_name>')`.
  2. Apply the skill's three stages (identify, rewrite, audit) on the draft body.
  3. Set `humanizer_applied = True` on the returned Pydantic instance.

Output: the SAME Pydantic shape you received with rewritten body strings.
DO NOT change `sources`, URLs, structure, word counts. ONLY rewrite text.
ENGLISH drafts are not your job — the polisher handles those.
"""

_POLISHER_PROMPT = """You are the OpenMark Polisher sub-agent — English only.

CALL `load_skill('polisher')` on your first turn for the 8-rule scrub list.

Output: the SAME Pydantic shape you received with the 8 AI tells removed:
em-dash-as-pause, weak hedges, pleasantries, rule-of-three padding, AI
vocabulary (leverage / delve / synergy / unlock / etc), negative parallelism,
inflated symbolism, vague attributions.

Preserve word count within 10%. Preserve every URL and citation. If you
finish and you're under the schema's min word count, replace with one
specific noun — never with filler.
"""

_VERIFIER_PROMPT = """You are the OpenMark Verifier sub-agent.

CALL `load_skill('verifier')` on your first turn for the 4-check grading rules.

You receive: a Pydantic composer output + a `seen_urls` list of every URL
that appeared in researcher tool results this turn.

Run 4 deterministic checks:
  1. cite_check       — every URL in the draft is in seen_urls
  2. voice_check      — format-specific voice rules respected
  3. word_count_check — body word count inside schema bounds (10% slack)
  4. schema_check     — Pydantic instance validates, language is allowed

Emit ONE `VerificationReport`. overall_passed = score >= 0.90.
If any check failed, write a SURGICAL `fix_instructions` for the composer's
retry — exact strings to change, not vague advice.
"""

_SKILL_AUTHOR_PROMPT = """You are the OpenMark Skill Author sub-agent.

You are called ONLY when the orchestrator decides a recurring prompt should
be baked into a reusable skill. You have access to `write_skill` and nothing
else.

Inputs: a `name` (kebab-case, 2-50 chars), a one-line `description`, and a
`body` — the skill recipe in plain markdown.

Call `write_skill(name, description, body)` ONCE. Return its result string.

Cap: 5 writes per session, sandboxed to `.claude/skills/agent-generated-*`.
The orchestrator can `load_skill('<name>')` immediately after.
"""


# ── Sub-agent factories ──────────────────────────────────────────────────────
# Each factory returns a `SubAgent` TypedDict suitable for create_deep_agent.


def _composer_subagent(fmt: ComposerFormat) -> SubAgent:
    schema = COMPOSER_SCHEMAS[fmt]
    # Map format -> matching openmark- skill short_name
    skill_for_format = {
        "linkedin":   "newsletter-thread",
        "thread":     "newsletter-thread",
        "essay":      "newsletter-essay",
        "roundup":    "newsletter-roundup",
        "comparison": "newsletter-comparison",
        "analytical": "newsletter",
    }[fmt]
    return SubAgent(
        name=f"composer-{fmt}",
        description=(
            f"Compose a `{fmt}` output ({schema.__name__}) from a topic + the "
            f"researcher's anchors. No research, no tools, schema-strict."
        ),
        system_prompt=_COMPOSER_PROMPT_TEMPLATE.format(
            format_name=fmt,
            skill_short_name=skill_for_format,
            schema_name=schema.__name__,
            lang_note="default English; the caller passes language= in the brief",
        ),
        tools=[],
        response_format=ToolStrategy(schema),
    )


def build_researcher() -> SubAgent:
    return SubAgent(
        name="researcher",
        description=(
            "Gather anchors + citations from Ahmad's OpenMark bookmarks (Neo4j "
            "Graph RAG) and the open web. Returns a JSON anchor list for the "
            "composer."
        ),
        system_prompt=_RESEARCHER_PROMPT,
        tools=RESEARCHER_TOOLS,
    )


def build_humanizer() -> SubAgent:
    return SubAgent(
        name="humanizer",
        description=(
            "Rewrite an Arabic (MSA / Egyptian / Levantine) or Hebrew composer "
            "draft to remove AI tells. Loads the matching humanizer-* skill."
        ),
        system_prompt=_HUMANIZER_PROMPT,
        tools=[],
    )


def build_polisher() -> SubAgent:
    return SubAgent(
        name="polisher",
        description=(
            "English AI-tell scrub. Removes em-dash-as-pause, AI vocabulary, "
            "rule-of-three padding, hedges, vague attributions."
        ),
        system_prompt=_POLISHER_PROMPT,
        tools=[],
    )


def build_verifier() -> SubAgent:
    return SubAgent(
        name="verifier",
        description=(
            "Grade a composer output on cite_check / voice_check / "
            "word_count_check / schema_check. Returns VerificationReport. "
            "score >= 0.90 passes."
        ),
        system_prompt=_VERIFIER_PROMPT,
        tools=[],
        response_format=ToolStrategy(VerificationReport),
    )


def build_skill_author() -> SubAgent:
    return SubAgent(
        name="skill-author",
        description=(
            "Author a new reusable skill at runtime. Sandboxed; max 5/session. "
            "The orchestrator can use the new skill on the same turn."
        ),
        system_prompt=_SKILL_AUTHOR_PROMPT,
        tools=[write_skill],
    )


def build_all_subagents() -> list[SubAgent]:
    """Return the full sub-agent registry for the orchestrator."""
    composers = [_composer_subagent(fmt) for fmt in COMPOSER_SCHEMAS.keys()]
    return [
        build_researcher(),
        *composers,
        build_humanizer(),
        build_polisher(),
        build_verifier(),
        build_skill_author(),
    ]
