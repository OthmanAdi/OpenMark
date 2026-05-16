"""
Intent classification middleware — pure LangChain.

Classifies the user's first message into one of five intents and writes the
label into agent state. The orchestrator's dynamic_prompt then reads the
label and tailors its behaviour.

Resolution order (cheapest to most expensive):
    1. Slash command   — user typed `/<skill>` → direct lookup
    2. Heuristic regex — easy 80 percent caught without an LLM call
    3. Fast LLM call   — gpt-4.1-mini (or whatever role_model_id('classifier') resolves to)

Once a label is set in state, this middleware is a no-op for subsequent turns
in the same thread — classification is sticky.
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal

from langchain.agents.middleware import AgentState, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from openmark.agent import skills as _skill_loader
from openmark.agent.llms import build_classifier
from openmark.agent.middleware import log


Intent = Literal["fast", "deep", "newsletter", "digest", "dive"]


class IntentLabel(BaseModel):
    """Structured output schema for the LLM classifier."""

    intent: Intent = Field(
        description="One of: fast (single lookup), deep (research/landscape), "
                    "newsletter (drafting), digest (time-window recap), dive (one URL)."
    )


class OrchestratorState(AgentState):
    """AgentState extension carrying the classified intent label."""

    intent: str | None
    intent_source: str | None


# ── Slash + heuristic helpers ───────────────────────────────────────────────


_SLASH_TO_INTENT: dict[str, Intent] = {
    "fast-search":           "fast",
    "deep-research":         "deep",
    "newsletter":            "newsletter",
    "newsletter-essay":      "newsletter",
    "newsletter-roundup":    "newsletter",
    "newsletter-thread":     "newsletter",
    "newsletter-comparison": "newsletter",
    "weekly-digest":         "digest",
    "bookmark-dive":         "dive",
    "repo-research":         "deep",
    "niche-hunter":          "deep",
}

_URL_RE       = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
_DIVE_VERBS   = re.compile(r"\b(expand|dig\s+into|neighbors\s+of|related\s+to|dive\s+into)\b", re.IGNORECASE)
_NEWSLETTER   = re.compile(r"\bnewsletter\b", re.IGNORECASE)
_DIGEST       = re.compile(
    r"\b(this\s+(week|month)|last\s+(\d+\s*)?(day|week|month)s?|"
    r"past\s+(\d+\s*)?(day|week|month)s?|weekly|monthly|yesterday|today|recap)\b",
    re.IGNORECASE,
)
_DEEP         = re.compile(
    r"\b(research|compare|comparison|landscape|deep[-\s]?dive|"
    r"overview|state\s+of|survey|map\s+out)\b",
    re.IGNORECASE,
)


def _slash_intent(text: str) -> Intent | None:
    name, _ = _skill_loader.parse_slash(text or "")
    if not name:
        return None
    short = name.lower().lstrip("/")
    if short.startswith("openmark-"):
        short = short[len("openmark-"):]
    return _SLASH_TO_INTENT.get(short)


def _heuristic_intent(text: str) -> Intent | None:
    t = (text or "").strip()
    if not t:
        return None
    if _URL_RE.search(t) and _DIVE_VERBS.search(t):
        return "dive"
    if _NEWSLETTER.search(t):
        return "newsletter"
    if _DIGEST.search(t):
        return "digest"
    if _DEEP.search(t):
        return "deep"
    return None


# ── LLM classifier (cached) ─────────────────────────────────────────────────


_CLASSIFIER_LLM = None


def _get_classifier_llm():
    global _CLASSIFIER_LLM
    if _CLASSIFIER_LLM is None:
        _CLASSIFIER_LLM = build_classifier().with_structured_output(IntentLabel)
    return _CLASSIFIER_LLM


def _llm_classify(text: str) -> Intent | None:
    if not text or not text.strip():
        return None
    prompt = (
        "Classify the user's query about Ahmad's bookmark knowledge base.\n\n"
        "Labels:\n"
        "- fast        : single concrete lookup (one-tool answer)\n"
        "- deep        : multi-angle research, comparison, landscape questions\n"
        "- newsletter  : asked to draft / compose / write a newsletter\n"
        "- digest      : 'what did I save this week / last N days', time-window recap\n"
        "- dive        : a single URL with 'expand', 'dig into', 'neighbors of'\n\n"
        f"User query: {text.strip()}\n\nLabel:"
    )
    try:
        t0 = time.time()
        out = _get_classifier_llm().invoke(prompt)
        log.info(f"[classify-llm] {out.intent!r} in {int((time.time()-t0)*1000)}ms")
        return out.intent
    except Exception as e:
        log.info(f"[classify-llm] failed: {e!r}; defaulting fast")
        return None


# ── Helpers ─────────────────────────────────────────────────────────────────


def _last_human_text(messages: list) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
            c = getattr(m, "content", "")
            if isinstance(c, list):
                return " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in c
                )
            return c or ""
    return ""


# ── before_model hook ───────────────────────────────────────────────────────


@before_model(state_schema=OrchestratorState)
def classify_intent(state: OrchestratorState, runtime: Any) -> dict | None:
    """
    Sticky classifier. Sets state['intent'] + state['intent_source'] exactly
    once per thread. Subsequent turns short-circuit since the label is set.
    """
    if state.get("intent"):
        return None
    msg = _last_human_text(state.get("messages") or [])
    if not msg.strip():
        return None
    # 1) Slash
    label = _slash_intent(msg)
    if label:
        log.info(f"[classify] slash -> {label}")
        return {"intent": label, "intent_source": "slash"}
    # 2) Heuristic
    label = _heuristic_intent(msg)
    if label:
        log.info(f"[classify] heuristic -> {label}")
        return {"intent": label, "intent_source": "heuristic"}
    # 3) LLM
    label = _llm_classify(msg) or "fast"
    return {"intent": label, "intent_source": "llm"}


# ── dynamic_prompt — orchestrator system prompt with intent hint ────────────


_ORCHESTRATOR_BASE_PROMPT = """You are OpenMark — Ahmad's personal AI orchestrator.

Your job: take Ahmad's request, plan, delegate to the RIGHT sub-agents, then
return a tight, grounded final answer.

You have ZERO retrieval tools yourself. Every search / fetch goes through
`task_researcher`. Every composition goes through `task_compose_*`. Every
polish or humanize goes through `task_polish` / `task_humanize`. Every
verification goes through `task_verify`. Use `write_todos` to plan first.

CURRENT KNOWLEDGE BASE
- {bookmarks:,} bookmarks across {tags:,} tags, {categories} categories, {communities} communities.
- Stored in Neo4j Graph RAG with pplx-embed 1024-dim vectors.

SUB-AGENTS YOU CAN DELEGATE TO
- task_researcher(brief)          — retrieval + web research, returns anchor list.
- task_compose_linkedin(brief)    — LinkedIn / short-form post (LinkedInPost shape).
- task_compose_essay(brief)       — long-form thesis essay (NewsletterEssay).
- task_compose_roundup(brief)     — categorical news roundup (NewsletterRoundup).
- task_compose_comparison(brief)  — side-by-side comparison (NewsletterComparison).
- task_compose_analytical(brief)  — default analytical newsletter (NewsletterAnalytical).
- task_humanize(brief)            — Arabic / Hebrew rewrite of an existing draft.
- task_polish(brief)              — English AI-tell scrub of an existing draft.
- task_verify(brief)              — grade a composer output (VerificationReport).
- task_author_skill(brief)        — bake a recurring prompt into a reusable skill.

THE STANDARD COMPOSE LOOP (when intent is newsletter or composer-shaped)
1. write_todos to plan the run.
2. task_researcher with the topic + time window + angle.
3. task_compose_<format> with topic + angle + language + researcher anchors verbatim.
4. If language is 'en' -> task_polish; else (ar-*/he) -> task_humanize.
5. task_verify with the composer output + seen_urls list.
6. If verify.overall_passed=False, retry task_compose_<format> ONCE with verifier
   fix_instructions, then re-verify.
7. Return the final composer output verbatim to the user.

CITATION DISCIPLINE
- Every URL in your final answer must have appeared in a researcher tool result.
- Never invent a URL. Never paraphrase a URL.

OUTPUT
- Be direct. No 'Sure!', no 'Here's what I found:'. Get to the answer.
- For bookmark lists: numbered list with `Title — URL — short why`.
- For analytical / research questions: 2-4 sentence summary + citations block.
- For composer outputs: return the composer's structured output verbatim AND a
  user-facing markdown summary above it.
"""


_INTENT_HINTS: dict[str, str] = {
    "fast":       "INTENT: fast. Single task_researcher call, surface 5-10 bookmarks. No composers.",
    "deep":       "INTENT: deep. Multi-angle research via 1-2 task_researcher calls, then a concise synthesis.",
    "newsletter": "INTENT: newsletter. Full compose loop: researcher -> compose_<format> -> polish/humanize -> verify -> return.",
    "digest":     "INTENT: digest. task_researcher with time window, group by category, compact bulleted output.",
    "dive":       "INTENT: dive. task_researcher with the URL + 'graph_expand and SIMILAR_TO neighbors'. Return structured neighborhood.",
}


@dynamic_prompt
def dynamic_orchestrator_prompt(request: ModelRequest) -> str:
    intent = request.state.get("intent") or "fast"
    try:
        from openmark.stores import neo4j_store
        s = neo4j_store.get_stats()
        bookmarks = s.get("bookmarks", 0)
        tags = s.get("tags", 0)
        categories = s.get("categories", 0) or 19
        communities = s.get("communities", 0)
    except Exception:
        bookmarks, tags, categories, communities = 13000, 5000, 19, 0
    base = _ORCHESTRATOR_BASE_PROMPT.format(
        bookmarks=bookmarks, tags=tags, categories=categories, communities=communities,
    )
    hint = _INTENT_HINTS.get(intent, _INTENT_HINTS["fast"])
    return base + "\n\n" + hint
