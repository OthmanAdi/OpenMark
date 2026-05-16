"""
LLM factory for the v3 orchestrator + sub-agent stack.

Single source of truth for Foundry deployments via `openmark.models.role_model_id`.
Every builder honors AGENT_PROVIDER=azure|local but Foundry is the production
path; local is a thin escape hatch using BONSAI_URL / BONSAI_MODEL.

Public API (back-compat with v2 callers):
    build_orchestrator()    — frontier reasoning, long context
    build_classifier()      — fast non-reasoning, cheap
    build_summarizer()      — alias of classifier
    build_researcher()      — tool-heavy reasoning
    build_composer()        — long-output reasoning
    build_humanizer()       — multilingual reasoning
    build_polisher()        — fast English editor
    build_verifier()        — structured-output reasoning
    build_skill_author()    — cheap non-reasoning

Legacy aliases kept so existing imports don't break:
    build_executor()        -> build_orchestrator()
    build_planner()         -> build_orchestrator()
    build_synthesizer()     -> build_composer()
    build_default()         -> build_orchestrator()

Foundry-specific quirks:
- Grok deployments need `model=deployment` in the body (Foundry deserializer
  rejects null `model`). `_azure_grok` handles it.
- gpt-5.x and codex are Responses API; reasoning effort goes in `model_kwargs`
  with `use_responses_api=True`. `_azure_codex` handles it.
- gpt-4.x and o-series chat-completions are vanilla `AzureChatOpenAI`.
"""

from __future__ import annotations

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from openmark import config
from openmark.models import get as get_model_spec
from openmark.models import role_model_id


def _is_local() -> bool:
    return (getattr(config, "AGENT_PROVIDER", "azure") or "azure").lower() == "local"


def _is_grok(deployment: str) -> bool:
    return "grok" in (deployment or "").lower()


def _is_reasoning_model(deployment: str) -> bool:
    """True for gpt-5*, codex, o1, o3, deepseek-reasoner — anything that
    accepts a `reasoning` dict via Responses API."""
    spec = get_model_spec(deployment)
    if spec is not None:
        return spec.reasoning
    # Fallback heuristic when deployment id is not in our bank
    name = (deployment or "").lower()
    return (
        name.startswith("gpt-5")
        or "codex" in name
        or name.startswith("o1")
        or name.startswith("o3")
        or "reasoner" in name
    )


# ── Build paths ─────────────────────────────────────────────────────────────


def _local_chat(*, temperature: float = 0.3, streaming: bool = True) -> ChatOpenAI:
    """OpenAI-compatible client for llama.cpp / Ollama / vLLM running locally."""
    return ChatOpenAI(
        base_url=config.BONSAI_URL,
        api_key="local",
        model=config.BONSAI_MODEL,
        temperature=temperature,
        streaming=streaming,
    )


def _azure_base_kwargs() -> dict:
    return dict(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )


def _azure_grok(deployment: str, *, streaming: bool = True, effort: str = "high") -> AzureChatOpenAI:
    """
    Foundry-hosted xAI Grok over Chat Completions API.

    Quirk: Foundry's deserializer rejects null `model` in the body. We pass
    both `azure_deployment=` (URL path) AND `model=` (body field via model_name)
    so the request passes validation. Reasoning effort is a top-level field on
    AzureChatOpenAI in langchain-openai 1.2+.
    """
    return AzureChatOpenAI(
        azure_deployment=deployment,
        model_name=deployment,           # required for Foundry-hosted partner models
        streaming=streaming,
        reasoning_effort=effort,
        **_azure_base_kwargs(),
    )


def _azure_codex(
    deployment: str,
    *,
    streaming: bool = True,
    effort: str = "high",
    verbosity: str = "medium",
) -> AzureChatOpenAI:
    """
    Foundry-hosted OpenAI reasoning model (gpt-5*, codex, o-series) over the
    Responses API. Uses top-level `reasoning` + `verbosity` fields on
    AzureChatOpenAI (langchain-openai 1.2+ idiom). The SDK auto-routes to the
    Responses API whenever `reasoning=` is set.
    """
    return AzureChatOpenAI(
        azure_deployment=deployment,
        use_responses_api=True,
        streaming=streaming,
        reasoning={"effort": effort, "summary": "detailed"},
        verbosity=verbosity,
        **_azure_base_kwargs(),
    )


def _azure_chat(deployment: str, *, streaming: bool = True, temperature: float = 0.3) -> AzureChatOpenAI:
    """Vanilla Foundry Chat Completions deployment (gpt-4.x, mistral, llama)."""
    return AzureChatOpenAI(
        azure_deployment=deployment,
        streaming=streaming,
        temperature=temperature,
        **_azure_base_kwargs(),
    )


def _build_for_role(role: str, *, effort: str = "high", verbosity: str = "low",
                    streaming: bool = True, temperature: float | None = None):
    """Resolve role -> deployment id -> right AzureChatOpenAI wrapper."""
    if _is_local():
        return _local_chat(temperature=(temperature if temperature is not None else 0.3),
                           streaming=streaming)

    deployment = role_model_id(role)  # type: ignore[arg-type]
    if _is_grok(deployment):
        return _azure_grok(deployment, streaming=streaming, effort=effort)
    if _is_reasoning_model(deployment):
        return _azure_codex(deployment, streaming=streaming, effort=effort, verbosity=verbosity)
    return _azure_chat(deployment, streaming=streaming,
                       temperature=(temperature if temperature is not None else 0.3))


# ── Role builders ────────────────────────────────────────────────────────────


def build_orchestrator():
    """Frontier reasoning model with long context. Drives the chat agent."""
    return _build_for_role("orchestrator", effort="high", verbosity="low")


def build_classifier():
    """Cheap fast non-reasoning model for intent classification + summarization."""
    return _build_for_role("classifier", effort="low", verbosity="low",
                           streaming=False, temperature=0.0)


def build_summarizer():
    """Alias of classifier — same cheap model is fine for history summarization."""
    return build_classifier()


def build_researcher():
    """Tool-heavy reasoning model for the researcher sub-agent."""
    return _build_for_role("researcher", effort="high", verbosity="low")


def build_composer():
    """Long-output reasoning model for composer sub-agents."""
    return _build_for_role("composer", effort="high", verbosity="medium")


def build_humanizer():
    """Multilingual reasoning model for Arabic / Hebrew humanizer sub-agent."""
    return _build_for_role("humanizer", effort="medium", verbosity="medium")


def build_polisher():
    """Fast English editor model for the polisher sub-agent."""
    return _build_for_role("polisher", effort="low", verbosity="low", temperature=0.3)


def build_verifier():
    """Structured-output reasoning model for the verifier sub-agent."""
    return _build_for_role("verifier", effort="medium", verbosity="low")


def build_skill_author():
    """Cheap non-reasoning model for the skill-author sub-agent."""
    return _build_for_role("skill_author", effort="low", verbosity="low", temperature=0.3)


# ── Back-compat shims ───────────────────────────────────────────────────────

def build_executor():
    """Legacy alias — v2 callers used build_executor as the main agent LLM."""
    return build_orchestrator()


def build_planner():
    return build_orchestrator()


def build_synthesizer():
    return build_composer()


def build_default():
    return build_orchestrator()
