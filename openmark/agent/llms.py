"""
LLM factory for OpenMark agent — codex 5.3 only, reasoning=high, thinking visible.

Design choices (locked in by user 2026-05-10):
  - ALL roles use gpt-5.3-codex via Responses API
  - reasoning.effort=high everywhere
  - reasoning.summary=detailed → thinking always visible in UI
  - text.verbosity=medium for human-facing nodes
  - No silent fallback. If codex 5.3 is not available, raise hard so the user sees it.

Azure constraints (May 2026):
  - gpt-5.x-codex is Responses-API ONLY → use_responses_api=True
  - reasoning models reject temperature/top_p/max_tokens → use max_completion_tokens
  - 'minimal' effort disables parallel tool calls → NOT used here (we want high anyway)
"""

from langchain_openai import AzureChatOpenAI
from openmark import config


def _base_kwargs():
    """Auth + endpoint, shared across all roles."""
    return dict(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
        use_responses_api=True,   # codex = Responses only
    )


def _codex_kwargs(effort: str, verbosity: str = "medium"):
    """Reasoning + verbosity for codex models. summary=detailed → thinking always shown."""
    return {
        "reasoning": {"effort": effort, "summary": "detailed"},
        "text":      {"verbosity": verbosity},
    }


def build_planner():
    """Strategy planner — codex 5.3, reasoning=high, detailed thinking surfaced."""
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_PLANNER,
        model_kwargs=_codex_kwargs(effort=config.AZURE_REASONING_PLANNER, verbosity="low"),
        **_base_kwargs(),
    )


def build_executor():
    """Tool-call executor — codex 5.3, reasoning=high. Parallel tools active (we never use 'minimal')."""
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_EXECUTOR,
        model_kwargs=_codex_kwargs(effort=config.AZURE_REASONING_EXECUTOR, verbosity="low"),
        streaming=True,
        **_base_kwargs(),
    )


def build_synthesizer():
    """Grounded synthesis with structured output — codex 5.3, reasoning=high, verbose enough to cite."""
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_SYNTHESIZER,
        model_kwargs=_codex_kwargs(
            effort=config.AZURE_REASONING_SYNTHESIZER,
            verbosity=config.AZURE_VERBOSITY_SYNTHESIZER,
        ),
        streaming=True,
        **_base_kwargs(),
    )


def build_default():
    """Single LLM for the legacy v1 ReAct agent — codex 5.3 high. No fallback."""
    return build_executor()


def build_classifier():
    """
    Cheap, low-latency tier for query classification (fast vs deep vs newsletter vs digest vs dive).
    Defaults to gpt-5-mini. Can be pointed at Azure `model-router` deployment by setting
    AZURE_DEPLOYMENT_CLASSIFIER=model-router in .env — Foundry will then pick the smallest
    viable model per call automatically.
    """
    deployment = config.AZURE_DEPLOYMENT_CLASSIFIER
    is_reasoning = deployment.startswith("gpt-5") or "codex" in deployment.lower()

    if is_reasoning:
        return AzureChatOpenAI(
            azure_deployment=deployment,
            model_kwargs=_codex_kwargs(effort=config.AZURE_REASONING_CLASSIFIER, verbosity="low"),
            **_base_kwargs(),
        )
    # Chat-completion path for non-reasoning routers (model-router, gpt-4o, etc.)
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
        temperature=0,
        max_tokens=120,
    )
