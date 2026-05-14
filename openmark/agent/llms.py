"""
LLM factory for OpenMark agent.

Two routes:
  AGENT_PROVIDER=azure  → AzureChatOpenAI on gpt-5.3-codex via Responses API,
                          reasoning=high, summary=detailed (production default).
  AGENT_PROVIDER=local  → ChatOpenAI pointed at BONSAI_URL (OpenAI-compatible
                          server: llama.cpp / vLLM / Ollama / LM Studio).
                          No Azure call is made, even as a fallback.
                          Verify in stdout: "[llms] route=local ..." on first call.

To run Hermes-3-Llama-3.1-8B locally:
    huggingface-cli download bartowski/Hermes-3-Llama-3.1-8B-GGUF \
        Hermes-3-Llama-3.1-8B-Q4_K_M.gguf --local-dir ./models/hermes-3-8b
    llama-server -m ./models/hermes-3-8b/Hermes-3-Llama-3.1-8B-Q4_K_M.gguf \
        --port 8080 --host 127.0.0.1 -c 8192 --jinja
    # in .env:
    AGENT_PROVIDER=local
    BONSAI_URL=http://localhost:8080/v1
    BONSAI_MODEL=hermes-3-llama-3.1-8b
The --jinja flag is REQUIRED for Hermes's tool-call template translation.
"""

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from openmark import config


def _is_local() -> bool:
    return (getattr(config, "AGENT_PROVIDER", "azure") or "azure").lower() == "local"


def _local_chat(
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    streaming: bool = True,
) -> ChatOpenAI:
    """OpenAI-compatible client for a local llama.cpp / vLLM / Ollama server."""
    print(f"[llms] route=local url={config.BONSAI_URL} model={config.BONSAI_MODEL}")
    return ChatOpenAI(
        base_url=config.BONSAI_URL,
        api_key="local",          # llama-server ignores it; placeholder to satisfy SDK
        model=config.BONSAI_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )


# ── Azure-only helpers ────────────────────────────────────────────────────────

def _azure_base_kwargs():
    """Auth + endpoint, shared across Azure roles."""
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


# ── Role builders — each one branches on AGENT_PROVIDER ───────────────────────

def build_planner():
    """Strategy planner — Azure: codex 5.3 high. Local: same chat client as executor."""
    if _is_local():
        return _local_chat(temperature=0.2)
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_PLANNER,
        model_kwargs=_codex_kwargs(effort=config.AZURE_REASONING_PLANNER, verbosity="low"),
        **_azure_base_kwargs(),
    )


def build_executor():
    """Tool-call executor."""
    if _is_local():
        return _local_chat(temperature=0.3, streaming=True)
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_EXECUTOR,
        model_kwargs=_codex_kwargs(effort=config.AZURE_REASONING_EXECUTOR, verbosity="low"),
        streaming=True,
        **_azure_base_kwargs(),
    )


def build_synthesizer():
    """Grounded synthesis — Azure: codex 5.3 high. Local: shares the executor model."""
    if _is_local():
        return _local_chat(temperature=0.3, streaming=True)
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_SYNTHESIZER,
        model_kwargs=_codex_kwargs(
            effort=config.AZURE_REASONING_SYNTHESIZER,
            verbosity=config.AZURE_VERBOSITY_SYNTHESIZER,
        ),
        streaming=True,
        **_azure_base_kwargs(),
    )


def build_default():
    """Single LLM for the legacy v1 ReAct agent."""
    return build_executor()


def build_classifier():
    """
    Cheap, low-latency tier for query classification (fast / deep / newsletter /
    digest / dive). On local route, we just reuse the same local model since
    Hermes-3-8B is already cheap. On Azure, picks gpt-5-mini or whatever
    AZURE_DEPLOYMENT_CLASSIFIER points at.
    """
    if _is_local():
        return _local_chat(temperature=0, max_tokens=60, streaming=False)

    deployment = config.AZURE_DEPLOYMENT_CLASSIFIER
    is_reasoning = deployment.startswith("gpt-5") or "codex" in deployment.lower()
    if is_reasoning:
        return AzureChatOpenAI(
            azure_deployment=deployment,
            model_kwargs=_codex_kwargs(effort=config.AZURE_REASONING_CLASSIFIER, verbosity="low"),
            **_azure_base_kwargs(),
        )
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
        temperature=0,
        max_tokens=120,
    )
