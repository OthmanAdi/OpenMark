# Azure OpenAI Reasoning Models + LangChain (May 2026)

## 1. What is "codex 5.3" on Foundry?

Almost certainly **`gpt-5.3-codex`**, an Azure Foundry reasoning model (last updated 2026-02-24). The Foundry catalog now ships a full Codex family:

| Model | Release | Notes |
|---|---|---|
| `gpt-5.3-codex` | 2026-02 | Latest stable Codex reasoning model. Limited-access (apply via `aka.ms/OAI/gpt53codexaccess`). |
| `gpt-5.2-codex` | 2026-01 | Stable, Responses-API-only. |
| `gpt-5.1-codex-max` | 2025-12 | Adds `reasoning_effort="xhigh"`. |
| `gpt-5.1-codex`, `gpt-5.1-codex-mini` | 2025-11 | Responses-API only. |
| `gpt-5-codex` | 2025-09 | Original. Does **not** support `reasoning_effort="minimal"`. |
| `codex-mini` (o-series) | 2025-05 | Legacy, Responses-API only. |

The deployment name is user-chosen, so "codex 5.3" in the portal is the user's deployment alias pointing at the `gpt-5.3-codex` base model. Confirm via Foundry → Deployments → check `Model name` column.

## 2. Reasoning depth parameter (raw API)

Two surfaces, two names — same effect:

- **Chat Completions API**: flat kwarg `reasoning_effort="medium"`.
- **Responses API** (preferred for GPT-5 / Codex): nested `reasoning={"effort": "medium", "summary": "auto"}`.

Allowed values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`.

Caveats (from the Foundry reasoning doc, section "API & feature support"):

- `minimal`: only original GPT-5; **not** supported on `gpt-5.1+` or `gpt-5-codex`.
- `xhigh`: only `gpt-5.1-codex-max`.
- `none`: only `gpt-5.1`, `gpt-5.2`, `gpt-5.1-codex*` (skips reasoning entirely, faster).
- `gpt-5-pro`: locked to `high`.
- `gpt-5.1` defaults to `none` — pass effort explicitly when upgrading from o3/gpt-5.
- The codex variants (`gpt-5.1-codex`, `gpt-5.2-codex`, `gpt-5.3-codex`, `gpt-5.1-codex-max`, `gpt-5-codex`) only support the **Responses API**, not Chat Completions.

## 3. LangChain `AzureChatOpenAI` (langchain-openai 1.x)

Two working patterns. Pick by API surface:

### A. Chat Completions, simple effort

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="codex-5-3",          # your Foundry deployment alias
    api_version="2025-04-01-preview",      # or newer; v1 preferred
    azure_endpoint="https://<resource>.openai.azure.com",
    reasoning_effort="medium",             # direct kwarg, accepted in 1.x
)
```

### B. Responses API, full control (recommended for gpt-5-codex)

```python
llm = AzureChatOpenAI(
    azure_deployment="codex-5-3",
    api_version="preview",                 # or "2025-04-01-preview"
    azure_endpoint="https://<resource>.openai.azure.com",
    use_responses_api=True,                # required for codex variants
    model_kwargs={
        "reasoning": {"effort": "high", "summary": "auto"},
        "text": {"verbosity": "low"},      # GPT-5 only, Responses-API only
    },
)
```

Notes:

- `reasoning_effort=` as a top-level kwarg works on `ChatOpenAI` and (after late-2025 fixes) on `AzureChatOpenAI`. If you hit a 404 / "DeploymentNotFound" when passing it (GH issue [#32714](https://github.com/langchain-ai/langchain/issues/32714)), drop into `model_kwargs={"reasoning": {...}}` and set `use_responses_api=True`.
- Per-call override: `llm.invoke(msgs, reasoning={"effort": "high"})` via `with_config(...)`.
- Reasoning summaries land on `response.additional_kwargs["reasoning"]` and (1.x) `response.content_blocks`.

## 4. Verbosity

- Values: `low`, `medium`, `high`.
- Works on GPT-5 / 5.1 / 5.2 / 5.3 series **only via the Responses API** on Azure. Chat Completions rejects it (`Unrecognized request argument supplied: verbosity`).
- Pass as `model_kwargs={"text": {"verbosity": "low"}}` with `use_responses_api=True`.

## 5. Constraints to remember

- **Unsupported on all reasoning models**: `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `logit_bias`, `max_tokens`.
- Use **`max_completion_tokens`** (Chat Completions) or **`max_output_tokens`** (Responses).
- **System messages**: supported on GPT-5 series; treated as developer messages on o-series. Never send both `system` and `developer` in one call.
- **Streaming**: supported on most (not `gpt-5-pro`, `gpt-5-codex` original, `o3-pro`, `o1`). `gpt-5.2-codex` + Responses + tools + reasoning has a known intermittent 500 bug.
- **Parallel tool calls**: disabled when `reasoning_effort="minimal"`; not supported on the o-series at all.
- Codex models require **Responses API** — don't try Chat Completions.

## Sources

- [Azure OpenAI reasoning models (GPT-5 series, o-series)](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning) — Microsoft Learn, updated 2026-05-06
- [Codex with Azure OpenAI in Foundry](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/codex) — Microsoft Learn
- [Foundry catalog: gpt-5.3-codex](https://ai.azure.com/catalog/models/gpt-5.3-codex), [gpt-5.2-codex](https://ai.azure.com/catalog/models/gpt-5.2-codex), [gpt-5-codex](https://ai.azure.com/catalog/models/gpt-5-codex)
- [Does GPT-5 via Azure support reasoning_effort and verbosity?](https://learn.microsoft.com/en-us/answers/questions/5519548/does-gpt-5-via-azure-support-reasoning-effort-and) — Microsoft Q&A
- [LangChain ChatOpenAI integration docs](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [LangChain Python reference: langchain_openai](https://reference.langchain.com/python/integrations/langchain_openai/)
- [GH #32714 — AzureChatOpenAI doesn't accept reasoning parameter](https://github.com/langchain-ai/langchain/issues/32714)
- [LangChain forum: extracting GPT-5 reasoning summaries](https://forum.langchain.com/t/how-to-extract-gpt-5-reasoning-summaries-with-langchain-openai/1802)
- [OpenAI reasoning models guide](https://developers.openai.com/api/docs/guides/reasoning)
