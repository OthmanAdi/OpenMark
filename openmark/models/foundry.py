"""
Foundry-served frontier model registry.

Source of truth: models.dev (snapshot 2026-05-16). Cross-verified against
provider announcements (OpenAI gpt-5.5, Anthropic claude-opus-4-7, xAI grok-4.3).

Records are static dataclasses keyed by Foundry deployment id. Bank covers
OpenAI, Anthropic, xAI, DeepSeek, Meta, Mistral. Re-runnable extractor lives in
research/agent_v3/raw/03_extract_models_bank.py — re-run when models.dev updates.

Helpers:
    get(model_id)              -> ModelSpec
    context_window(model_id)   -> int
    max_output(model_id)       -> int
    supports_reasoning(model_id) -> bool
    pricing(model_id)          -> (input_per_1m_usd, output_per_1m_usd)
    list_ids()                 -> list[str]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Vendor = Literal["openai", "anthropic", "xai", "deepseek", "meta", "mistral"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    vendor: Vendor
    context_window: int
    max_output: int
    reasoning: bool
    tool_use: bool
    modalities_in: tuple[str, ...]
    modalities_out: tuple[str, ...]
    price_in_per_1m: float | None
    price_out_per_1m: float | None
    release: str | None
    notes: str = ""

    @property
    def is_long_context(self) -> bool:
        return self.context_window >= 400_000

    @property
    def is_cheap(self) -> bool:
        if self.price_in_per_1m is None:
            return False
        return self.price_in_per_1m <= 0.5


def _spec(**kw) -> ModelSpec:
    kw.setdefault("notes", "")
    kw["modalities_in"] = tuple(kw["modalities_in"])
    kw["modalities_out"] = tuple(kw["modalities_out"])
    return ModelSpec(**kw)


BANK: dict[str, ModelSpec] = {
    # ── OpenAI ──────────────────────────────────────────────────────────────
    "gpt-5.5": _spec(
        id="gpt-5.5", vendor="openai",
        context_window=1_050_000, max_output=128_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=5.0, price_out_per_1m=30.0,
        release="2026-04-23",
    ),
    "gpt-5.3-chat-latest": _spec(
        id="gpt-5.3-chat-latest", vendor="openai",
        context_window=128_000, max_output=16_384,
        reasoning=False, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=1.75, price_out_per_1m=14.0,
        release="2026-03-03",
    ),
    "gpt-5.3-codex": _spec(
        id="gpt-5.3-codex", vendor="openai",
        context_window=400_000, max_output=128_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=1.75, price_out_per_1m=14.0,
        release="2026-02-05",
    ),
    "gpt-5": _spec(
        id="gpt-5", vendor="openai",
        context_window=400_000, max_output=128_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=1.25, price_out_per_1m=10.0,
        release="2025-08-07",
    ),
    "gpt-5-mini": _spec(
        id="gpt-5-mini", vendor="openai",
        context_window=400_000, max_output=128_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=0.25, price_out_per_1m=2.0,
        release="2025-08-07",
    ),
    "gpt-5-nano": _spec(
        id="gpt-5-nano", vendor="openai",
        context_window=400_000, max_output=128_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=0.05, price_out_per_1m=0.4,
        release="2025-08-07",
    ),
    "gpt-4.1": _spec(
        id="gpt-4.1", vendor="openai",
        context_window=1_047_576, max_output=32_768,
        reasoning=False, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=2.0, price_out_per_1m=8.0,
        release="2025-04-14",
    ),
    "gpt-4.1-mini": _spec(
        id="gpt-4.1-mini", vendor="openai",
        context_window=1_047_576, max_output=32_768,
        reasoning=False, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=0.4, price_out_per_1m=1.6,
        release="2025-04-14",
    ),
    "gpt-4o": _spec(
        id="gpt-4o", vendor="openai",
        context_window=128_000, max_output=16_384,
        reasoning=False, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=2.5, price_out_per_1m=10.0,
        release="2024-05-13",
    ),
    "gpt-4o-mini": _spec(
        id="gpt-4o-mini", vendor="openai",
        context_window=128_000, max_output=16_384,
        reasoning=False, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=0.15, price_out_per_1m=0.6,
        release="2024-07-18",
    ),
    "o1": _spec(
        id="o1", vendor="openai",
        context_window=200_000, max_output=100_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=15.0, price_out_per_1m=60.0,
        release="2024-12-05",
    ),
    "o3": _spec(
        id="o3", vendor="openai",
        context_window=200_000, max_output=100_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=2.0, price_out_per_1m=8.0,
        release="2025-04-16",
    ),
    "o3-mini": _spec(
        id="o3-mini", vendor="openai",
        context_window=200_000, max_output=100_000,
        reasoning=True, tool_use=True,
        modalities_in=["text"], modalities_out=["text"],
        price_in_per_1m=1.1, price_out_per_1m=4.4,
        release="2024-12-20",
    ),
    "o1-pro": _spec(
        id="o1-pro", vendor="openai",
        context_window=200_000, max_output=100_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=150.0, price_out_per_1m=600.0,
        release="2025-03-19",
    ),
    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-opus-4-7": _spec(
        id="claude-opus-4-7", vendor="anthropic",
        context_window=1_000_000, max_output=128_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=5.0, price_out_per_1m=25.0,
        release="2026-04-16",
    ),
    "claude-sonnet-4-6": _spec(
        id="claude-sonnet-4-6", vendor="anthropic",
        context_window=1_000_000, max_output=64_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=3.0, price_out_per_1m=15.0,
        release="2026-02-17",
    ),
    "claude-haiku-4-5": _spec(
        id="claude-haiku-4-5", vendor="anthropic",
        context_window=200_000, max_output=64_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image", "pdf"], modalities_out=["text"],
        price_in_per_1m=1.0, price_out_per_1m=5.0,
        release="2025-10-15",
    ),
    # ── xAI ──────────────────────────────────────────────────────────────────
    "grok-4.3": _spec(
        id="grok-4.3", vendor="xai",
        context_window=1_000_000, max_output=30_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=1.25, price_out_per_1m=2.5,
        release="2026-05-01",
        notes="Tier kicker: >200K tokens charges $2.5 in / $5 out (2x).",
    ),
    "grok-4.20-0309-reasoning": _spec(
        id="grok-4.20-0309-reasoning", vendor="xai",
        context_window=2_000_000, max_output=30_000,
        reasoning=True, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=2.0, price_out_per_1m=6.0,
        release="2026-03-09",
    ),
    "grok-4.20-0309-non-reasoning": _spec(
        id="grok-4.20-0309-non-reasoning", vendor="xai",
        context_window=2_000_000, max_output=30_000,
        reasoning=False, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=2.0, price_out_per_1m=6.0,
        release="2026-03-09",
    ),
    # ── DeepSeek ────────────────────────────────────────────────────────────
    "deepseek-reasoner": _spec(
        id="deepseek-reasoner", vendor="deepseek",
        context_window=1_000_000, max_output=384_000,
        reasoning=True, tool_use=True,
        modalities_in=["text"], modalities_out=["text"],
        price_in_per_1m=0.14, price_out_per_1m=0.28,
        release="2025-12-01",
    ),
    "deepseek-chat": _spec(
        id="deepseek-chat", vendor="deepseek",
        context_window=1_000_000, max_output=384_000,
        reasoning=False, tool_use=True,
        modalities_in=["text"], modalities_out=["text"],
        price_in_per_1m=0.14, price_out_per_1m=0.28,
        release="2025-12-01",
    ),
    # ── Meta ─────────────────────────────────────────────────────────────────
    "llama-4-maverick-17b-128e-instruct-fp8": _spec(
        id="llama-4-maverick-17b-128e-instruct-fp8", vendor="meta",
        context_window=128_000, max_output=4_096,
        reasoning=False, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        release="2025-04-05",
        notes="First-party price 0; actual cost depends on hosting.",
    ),
    "llama-4-scout-17b-16e-instruct-fp8": _spec(
        id="llama-4-scout-17b-16e-instruct-fp8", vendor="meta",
        context_window=128_000, max_output=4_096,
        reasoning=False, tool_use=True,
        modalities_in=["text", "image"], modalities_out=["text"],
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        release="2025-04-05",
        notes="First-party price 0; actual cost depends on hosting.",
    ),
    # ── Mistral ──────────────────────────────────────────────────────────────
    "mistral-large-2411": _spec(
        id="mistral-large-2411", vendor="mistral",
        context_window=131_072, max_output=16_384,
        reasoning=False, tool_use=True,
        modalities_in=["text"], modalities_out=["text"],
        price_in_per_1m=2.0, price_out_per_1m=6.0,
        release="2024-11-01",
    ),
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def get(model_id: str) -> ModelSpec | None:
    return BANK.get(model_id)


def list_ids() -> list[str]:
    return sorted(BANK.keys())


def context_window(model_id: str) -> int | None:
    spec = BANK.get(model_id)
    return spec.context_window if spec else None


def max_output(model_id: str) -> int | None:
    spec = BANK.get(model_id)
    return spec.max_output if spec else None


def supports_reasoning(model_id: str) -> bool:
    spec = BANK.get(model_id)
    return bool(spec and spec.reasoning)


def pricing(model_id: str) -> tuple[float, float] | None:
    spec = BANK.get(model_id)
    if spec is None or spec.price_in_per_1m is None or spec.price_out_per_1m is None:
        return None
    return spec.price_in_per_1m, spec.price_out_per_1m
