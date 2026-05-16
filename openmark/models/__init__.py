"""Foundry-served frontier model registry + role-based router."""

from openmark.models.foundry import (
    BANK,
    ModelSpec,
    get,
    context_window,
    max_output,
    supports_reasoning,
    pricing,
    list_ids,
)
from openmark.models.router import (
    ROLE_DEFAULTS,
    role_model_id,
)

__all__ = [
    "BANK",
    "ModelSpec",
    "get",
    "context_window",
    "max_output",
    "supports_reasoning",
    "pricing",
    "list_ids",
    "ROLE_DEFAULTS",
    "role_model_id",
]
