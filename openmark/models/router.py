"""
Role → model id router.

Single source of truth for which model each role uses, with .env overrides
respected. Roles:
    orchestrator    — frontier reasoning, long context. gpt-5.5 by default.
    summarizer      — cheap fast non-reasoning, OK at structured extraction.
                      gpt-4.1-mini by default.
    classifier      — cheapest fast non-reasoning. gpt-4.1-mini default.
    researcher      — frontier reasoning + tool-use heavy. gpt-5.3-codex default.
    composer        — frontier reasoning + long output. gpt-5.5 default.
    humanizer       — fluent multilingual, medium reasoning. gpt-5 default.
    polisher        — fast non-reasoning English editing. gpt-4.1-mini default.
    verifier        — structured-output reasoning. gpt-5-mini default.
    skill_author    — cheap non-reasoning, just writes a recipe. gpt-4.1-mini default.

Every role honors an env override `OPENMARK_MODEL_<ROLE>` and the legacy
`AZURE_DEPLOYMENT_<ROLE>` so existing .env keeps working.
"""

from __future__ import annotations

import os
from typing import Literal

Role = Literal[
    "orchestrator",
    "summarizer",
    "classifier",
    "researcher",
    "composer",
    "humanizer",
    "polisher",
    "verifier",
    "skill_author",
]


# Default model ids per role. Sourced from openmark/models/foundry.py BANK.
ROLE_DEFAULTS: dict[Role, str] = {
    "orchestrator": "gpt-5.5",
    "summarizer":   "gpt-4.1-mini",
    "classifier":   "gpt-4.1-mini",
    "researcher":   "gpt-5.3-codex",
    "composer":     "gpt-5.5",
    "humanizer":    "gpt-5",
    "polisher":     "gpt-4.1-mini",
    "verifier":     "gpt-5-mini",
    "skill_author": "gpt-4.1-mini",
}


# Legacy env var names from the v2 setup we want to honor.
_LEGACY_ENV: dict[Role, str] = {
    "orchestrator": "AZURE_DEPLOYMENT_EXECUTOR",
    "summarizer":   "AZURE_DEPLOYMENT_CLASSIFIER",
    "classifier":   "AZURE_DEPLOYMENT_CLASSIFIER",
    "researcher":   "AZURE_DEPLOYMENT_EXECUTOR",
    "composer":     "AZURE_DEPLOYMENT_SYNTHESIZER",
    "humanizer":    "AZURE_DEPLOYMENT_SYNTHESIZER",
    "polisher":     "AZURE_DEPLOYMENT_CLASSIFIER",
    "verifier":     "AZURE_DEPLOYMENT_PLANNER",
    "skill_author": "AZURE_DEPLOYMENT_CLASSIFIER",
}


def role_model_id(role: Role) -> str:
    """
    Resolve a role to a Foundry deployment id.

    Resolution order (first non-empty wins):
        1. OPENMARK_MODEL_<ROLE>      (new, explicit override)
        2. legacy AZURE_DEPLOYMENT_*  (back-compat with existing .env)
        3. ROLE_DEFAULTS[role]
    """
    explicit = os.environ.get(f"OPENMARK_MODEL_{role.upper()}")
    if explicit:
        return explicit
    legacy_key = _LEGACY_ENV.get(role)
    if legacy_key:
        legacy_val = os.environ.get(legacy_key)
        if legacy_val:
            return legacy_val
    return ROLE_DEFAULTS[role]
