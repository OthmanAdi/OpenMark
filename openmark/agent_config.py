"""Safe UI-editable agent configuration stored in the project .env file."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

FieldType = Literal["text", "textarea", "select", "int", "bool"]


@dataclass(frozen=True)
class AgentConfigField:
    key: str
    label: str
    default: str
    type: FieldType = "text"
    section: str = "General"
    help: str = ""
    choices: tuple[str, ...] = ()
    min_value: int | None = None
    max_value: int | None = None


EFFORT_CHOICES = ("minimal", "low", "medium", "high")
VERBOSITY_CHOICES = ("low", "medium", "high")
MODEL_ROLE_DEFAULTS = {
    "OPENMARK_MODEL_ORCHESTRATOR": "gpt-5.5",
    "OPENMARK_MODEL_CLASSIFIER": "gpt-4.1-mini",
    "OPENMARK_MODEL_SUMMARIZER": "gpt-4.1-mini",
    "OPENMARK_MODEL_RESEARCHER": "gpt-5.3-codex",
    "OPENMARK_MODEL_COMPOSER": "gpt-5.5",
    "OPENMARK_MODEL_HUMANIZER": "gpt-5",
    "OPENMARK_MODEL_POLISHER": "gpt-4.1-mini",
    "OPENMARK_MODEL_VERIFIER": "gpt-5-mini",
    "OPENMARK_MODEL_SKILL_AUTHOR": "gpt-4.1-mini",
}


FIELDS: tuple[AgentConfigField, ...] = (
    AgentConfigField("OPENMARK_AGENT_SYSTEM_PROMPT", "System prompt addendum", "", "textarea", "Prompt", "Appended to the orchestrator system prompt. Use this for durable behavior instructions, not secrets."),
    AgentConfigField("OPENMARK_AGENT_MEMORY", "Preference memory", "1", "bool", "Memory", "Enable explicit remember_preference storage."),
    AgentConfigField("OPENMARK_AGENT_MEMORY_DB", "Preference memory DB", str(ROOT / "data" / "openmark_agent_memory.db"), "text", "Memory", "SQLite file for cross-thread remembered preferences."),
    AgentConfigField("AGENT_PROVIDER", "Agent provider", "azure", "select", "Models", "azure uses Foundry deployments. local uses BONSAI_URL and BONSAI_MODEL.", ("azure", "local")),
    AgentConfigField("BONSAI_URL", "Local model base URL", "http://localhost:11434/v1", "text", "Models", "OpenAI-compatible endpoint used when AGENT_PROVIDER=local."),
    AgentConfigField("BONSAI_MODEL", "Local model name", "hermes3:8b", "text", "Models", "Model id used when AGENT_PROVIDER=local."),
    *(AgentConfigField(k, k.replace("OPENMARK_MODEL_", "").replace("_", " ").title(), v, "text", "Role Models", "Foundry deployment/model id for this agent role.") for k, v in MODEL_ROLE_DEFAULTS.items()),
    AgentConfigField("OPENMARK_EFFORT_ORCHESTRATOR", "Orchestrator effort", "medium", "select", "Reasoning", "Main chat planner/reasoner.", EFFORT_CHOICES),
    AgentConfigField("OPENMARK_EFFORT_CLASSIFIER", "Classifier effort", "low", "select", "Reasoning", "Intent and complexity classifier.", EFFORT_CHOICES),
    AgentConfigField("OPENMARK_EFFORT_RESEARCHER", "Researcher effort", "medium", "select", "Reasoning", "Tool-heavy retrieval sub-agent.", EFFORT_CHOICES),
    AgentConfigField("OPENMARK_EFFORT_COMPOSER", "Composer effort", "medium", "select", "Reasoning", "Newsletter and long-form writing sub-agents.", EFFORT_CHOICES),
    AgentConfigField("OPENMARK_EFFORT_HUMANIZER", "Humanizer effort", "medium", "select", "Reasoning", "Arabic/Hebrew humanizer sub-agent.", EFFORT_CHOICES),
    AgentConfigField("OPENMARK_EFFORT_POLISHER", "Polisher effort", "low", "select", "Reasoning", "English AI-tell scrubber.", EFFORT_CHOICES),
    AgentConfigField("OPENMARK_EFFORT_VERIFIER", "Verifier effort", "medium", "select", "Reasoning", "Structured output checker.", EFFORT_CHOICES),
    AgentConfigField("OPENMARK_EFFORT_SKILL_AUTHOR", "Skill author effort", "low", "select", "Reasoning", "Skill authoring helper.", EFFORT_CHOICES),
    AgentConfigField("OPENMARK_VERBOSITY_ORCHESTRATOR", "Orchestrator verbosity", "medium", "select", "Verbosity", "Reasoning model answer verbosity.", VERBOSITY_CHOICES),
    AgentConfigField("OPENMARK_VERBOSITY_CLASSIFIER", "Classifier verbosity", "low", "select", "Verbosity", "Classifier response verbosity.", VERBOSITY_CHOICES),
    AgentConfigField("OPENMARK_VERBOSITY_RESEARCHER", "Researcher verbosity", "low", "select", "Verbosity", "Researcher response verbosity.", VERBOSITY_CHOICES),
    AgentConfigField("OPENMARK_VERBOSITY_COMPOSER", "Composer verbosity", "medium", "select", "Verbosity", "Composer output verbosity.", VERBOSITY_CHOICES),
    AgentConfigField("OPENMARK_VERBOSITY_HUMANIZER", "Humanizer verbosity", "medium", "select", "Verbosity", "Humanizer output verbosity.", VERBOSITY_CHOICES),
    AgentConfigField("OPENMARK_VERBOSITY_POLISHER", "Polisher verbosity", "low", "select", "Verbosity", "Polisher response verbosity.", VERBOSITY_CHOICES),
    AgentConfigField("OPENMARK_VERBOSITY_VERIFIER", "Verifier verbosity", "low", "select", "Verbosity", "Verifier output verbosity.", VERBOSITY_CHOICES),
    AgentConfigField("OPENMARK_VERBOSITY_SKILL_AUTHOR", "Skill author verbosity", "low", "select", "Verbosity", "Skill author response verbosity.", VERBOSITY_CHOICES),
    AgentConfigField("OPENMARK_CONTEXT_EDIT_TRIGGER", "Orchestrator trim trigger", "120000", "int", "Context", "Token threshold before clearing bulky tool-use history.", min_value=1000, max_value=1000000),
    AgentConfigField("OPENMARK_CONTEXT_EDIT_KEEP", "Orchestrator tool calls kept", "4", "int", "Context", "How many recent tool-use blocks to keep after trimming.", min_value=1, max_value=50),
    AgentConfigField("OPENMARK_SUMMARY_TOKEN_TRIGGER", "Summary token trigger", "100000", "int", "Context", "Token threshold before conversation summarization.", min_value=1000, max_value=1000000),
    AgentConfigField("OPENMARK_SUMMARY_MESSAGE_TRIGGER", "Summary message trigger", "80", "int", "Context", "Message count threshold before summarization.", min_value=5, max_value=1000),
    AgentConfigField("OPENMARK_SUMMARY_KEEP_MESSAGES", "Summary messages kept", "24", "int", "Context", "Recent messages preserved after summarization.", min_value=1, max_value=200),
    AgentConfigField("OPENMARK_ORCH_MODEL_CALL_LIMIT", "Orchestrator model-call limit", "30", "int", "Limits", "Hard cap on model calls per turn.", min_value=1, max_value=200),
    AgentConfigField("OPENMARK_ORCH_TOOL_CALL_LIMIT", "Orchestrator tool-call limit", "40", "int", "Limits", "Hard cap on tool calls per turn.", min_value=1, max_value=300),
    AgentConfigField("OPENMARK_COMPOSE_ESSAY_CALL_LIMIT", "Essay composer call limit", "2", "int", "Limits", "Max essay composer calls per turn.", min_value=1, max_value=20),
    AgentConfigField("OPENMARK_COMPOSE_ANALYTICAL_CALL_LIMIT", "Analytical composer call limit", "2", "int", "Limits", "Max analytical composer calls per turn.", min_value=1, max_value=20),
    AgentConfigField("OPENMARK_SUBAGENT_CONTEXT_EDIT_TRIGGER", "Sub-agent trim trigger", "80000", "int", "Sub-agents", "Token threshold before sub-agent context editing.", min_value=1000, max_value=1000000),
    AgentConfigField("OPENMARK_SUBAGENT_CONTEXT_EDIT_KEEP", "Sub-agent tool calls kept", "5", "int", "Sub-agents", "Recent sub-agent tool blocks kept after trimming.", min_value=1, max_value=50),
    AgentConfigField("OPENMARK_SUBAGENT_SUMMARY_TRIGGER", "Sub-agent summary trigger", "40000", "int", "Sub-agents", "Token threshold before sub-agent summarization.", min_value=1000, max_value=1000000),
    AgentConfigField("OPENMARK_SUBAGENT_SUMMARY_KEEP", "Sub-agent messages kept", "12", "int", "Sub-agents", "Recent sub-agent messages kept after summarization.", min_value=1, max_value=100),
    AgentConfigField("OPENMARK_SUBAGENT_MODEL_CALL_LIMIT", "Sub-agent model-call limit", "8", "int", "Sub-agents", "Default model-call cap inside sub-agents.", min_value=1, max_value=100),
    AgentConfigField("OPENMARK_MCP_TRENDRADAR", "TrendRadar MCP", "0", "bool", "Tools", "Adds TrendRadar retrieval tools to researcher when enabled."),
    AgentConfigField("OPENMARK_RERANK", "Cross-encoder rerank", "0", "bool", "Tools", "Enable optional reranking if dependencies are available."),
    AgentConfigField("OPENMARK_OBSIDIAN_ARTIFACT_DIR", "Obsidian artifact directory", str(ROOT / "drafts" / "obsidian"), "text", "Output", "Local folder where write_obsidian_artifact saves Markdown reports."),
    AgentConfigField("OPENMARK_THEME", "UI theme", "light", "select", "UI", "Theme applied on next restart or page reload.", ("light", "dark", "system")),
    AgentConfigField("OPENMARK_PORT", "UI port", "7860", "int", "UI", "Port used by Gradio on restart.", min_value=1, max_value=65535),
)

FIELD_MAP = {f.key: f for f in FIELDS}


def _read_env_lines() -> list[str]:
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(encoding="utf-8").splitlines()


def _parse_env(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in FIELD_MAP:
            values[key] = value.strip().strip('"').strip("'")
    return values


def get_agent_config_values() -> dict[str, str]:
    values = _parse_env(_read_env_lines())
    return {field.key: values.get(field.key, os.getenv(field.key, field.default)) for field in FIELDS}


def grouped_fields() -> dict[str, list[AgentConfigField]]:
    groups: dict[str, list[AgentConfigField]] = {}
    for field in FIELDS:
        groups.setdefault(field.section, []).append(field)
    return groups


def validate_agent_config(values: dict[str, object]) -> tuple[bool, list[str], dict[str, str]]:
    errors: list[str] = []
    cleaned: dict[str, str] = {}
    for field in FIELDS:
        raw = values.get(field.key, field.default)
        value = str(raw if raw is not None else "").strip()
        if field.type == "bool":
            value = "1" if value.lower() in {"1", "true", "yes", "on"} else "0"
        elif field.type == "select":
            if value not in field.choices:
                errors.append(f"{field.label}: choose one of {', '.join(field.choices)}")
        elif field.type == "int":
            try:
                ivalue = int(value)
            except ValueError:
                errors.append(f"{field.label}: must be a number")
                continue
            if field.min_value is not None and ivalue < field.min_value:
                errors.append(f"{field.label}: minimum is {field.min_value}")
            if field.max_value is not None and ivalue > field.max_value:
                errors.append(f"{field.label}: maximum is {field.max_value}")
            value = str(ivalue)
        elif field.key == "OPENMARK_AGENT_SYSTEM_PROMPT" and len(value) > 12000:
            errors.append("System prompt addendum: maximum is 12000 characters")

        if re.search(r"(api[_-]?key|token|cookie|password|secret|li_at|jsessionid)", value, re.I):
            errors.append(f"{field.label}: secrets are not allowed in the Agent Config UI")
        cleaned[field.key] = value
    return not errors, errors, cleaned


def _quote_env_value(value: str) -> str:
    if value == "":
        return ""
    if any(ch in value for ch in " #\t\n\r\""):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return value


def save_agent_config(values: dict[str, object]) -> tuple[bool, str]:
    ok, errors, cleaned = validate_agent_config(values)
    if not ok:
        return False, "Fix these values before saving:\n" + "\n".join(f"- {e}" for e in errors)

    lines = _read_env_lines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        active = bool(stripped and not stripped.startswith("#") and "=" in stripped)
        if active:
            key = line.split("=", 1)[0].strip()
            if key in cleaned:
                out.append(f"{key}={_quote_env_value(cleaned[key])}")
                seen.add(key)
                continue
        out.append(line)

    missing = [field.key for field in FIELDS if field.key not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# -- OpenMark Agent Config UI --")
        for key in missing:
            out.append(f"{key}={_quote_env_value(cleaned[key])}")

    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.environ.update(cleaned)
    return True, "Saved to .env. Prompt addendum applies to new chat turns immediately. Restart OpenMark for model, memory, limit, provider, MCP, theme, and port changes to take effect."
