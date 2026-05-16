"""
Sub-agent registry — exports every `task_*` tool the orchestrator can call.

Each sub-agent lives in its own module under this package. The orchestrator
imports `ALL_SUBAGENT_TOOLS` and merges it with its own tool list.
"""

from openmark.agent.subagents.researcher import task_researcher
from openmark.agent.subagents.composer_linkedin import task_compose_linkedin
from openmark.agent.subagents.composer_essay import task_compose_essay
from openmark.agent.subagents.composer_roundup import task_compose_roundup
from openmark.agent.subagents.composer_comparison import task_compose_comparison
from openmark.agent.subagents.composer_analytical import task_compose_analytical
from openmark.agent.subagents.humanizer import task_humanize
from openmark.agent.subagents.polisher import task_polish
from openmark.agent.subagents.verifier import task_verify
from openmark.agent.subagents.skill_author import task_author_skill


ALL_SUBAGENT_TOOLS = [
    task_researcher,
    task_compose_linkedin,
    task_compose_essay,
    task_compose_roundup,
    task_compose_comparison,
    task_compose_analytical,
    task_humanize,
    task_polish,
    task_verify,
    task_author_skill,
]


# Map composer-format name -> tool name, for the orchestrator's planning prompt
COMPOSER_FORMAT_TO_TOOL = {
    "linkedin":   "task_compose_linkedin",
    "thread":     "task_compose_linkedin",     # alias — same LinkedInPost shape
    "essay":      "task_compose_essay",
    "roundup":    "task_compose_roundup",
    "comparison": "task_compose_comparison",
    "analytical": "task_compose_analytical",
}


__all__ = [
    "ALL_SUBAGENT_TOOLS",
    "COMPOSER_FORMAT_TO_TOOL",
    "task_researcher",
    "task_compose_linkedin",
    "task_compose_essay",
    "task_compose_roundup",
    "task_compose_comparison",
    "task_compose_analytical",
    "task_humanize",
    "task_polish",
    "task_verify",
    "task_author_skill",
]
