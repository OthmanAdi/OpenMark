"""
Composer module — slim renderer surface only.

The v2 orchestrator + sub-agent system was replaced by the v3 chat-tab
orchestrator (openmark.agent.graph). What remains here is the pure-stdlib
render layer that converts composer Pydantic output (LinkedInPost, Newsletter*)
into markdown / LinkedIn plaintext / LinkedIn HTML.
"""

from openmark.composer.export import (
    ComposerOutput,
    to_linkedin_html,
    to_linkedin_plaintext,
    to_markdown,
)

__all__ = [
    "ComposerOutput",
    "to_markdown",
    "to_linkedin_plaintext",
    "to_linkedin_html",
]
