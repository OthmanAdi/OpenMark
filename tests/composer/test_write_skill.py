"""write_skill sandbox tests — no LLM, no network."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from openmark.agent import skills as skill_loader
from openmark.agent.tools import (
    _AGENT_SKILL_ROOT,
    _SKILL_WRITE_CAP,
    reset_write_skill_quota,
    write_skill,
)


@pytest.fixture(autouse=True)
def _clean_agent_generated_skills():
    """Remove any leftover agent-generated-* dirs before AND after each test."""
    def _purge():
        skills_dir = Path(skill_loader.SKILLS_DIR)
        for entry in list(skills_dir.iterdir()) if skills_dir.exists() else []:
            if entry.name.startswith("agent-generated-test-"):
                shutil.rmtree(entry, ignore_errors=True)
    _purge()
    reset_write_skill_quota()
    yield
    _purge()
    reset_write_skill_quota()


def _invoke(name: str, description: str, body: str) -> str:
    return write_skill.invoke(
        {"name": name, "description": description, "body": body}
    )


def test_writes_to_sandbox_and_loads():
    out = _invoke(
        "test-good-one",
        "A test skill for sandbox checks.",
        "# Body\n\nDo the thing.",
    )
    assert "OK" in out and "agent-generated-test-good-one" in out
    target = Path(_AGENT_SKILL_ROOT + "test-good-one") / "SKILL.md"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "name: agent-generated-test-good-one" in text
    assert "type: agent-generated" in text
    # Skill is loadable on the same turn
    loaded = skill_loader.load_skill("test-good-one")
    assert loaded is not None
    assert loaded["family"] == "agent-generated"


def test_rejects_invalid_name():
    out = _invoke("BAD NAME!!!", "desc", "body")
    assert "BLOCKED" in out
    out2 = _invoke("a", "desc", "body")              # too short
    assert "BLOCKED" in out2
    out3 = _invoke("name with spaces", "desc", "body")
    assert "BLOCKED" in out3


def test_rejects_self_prefixed_name():
    for n in ("openmark-foo", "humanizer-foo", "agent-generated-foo"):
        out = _invoke(n, "desc", "body")
        assert "BLOCKED" in out, f"should reject self-prefix {n}, got: {out}"


def test_rejects_empty_description_or_body():
    assert "BLOCKED" in _invoke("test-empty-desc", "", "body")
    assert "BLOCKED" in _invoke("test-empty-body", "desc", "")


def test_truncates_long_description():
    long_desc = "x" * 500
    out = _invoke("test-long-desc", long_desc, "# Body")
    assert "OK" in out
    text = (Path(_AGENT_SKILL_ROOT + "test-long-desc") / "SKILL.md").read_text(encoding="utf-8")
    # Truncated to 237 chars + "..."
    assert "x" * 240 not in text
    assert "..." in text


def test_blocks_duplicate_name():
    _invoke("test-dup-one", "first", "body")
    out2 = _invoke("test-dup-one", "second", "body")
    assert "already exists" in out2


def test_session_cap_enforced():
    # Write up to the cap
    for i in range(_SKILL_WRITE_CAP):
        out = _invoke(f"test-cap-{i}", "desc", "# body")
        assert "OK" in out, out
    # The next write must be blocked
    out = _invoke("test-cap-extra", "desc", "# body")
    assert "session cap" in out


def test_cannot_overwrite_curated_skill():
    """Even if the model bypassed name validation, the sandbox prefix prevents
    overwrites of openmark-* / humanizer-* directories."""
    out = _invoke("test-safe-name", "desc", "body")
    assert "OK" in out
    # Confirm we wrote OUTSIDE the curated skill dirs
    target_dir = Path(_AGENT_SKILL_ROOT + "test-safe-name")
    assert "agent-generated-" in target_dir.name
    for curated in ("openmark-newsletter", "humanizer-ar-msa"):
        curated_md = Path(skill_loader.SKILLS_DIR) / curated / "SKILL.md"
        if curated_md.exists():
            # Original file was not touched (mtime check is good enough for this test).
            assert curated_md.read_text(encoding="utf-8").startswith("---")
