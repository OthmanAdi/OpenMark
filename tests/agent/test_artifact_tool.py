"""Obsidian artifact writer tests, no LLM, no network."""


def test_write_obsidian_artifact_creates_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMARK_OBSIDIAN_ARTIFACT_DIR", str(tmp_path))

    from openmark.agent.tools import write_obsidian_artifact

    result = write_obsidian_artifact.invoke({
        "title": "Money Scout Plan",
        "markdown_body": "## Summary\n\nA practical plan.\n\n## Citations\n\n1. https://example.com",
        "sources": ["https://example.com"],
        "tags": ["Money", "AI Sprint"],
    })

    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert result.startswith("[artifact:written]")
    assert "Absolute path:" in result
    assert "---" in content
    assert "title: \"Money Scout Plan\"" in content
    assert "# Money Scout Plan" in content
    assert "## Summary" in content
    assert "## Citations" in content
    assert "## Artifact Metadata" in content
    assert "ai-sprint" in content


def test_write_obsidian_artifact_adds_resources_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMARK_OBSIDIAN_ARTIFACT_DIR", str(tmp_path))

    from openmark.agent.tools import write_obsidian_artifact

    write_obsidian_artifact.invoke({
        "title": "No Citations Yet",
        "markdown_body": "# No Citations Yet\n\nBody only.",
        "sources": ["https://source-one.test", "https://source-two.test"],
        "tags": [],
    })

    content = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert "## Resources" in content
    assert "https://source-one.test" in content
    assert "https://source-two.test" in content


def test_write_obsidian_artifact_rejects_empty_body(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMARK_OBSIDIAN_ARTIFACT_DIR", str(tmp_path))

    from openmark.agent.tools import write_obsidian_artifact

    result = write_obsidian_artifact.invoke({
        "title": "Empty",
        "markdown_body": "",
        "sources": [],
        "tags": [],
    })

    assert result.startswith("[artifact:error]")
    assert list(tmp_path.glob("*.md")) == []
