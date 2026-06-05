from __future__ import annotations


def test_agent_config_rejects_secret_values():
    from openmark.agent_config import validate_agent_config

    ok, errors, _ = validate_agent_config({"OPENMARK_AGENT_SYSTEM_PROMPT": "remember token abc"})
    assert not ok
    assert any("secrets" in e for e in errors)


def test_agent_config_save_preserves_unrelated_keys(tmp_path, monkeypatch):
    import openmark.agent_config as cfg

    env_path = tmp_path / ".env"
    env_path.write_text("AZURE_API_KEY=secret\nOPENMARK_ORCH_MODEL_CALL_LIMIT=30\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "ENV_PATH", env_path)

    values = cfg.get_agent_config_values()
    values["OPENMARK_ORCH_MODEL_CALL_LIMIT"] = "12"
    ok, msg = cfg.save_agent_config(values)

    text = env_path.read_text(encoding="utf-8")
    assert ok, msg
    assert "AZURE_API_KEY=secret" in text
    assert "OPENMARK_ORCH_MODEL_CALL_LIMIT=12" in text
    assert "OPENMARK_AGENT_SYSTEM_PROMPT=" in text


def test_dynamic_prompt_includes_env_prompt(monkeypatch):
    from openmark.agent.classification import dynamic_orchestrator_prompt

    monkeypatch.setenv("OPENMARK_AGENT_SYSTEM_PROMPT", "Always answer with concise bullets.")

    class FakeRequest:
        state = {"intent": "fast", "complex": False, "named_skill": None}

        def override(self, *, system_message):
            self.system_message = system_message
            return self

    captured = {}

    def handler(request):
        captured["system_message"] = request.system_message

    dynamic_orchestrator_prompt.wrap_model_call(FakeRequest(), handler)

    prompt = captured["system_message"].content
    assert "User-configured Agent Instructions" in prompt
    assert "Always answer with concise bullets." in prompt
