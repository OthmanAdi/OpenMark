from __future__ import annotations


def test_dynamic_prompt_includes_model_identity(monkeypatch):
    from openmark.agent.classification import dynamic_orchestrator_prompt

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

    assert "MODEL IDENTITY" in prompt
    assert "orchestrator model=" in prompt
