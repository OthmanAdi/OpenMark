from __future__ import annotations


def test_preference_memory_blocks_secret_keys(monkeypatch):
    monkeypatch.setenv("OPENMARK_AGENT_MEMORY", "0")
    from openmark.agent.memory import remember_preference

    assert "disabled" in remember_preference.invoke({"key": "theme", "value": "dark"})


def test_preference_memory_key_validation(monkeypatch):
    monkeypatch.setenv("OPENMARK_AGENT_MEMORY", "1")
    from openmark.agent import memory

    class FakeStore:
        def put(self, *args, **kwargs):
            raise AssertionError("invalid key should not write")

    monkeypatch.setattr(memory, "get_store", lambda: FakeStore())
    out = memory.remember_preference.invoke({"key": "bad key with spaces", "value": "x"})
    assert out.startswith("BLOCKED")


def test_preference_memory_writes_safe_value(monkeypatch):
    monkeypatch.setenv("OPENMARK_AGENT_MEMORY", "1")
    from openmark.agent import memory

    writes = []

    class FakeStore:
        def put(self, *args, **kwargs):
            writes.append((args, kwargs))

    monkeypatch.setattr(memory, "get_store", lambda: FakeStore())
    out = memory.remember_preference.invoke({"key": "newsletter_format", "value": "analytical"})
    assert "Remembered preference" in out
    assert writes
    assert writes[0][0][1] == "newsletter_format"


def test_orchestrator_wires_preference_store(monkeypatch):
    import openmark.agent.tools as _t
    _t.warm_up = lambda: None

    captured = {}
    import openmark.agent.graph as graph_mod

    def _spy(*args, **kw):
        captured["kwargs"] = kw
        class _Stub:
            def invoke(self, *a, **k): pass
            def stream(self, *a, **k): pass
            def get_state(self, *a, **k): pass
        return _Stub()

    monkeypatch.setattr(graph_mod, "create_agent", _spy)
    monkeypatch.setattr(graph_mod, "get_store", lambda: "STORE")
    graph_mod.build_agent()
    assert captured["kwargs"]["store"] == "STORE"
    assert any(getattr(t, "name", "") == "remember_preference" for t in captured["kwargs"]["tools"])
