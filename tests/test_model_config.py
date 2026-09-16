"""Provider routing: ollama override, no silent fallbacks, $0 local pricing."""
import os

import pytest

from interview_engine import model_config
from interview_engine.telemetry import price_for


def test_ollama_override_routes(monkeypatch):
    monkeypatch.setenv("INTERVIEW_ENGINE_MODEL_PROBING", "ollama:qwen2.5-coder:14b")
    cfg = model_config.for_stage("probing")
    assert cfg.provider == "ollama" and cfg.model == "qwen2.5-coder:14b"


def test_default_is_local_ollama(monkeypatch):
    # ADR-0018: local-first until production - no stage defaults to a paid provider
    for stage in ("probing", "distillation", "phrasing", "scope_tiebreak"):
        monkeypatch.delenv(f"INTERVIEW_ENGINE_MODEL_{stage.upper()}", raising=False)
        cfg = model_config.for_stage(stage)
        assert cfg.provider == "ollama", stage
        assert cfg.model == "qwen2.5-coder:14b", stage  # his standing default; 7b is the lighter fallback


def test_unknown_stage_raises():
    with pytest.raises(KeyError):
        model_config.for_stage("vibes")


def test_ollama_is_free_not_unpriced():
    assert price_for("qwen2.5-coder:14b", provider="ollama") == (0.0, 0.0)
    with pytest.raises(KeyError):
        price_for("qwen2.5-coder:14b", provider="deepseek")  # unknown paid model still raises


def test_ollama_stage_end_to_end_through_telemetry(monkeypatch):
    """Regression: a stage routed to ollama must log $0.00, never KeyError."""
    from interview_engine.telemetry import UsageLog

    monkeypatch.setenv("INTERVIEW_ENGINE_MODEL_PROBING", "ollama:qwen2.5-coder:7b")
    cfg = model_config.for_stage("probing")
    log = UsageLog("test-session")
    rec = log.record("probing", cfg.provider, cfg.model, {"input_tokens": 5000, "output_tokens": 500})
    assert rec.cost_usd == 0.0
    assert log.total_usd() == 0.0


def test_unpriced_remote_model_still_hard_fails():
    from interview_engine.telemetry import UsageLog

    log = UsageLog("test-session")
    with pytest.raises(KeyError):
        log.record("probing", "openai", "gpt-9000-turbo", {"input_tokens": 1, "output_tokens": 1})


# ---- error handling ----

def test_preflight_names_missing_remote_key(monkeypatch):
    from interview_engine.errors import UserFacingError

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("INTERVIEW_ENGINE_MODEL_PROBING", "deepseek:deepseek-chat")  # remote is opt-in
    with pytest.raises(UserFacingError) as ei:
        model_config.build_chat_model("probing")
    msg = ei.value.render()
    assert "DEEPSEEK_API_KEY" in msg and "ollama" in msg.lower()


def test_ollama_connection_refused_is_friendly():
    import httpx

    from interview_engine.errors import translate_llm_error

    e = translate_llm_error(httpx.ConnectError("connection refused"), "probing", "ollama",
                            "qwen2.5-coder:7b", "http://localhost:11434/v1")
    assert e is not None and "ollama serve" in e.render()


def test_ollama_model_not_pulled_is_friendly():
    from interview_engine.errors import translate_llm_error

    exc = type("NotFoundError", (Exception,), {})("404 model not found")
    # name-based mapping: synthesize with the real class name
    exc.__class__.__name__ = "NotFoundError"
    e = translate_llm_error(exc, "probing", "ollama", "qwen2.5-coder:7b", "http://localhost:11434/v1")
    assert e is not None and "ollama pull qwen2.5-coder:7b" in e.render()


def test_unknown_llm_error_returns_none():
    from interview_engine.errors import translate_llm_error

    assert translate_llm_error(ValueError("weird"), "probing", "deepseek", "deepseek-chat", None) is None


def test_json_error_names_stage_and_fix():
    import json as j

    from interview_engine.errors import translate_json_error

    e = translate_json_error(j.JSONDecodeError("x", "doc", 0), "distillation")
    r = e.render()
    assert "distillation" in r and "stronger model" in r



def test_current_deepseek_flash_peak_price():
    assert price_for("deepseek-flash", provider="deepseek") == (0.30, 1.20)


def test_probing_schema_is_sent_as_strict_response_format(monkeypatch):
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    model = model_config.build_chat_model("probing", schema=schema)
    response_format = model.model_kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema



def test_deepseek_schema_request_avoids_broken_json_mode_and_warns(monkeypatch, capsys):
    monkeypatch.setenv("INTERVIEW_ENGINE_MODEL_PROBING", "deepseek:deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    model = model_config.build_chat_model("probing", schema={"type": "object"})
    assert "response_format" not in model.model_kwargs
    assert model.extra_body == {"thinking": {"type": "disabled"}}
    warning = capsys.readouterr().err
    assert "WARNING output mode [probing]" in warning
    assert "prompt-structured mode (thinking disabled)" in warning
    assert "field shape remains validation+retry guarded" in warning

def test_anthropic_structured_adapter_preserves_raw_metadata():
    from langchain_core.messages import AIMessage
    raw = AIMessage(content="tool call", response_metadata={"stop_reason": "end_turn"},
                    usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
    class Runnable:
        def invoke(self, messages):
            return {"raw": raw, "parsed": {"question": "Who uses it?", "covered": [],
                    "thin": [], "skipped": [], "complete": False}, "parsing_error": None}
    out = model_config._StructuredOutputAdapter(Runnable()).invoke([])
    assert out is raw
    assert out.response_metadata["stop_reason"] == "end_turn"
    assert out.usage_metadata["output_tokens"] == 2
    assert __import__("json").loads(out.content)["question"] == "Who uses it?"


def test_conops_uses_shared_distillation_route_never_its_own_default(monkeypatch, tmp_path):
    """Regression: first live CONOPS run used qwen while every configured stage
    used phi4 because the new stage invented an independent default/key."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "INTERVIEW_ENGINE_MODEL_PROBING=ollama:phi4\n"
        "INTERVIEW_ENGINE_MODEL_DISTILLATION=ollama:phi4\n"
        "INTERVIEW_ENGINE_MODEL_CONOPS=ollama:wrong-stage-specific-model\n"
    )
    monkeypatch.setattr(model_config, "_DOTENV", None)
    monkeypatch.delenv("INTERVIEW_ENGINE_MODEL_DISTILLATION", raising=False)
    monkeypatch.delenv("INTERVIEW_ENGINE_MODEL_CONOPS", raising=False)
    assert model_config.for_stage("distillation").model == "phi4"
    assert model_config.for_stage("conops").model == "phi4"
    assert "conops=ollama/phi4" in model_config.describe_models()


def test_deepseek_debug_http_wires_payload_capture(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("INTERVIEW_ENGINE_MODEL_PROBING", "deepseek:deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("VMODEL_DEBUG_HTTP", "1")
    monkeypatch.setenv("VMODEL_LLM_LOG", str(tmp_path / "llm-calls.jsonl"))
    model = model_config.build_chat_model("probing", schema={"type": "object"})
    assert model.http_client is not None
    assert model.http_client.event_hooks["request"]
    assert model.http_client.event_hooks["response"]
    shown = capsys.readouterr().err
    assert str(tmp_path / "debug-http.jsonl") in shown
    assert "request messages included" in shown


def test_debug_http_startup_warning_is_deduplicated(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("INTERVIEW_ENGINE_MODEL_PROBING", "deepseek:deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("VMODEL_DEBUG_HTTP", "1")
    monkeypatch.setenv("VMODEL_LLM_LOG", str(tmp_path / "llm-calls.jsonl"))
    monkeypatch.setattr(model_config, "_DEBUG_HTTP_ANNOUNCED", set())
    model_config.build_chat_model("probing", schema={"type":"object"})
    model_config.build_chat_model("probing", schema={"type":"object"})
    shown = capsys.readouterr().err
    assert shown.count("full LLM HTTP debug enabled") == 1
