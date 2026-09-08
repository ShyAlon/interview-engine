"""Provider routing: ollama override, no silent fallbacks, $0 local pricing."""
import os

import pytest

from interview_engine import model_config
from interview_engine.telemetry import price_for


def test_ollama_override_routes(monkeypatch):
    monkeypatch.setenv("INTERVIEW_ENGINE_MODEL_PROBING", "ollama:qwen2.5-coder:14b")
    cfg = model_config.for_stage("probing")
    assert cfg.provider == "ollama" and cfg.model == "qwen2.5-coder:14b"


def test_default_is_deepseek():
    cfg = model_config.for_stage("distillation")
    assert cfg.provider == "deepseek"


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
