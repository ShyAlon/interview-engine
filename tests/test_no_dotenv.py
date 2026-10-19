"""Tests never read a developer's .env files: a route set there must not change what a test sees."""
import interview_engine.model_config as mc


def test_the_env_files_are_not_consulted_during_tests(monkeypatch, tmp_path):
    monkeypatch.delenv("INTERVIEW_ENGINE_MODEL_PROBING", raising=False)
    assert mc._DOTENV == {}
    assert mc._env("INTERVIEW_ENGINE_MODEL_PROBING") is None
    assert mc.for_stage("probing").provider == "ollama"


def test_a_test_can_still_load_its_own_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("INTERVIEW_ENGINE_MODEL_PROBING", raising=False)
    (tmp_path / ".env").write_text("INTERVIEW_ENGINE_MODEL_PROBING=deepseek:deepseek-chat\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mc, "_DOTENV", None)
    assert mc._env("INTERVIEW_ENGINE_MODEL_PROBING") == "deepseek:deepseek-chat"
