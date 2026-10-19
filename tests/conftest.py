import pytest

import interview_engine.model_config as mc
@pytest.fixture(autouse=True)
def _reset_llm_call_budget(monkeypatch):
    """Each test is one run: the run-wide LLM call budget starts clean."""
    monkeypatch.setattr(mc, "_CALLS_USED", 0)


@pytest.fixture(autouse=True)
def _no_dotenv_files(monkeypatch):
    """No test reads a developer's .env files: model routes come from the code defaults and the test's own settings.
    A test that needs a .env builds its own and resets mc._DOTENV itself."""
    monkeypatch.setattr(mc, "_DOTENV", {})
