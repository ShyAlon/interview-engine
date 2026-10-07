"""No network: client construction must apply one non-thinking policy."""
import pytest
from interview_engine import model_config, session_models


@pytest.mark.parametrize('stage', session_models.STAGES)
def test_pinned_deepseek_plain_calls_disable_thinking(monkeypatch, stage):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    with session_models.use(session_models.resolve('deepseek')):
        model = model_config.build_chat_model(stage)
    assert model.extra_body == {'thinking': {'type': 'disabled'}}
    assert 'response_format' not in model.model_kwargs


def test_openai_policy_not_changed(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('INTERVIEW_ENGINE_MODEL_PROBING', 'openai:gpt-4o-mini')
    model = model_config.build_chat_model('probing')
    assert model.extra_body is None
