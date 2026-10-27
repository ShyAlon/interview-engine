"""Child SDK construction never needs a provider secret or unbrokered model."""
import socket
import pytest
from requi_child_contract.broker import BrokerClient
from interview_engine import model_config
from interview_engine.wire_binding import broker_clients,validate_provider


def test_broker_preflight_uses_no_real_api_key(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    monkeypatch.setattr(model_config,'for_stage',lambda s:model_config.StageModel('deepseek','fake'))
    monkeypatch.setattr(model_config,'_env',lambda key:None)
    a,b=socket.socketpair();broker=BrokerClient(b.fileno(),next_call_id=lambda s:'call-1')
    try:
        with broker_clients(broker):
            model=model_config.build_chat_model('intent')
            assert model.openai_api_key.get_secret_value()=='parent-broker-no-secret'
            with pytest.raises(Exception):validate_provider('anthropic')
    finally:broker.close();a.close();b.close()


def test_broker_scope_never_reads_dotenv_credentials_or_provider_controls(monkeypatch):
    a,b=socket.socketpair();broker=BrokerClient(b.fileno(),next_call_id=lambda s:'call-1')
    monkeypatch.setattr(model_config,'_DOTENV',{'DEEPSEEK_API_KEY':'secret','INTERVIEW_ENGINE_JEV_SHADOW':'1'})
    monkeypatch.setenv('DEEPSEEK_API_KEY','ambient-secret')
    try:
        with broker_clients(broker):
            assert model_config._env('DEEPSEEK_API_KEY') is None
            assert model_config._env('INTERVIEW_ENGINE_JEV_SHADOW') is None
    finally:broker.close();a.close();b.close()
