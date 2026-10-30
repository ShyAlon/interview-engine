"""Retired file mode must fail before side effects or model routing."""
import pytest


def test_application_requires_pg_before_local_paths_or_provider_setup(tmp_path,monkeypatch):
    from interview_engine.application import InterviewApplication
    from interview_engine import session_models
    calls=[]
    monkeypatch.setattr(session_models,'resolve',lambda *a,**k:calls.append(1))
    with pytest.raises(ValueError,match='PG'):
        InterviewApplication('retired','task','idea',artifact_root=str(tmp_path),model_route='ollama')
    assert calls==[] and not list(tmp_path.iterdir())


@pytest.mark.parametrize('tenant',[False,True])
def test_service_file_opt_in_never_bypasses_required_pg(monkeypatch,tenant):
    from interview_engine.http_service import _build_service_app
    monkeypatch.delenv('INTERVIEW_ENGINE_SESSIONS_DSN',raising=False)
    monkeypatch.setenv('INTERVIEW_ENGINE_SESSIONS_FILE','1')
    monkeypatch.setenv('INTERVIEW_ENGINE_TENANTS','1' if tenant else '0')
    monkeypatch.setenv('INTERVIEW_ENGINE_TOKEN','private-test-token-123456789')
    with pytest.raises(RuntimeError,match='PostgreSQL session persistence is required'):
        _build_service_app()
