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


def test_router_requires_pg_without_file_escape_hatch(tmp_path):
    from interview_engine.tenant_worker import TenantRegistryRouter,_owner
    router=TenantRegistryRouter(object,None,tmp_path)
    token=_owner.set('00000000-0000-0000-0000-000000000001')
    try:
        with pytest.raises(RuntimeError,match='PostgreSQL session persistence is required'):
            router._registry()
    finally:_owner.reset(token)


def test_registry_refuses_missing_pg_before_any_state_path(tmp_path):
    from interview_engine.http_service import SessionRegistry
    with pytest.raises(ValueError,match='PG session store required'):
        SessionRegistry(state_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_retired_sqlite_progress_module_absent():
    import importlib.util
    assert importlib.util.find_spec('interview_engine.progress_journal') is None


def test_sqlite_operator_paths_and_dependency_are_retired():
    from pathlib import Path
    root=Path(__file__).parents[1]
    assert not (root/'scripts/migrate-session-bindings.py').exists()
    assert not (root/'scripts/enrich-all-categorization-shadow.py').exists()
    assert 'langgraph-checkpoint-sqlite' not in (root/'pyproject.toml').read_text()
