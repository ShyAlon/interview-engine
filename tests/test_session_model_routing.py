"""An HTTP session's user choice outranks shared configuration and survives restart."""
import json
import pytest
from fastapi.testclient import TestClient
from interview_engine import model_config, session_models
from interview_engine.http_service import SessionRegistry,create_app
from vmodel_contracts.interactive import StartSessionRequest

TOKEN='test-token'

def test_selected_route_reads_matching_shared_env_and_is_isolated(monkeypatch):
    monkeypatch.setattr(model_config,'_env',lambda key:'ollama:phi4' if key.startswith('INTERVIEW_ENGINE_MODEL_') else None)
    with session_models.use(session_models.resolve('ollama')):
        assert model_config.for_stage('probing').model=='qwen2.5-coder:14b'
        assert model_config.for_stage('conops').model=='qwen2.5-coder:14b'
    assert set(session_models.resolve('deepseek').values())=={('deepseek','deepseek-flash')}
    assert model_config.for_stage('probing').model=='phi4'  # legacy CLI still sees env

@pytest.mark.parametrize('route',('unknown','local',''))
def test_unsupported_route_rejected_without_fallback(route):
    with pytest.raises(Exception,match='unsupported interview model route'):
        session_models.resolve(route)

def test_deepseek_route_has_price_before_first_call(monkeypatch):
    from vmodel_contracts import price_for
    assert price_for(session_models.CATALOG['deepseek'][1], 'deepseek') == (0.30,1.20)
    monkeypatch.setattr(model_config,'_env',lambda key:'test-key' if key=='DEEPSEEK_API_KEY' else None)
    models=session_models.resolve('deepseek')
    models['probing']=('deepseek','not-in-price-table')
    for stage in session_models.STAGES[1:]:models[stage]=models['probing']
    with pytest.raises(Exception,match='No price for selected model'):
        session_models.validate(models)

def test_deepseek_requires_key(monkeypatch):
    monkeypatch.setattr(model_config,'_env',lambda key:None)
    with pytest.raises(Exception,match='DEEPSEEK_API_KEY'):
        session_models.validate(session_models.resolve('deepseek'))

def test_http_preflight_422_before_session_created(monkeypatch,tmp_path,pg_registry):
    import interview_engine.http_service as service
    monkeypatch.setattr(model_config,'_env',lambda key:None)
    monkeypatch.setattr(service,'default_program_factory',service.default_program_factory)
    registry=pg_registry(service.default_program_factory,allow_db_provider_dispatch=True)
    client=TestClient(create_app(registry,service_token=TOKEN),headers={'X-Interview-Engine-Token':TOKEN})
    response=client.post('/v1/sessions',json=StartSessionRequest(task_id='t',idea='idea',model_route='deepseek').model_dump(mode='json'))
    assert response.status_code==422
    assert not list(tmp_path.rglob('http-session.json'))
    assert not registry._sessions

def test_pinned_map_written_and_restored_after_environment_change(monkeypatch,tmp_path,pg_registry):
    from interview_engine import http_service as service
    class Program:
        finished=False
        open_input=None
        def __init__(self,route,models):self.model_route,self.model_map=route,models
        def advance(self,response=None):return None
    def factory(sid,request):
        row=pg_registry.store.load_published(sid,pg_registry.owner)
        models=row['control']['model_map'] if row and row['control'] else session_models.resolve(request.model_route)
        session_models.validate(models)
        return Program(request.model_route,models)
    monkeypatch.setattr(model_config,'_env',lambda key:'test-key' if key=='DEEPSEEK_API_KEY' else None)
    registry=pg_registry(factory,lease_ttl=.1,allow_db_provider_dispatch=True)
    session,_=registry.start(StartSessionRequest(task_id='t',idea='idea',model_route='deepseek'))
    saved=pg_registry.store.load_published(session.session_id,pg_registry.owner)['control']
    assert saved['model_map']['probing']==['deepseek','deepseek-flash']
    monkeypatch.setattr(model_config,'_env',lambda key:'ollama:phi4' if key.startswith('INTERVIEW_ENGINE_MODEL_') else 'test-key' if key=='DEEPSEEK_API_KEY' else None)
    registry.close()
    import time;time.sleep(.15)
    restored=pg_registry(factory,allow_db_provider_dispatch=True).get(session.session_id)
    assert restored.program.model_map['probing']==['deepseek','deepseek-flash']

def test_ollama_preflight_missing_model(monkeypatch):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {"data":[{"id":"phi4"}]}
    monkeypatch.setattr('httpx.get',lambda *args,**kwargs:Response())
    with pytest.raises(Exception,match='unavailable'):
        session_models.validate(session_models.resolve('ollama'))

def test_ollama_preflight_accepts_implicit_latest_tag(monkeypatch):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {"data":[{"id":"phi4:latest"}]}
    monkeypatch.setattr('httpx.get',lambda *args,**kwargs:Response())
    models=session_models.resolve('ollama',initial_model='phi4')
    session_models.validate(models,'ollama')
    assert models['probing']==('ollama','phi4')


def test_ollama_preflight_does_not_match_other_tag(monkeypatch):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {"data":[{"id":"phi4:other"}]}
    monkeypatch.setattr('httpx.get',lambda *args,**kwargs:Response())
    with pytest.raises(Exception,match="unavailable") as caught:
        session_models.validate(session_models.resolve('ollama',initial_model='phi4'),'ollama')
    assert "choose an installed model" in caught.value.render()


def test_ollama_preflight_inaccessible(monkeypatch):
    import httpx
    def fail(*args,**kwargs):raise httpx.ConnectError('connection refused')
    monkeypatch.setattr(httpx,'get',fail)
    with pytest.raises(Exception,match='could not list models'):
        session_models.validate(session_models.resolve('ollama'))

def test_worker_thread_inherits_pinned_route(monkeypatch):
    monkeypatch.setattr(model_config,'_env',lambda key:'deepseek:deepseek-flash' if key.startswith('INTERVIEW_ENGINE_MODEL_') else None)
    with session_models.use(session_models.resolve('deepseek')):
        assert model_config._run_with_deadline(lambda:model_config.for_stage('probing').model,2,'probing')=='deepseek-flash'


def test_http_ollama_env_pins_phi4_and_usage_route(monkeypatch):
    monkeypatch.setattr(model_config,'_env',lambda key:'ollama:phi4' if key.startswith('INTERVIEW_ENGINE_MODEL_') else None)
    models=session_models.resolve('ollama')
    assert set(models.values())=={('ollama','qwen2.5-coder:14b')}
    with session_models.use(models):
        cfg=model_config.for_stage('probing')
        assert cfg.model=='qwen2.5-coder:14b'
        from interview_engine.telemetry import UsageLog
        from vmodel_contracts import StageTokenUsage
        recorded=[];UsageLog('test-session',sink=lambda item:recorded.append(item.model_dump(mode='json'))).record('probing',cfg.provider,cfg.model,{'input_tokens':12,'output_tokens':3})
        assert recorded[0]['provider']=='ollama' and recorded[0]['model']=='qwen2.5-coder:14b'


def test_http_route_preview_and_start_share_env_resolution(monkeypatch,pg_registry):
    monkeypatch.setattr(model_config,'_env',lambda key:'ollama:phi4' if key.startswith('INTERVIEW_ENGINE_MODEL_') else None)
    monkeypatch.setattr(session_models,'validate',lambda models,route=None:None)
    client=TestClient(create_app(pg_registry(),service_token=TOKEN),headers={'X-Interview-Engine-Token':TOKEN})
    response=client.get('/v1/model-routes/ollama')
    assert response.status_code==200
    assert set(tuple(pair) for pair in response.json()['model_map'].values())=={('ollama','qwen2.5-coder:14b')}


def test_explicit_route_not_blocked_by_legacy_env(monkeypatch):
    monkeypatch.setattr(model_config,'_env',lambda key:'ollama:phi4' if key.startswith('INTERVIEW_ENGINE_MODEL_') else None)
    assert set(session_models.resolve('deepseek').values())=={('deepseek','deepseek-flash')}
    assert session_models.initial_default()==('ollama','phi4')


def test_divergent_legacy_defaults_diagnosed_only_at_default_lookup(monkeypatch):
    settings={'INTERVIEW_ENGINE_MODEL_PROBING':'ollama:phi4','INTERVIEW_ENGINE_MODEL_EXPERT_OPINION':'ollama:qwen2.5-coder:14b'}
    monkeypatch.setattr(model_config,'_env',lambda key:settings.get(key))
    with pytest.raises(Exception) as caught:session_models.initial_default()
    assert all(key in str(caught.value) for key in settings)
    assert set(session_models.resolve('deepseek').values())=={('deepseek','deepseek-flash')}


def test_env_has_priority_over_delivery_ollama_model(monkeypatch):
    monkeypatch.setattr(model_config,'_env',lambda key:'ollama:phi4' if key.startswith('INTERVIEW_ENGINE_MODEL_') else 'qwen2.5-coder:14b' if key=='DELIVERY_OLLAMA_MODEL' else None)
    assert set(session_models.resolve('ollama').values())=={('ollama','qwen2.5-coder:14b')}


def test_catalog_fallback_when_shared_stage_env_absent(monkeypatch):
    monkeypatch.setattr(model_config,'_env',lambda key:None)
    assert set(session_models.resolve('ollama').values())=={('ollama','qwen2.5-coder:14b')}


def test_partial_shared_env_inherits_phi4_for_unset_expert_opinion(monkeypatch):
    configured={'PROBING','DISTILLATION','PHRASING','SCOPE_TIEBREAK'}
    monkeypatch.setattr(model_config,'_env',lambda key:'ollama:phi4' if key.startswith('INTERVIEW_ENGINE_MODEL_') and key.split('INTERVIEW_ENGINE_MODEL_')[1] in configured else None)
    models=session_models.resolve('ollama')
    assert set(models.values())=={('ollama','qwen2.5-coder:14b')}
    assert models['expert_opinion']==('ollama','qwen2.5-coder:14b')
    assert models['conops']==('ollama','qwen2.5-coder:14b')
    with session_models.use(models):
        assert model_config.for_stage('expert_opinion').model=='qwen2.5-coder:14b'


def test_route_preview_with_divergent_legacy_defaults_allows_explicit_choice(monkeypatch,pg_registry):
    settings={'INTERVIEW_ENGINE_MODEL_PROBING':'ollama:phi4','INTERVIEW_ENGINE_MODEL_EXPERT_OPINION':'ollama:qwen2.5-coder:14b'}
    monkeypatch.setattr(model_config,'_env',lambda key:settings.get(key))
    monkeypatch.setattr(session_models,'validate',lambda models,route=None:None)
    client=TestClient(create_app(pg_registry(),service_token=TOKEN),headers={'X-Interview-Engine-Token':TOKEN})
    assert client.get('/v1/model-routes/ollama').status_code==200
    default=client.get('/v1/model-routes/default')
    assert default.status_code==422
    assert all(key in str(default.json()) for key in settings)
