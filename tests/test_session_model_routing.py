"""An HTTP session's user choice outranks shared configuration and survives restart."""
import json
import pytest
from fastapi.testclient import TestClient
from interview_engine import model_config, session_models
from interview_engine.http_service import SessionRegistry,create_app
from vmodel_contracts.interactive import StartSessionRequest

TOKEN='test-token'

def test_selected_route_overrides_shared_config_and_is_isolated(monkeypatch):
    monkeypatch.setattr(model_config,'_env',lambda key:'deepseek:deepseek-flash' if key.startswith('INTERVIEW_ENGINE_MODEL_') else 'test-key' if key=='DEEPSEEK_API_KEY' else None)
    with session_models.use(session_models.resolve('ollama')):
        assert model_config.for_stage('probing').provider=='ollama'
        assert model_config.for_stage('conops').model=='qwen2.5-coder:14b'
    with session_models.use(session_models.resolve('deepseek')):
        assert model_config.for_stage('probing').model=='deepseek-flash'
        assert model_config.for_stage('conops').model=='deepseek-flash'
    assert model_config.for_stage('probing').model=='deepseek-flash'  # legacy CLI still sees env

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

def test_http_preflight_422_before_session_created(monkeypatch,tmp_path):
    import interview_engine.http_service as service
    monkeypatch.setattr(model_config,'_env',lambda key:None)
    monkeypatch.setattr(service,'default_program_factory',service.default_program_factory)
    registry=SessionRegistry(service.default_program_factory,state_dir=tmp_path)
    client=TestClient(create_app(registry,service_token=TOKEN),headers={'X-Interview-Engine-Token':TOKEN})
    response=client.post('/v1/sessions',json=StartSessionRequest(task_id='t',idea='idea',model_route='deepseek').model_dump(mode='json'))
    assert response.status_code==422
    assert not list(tmp_path.iterdir())
    assert not registry._sessions

def test_pinned_map_written_and_restored_after_environment_change(monkeypatch,tmp_path):
    from interview_engine import http_service as service
    class Program:
        finished=False
        open_input=None
        def __init__(self,route,models):self.model_route,self.model_map=route,models
        def advance(self,response=None):return None
    def factory(sid,request):
        path=tmp_path/sid/'interview'/'http-session.json'
        models=json.loads(path.read_text())['model_map'] if path.exists() else session_models.resolve(request.model_route)
        session_models.validate(models)
        return Program(request.model_route,models)
    monkeypatch.setattr(model_config,'_env',lambda key:'test-key' if key=='DEEPSEEK_API_KEY' else None)
    registry=SessionRegistry(factory,state_dir=tmp_path)
    session,_=registry.start(StartSessionRequest(task_id='t',idea='idea',model_route='deepseek'))
    saved=json.loads((tmp_path/session.session_id/'interview'/'http-session.json').read_text())
    assert saved['model_map']['probing']==['deepseek','deepseek-flash']
    monkeypatch.setattr(model_config,'_env',lambda key:'ollama:phi4' if key.startswith('INTERVIEW_ENGINE_MODEL_') else 'test-key' if key=='DEEPSEEK_API_KEY' else None)
    restored=SessionRegistry(factory,state_dir=tmp_path).get(session.session_id)
    assert restored.program.model_map['probing']==['deepseek','deepseek-flash']

def test_ollama_preflight_missing_model(monkeypatch):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {"data":[{"id":"phi4"}]}
    monkeypatch.setattr('httpx.get',lambda *args,**kwargs:Response())
    with pytest.raises(Exception,match='unavailable'):
        session_models.validate(session_models.resolve('ollama'))

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
