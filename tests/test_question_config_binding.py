"""App/worker config binding must not change a different app's default store."""
import pytest
from interview_engine import question_config as q


def test_context_binding_isolated_nested_and_reset_after_failure():
    default=q.store();one=q.ConfigStore();two=q.ConfigStore()
    one.put({'retrieval':{'max_words':120}});two.put({'retrieval':{'max_words':160}})
    with q.bound_store(one):
        assert q.current()['retrieval']['max_words']==120
        with pytest.raises(RuntimeError):
            with q.bound_store(two):
                assert q.current()['retrieval']['max_words']==160
                raise RuntimeError('test')
        assert q.store() is one
    assert q.store() is default


def test_app_injected_store_routes_and_program_share_binding_without_global_mutation():
    from fastapi.testclient import TestClient
    from interview_engine.http_service import create_app,SessionRegistry
    one=q.ConfigStore();two=q.ConfigStore();one.put({'retrieval':{'max_words':120}})
    default=q.store()
    a=create_app(SessionRegistry(),service_token='fixture',quality_store=one)
    b=create_app(SessionRegistry(),service_token='fixture',quality_store=two)
    @a.get('/probe')
    def probe():return {'words':q.current()['retrieval']['max_words']}
    with TestClient(a) as x,TestClient(b) as y:
        assert x.get('/probe').json()=={'words':120}
        assert x.get('/v1/question-quality-config',headers={'x-interview-engine-token':'fixture'}).json()['config']['retrieval']['max_words']==120
        assert y.get('/v1/question-quality-config',headers={'x-interview-engine-token':'fixture'}).json()['config']['retrieval']['max_words']==90
    assert q.store() is default


def test_readonly_or_unavailable_injected_store_has_safe_status():
    from fastapi.testclient import TestClient
    from interview_engine.http_service import create_app,SessionRegistry
    class Backend:
        def get(self):raise ConnectionError('postgresql://private-password@host/db')
        def put(self,cfg):raise PermissionError('private config source path')
        def status(self):return {'source':'postgres','fallback':False,'writable':False}
    app=create_app(SessionRegistry(),service_token='fixture',quality_store=Backend())
    with TestClient(app,raise_server_exceptions=False) as c:
        headers={'x-interview-engine-token':'fixture'}
        got=c.get('/v1/question-quality-config',headers=headers)
        assert got.status_code==503 and 'private-password' not in got.text
        got=c.put('/v1/question-quality-config',json={'config':{}},headers=headers)
        assert got.status_code==403 and 'source path' not in got.text
