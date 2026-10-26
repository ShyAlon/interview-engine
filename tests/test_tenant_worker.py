from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from interview_engine.tenant_worker import TenantWorkerMiddleware,TenantRegistryRouter,worker_root

A='00000000-0000-0000-0000-000000000001'
B='00000000-0000-0000-0000-000000000002'
SA='00000000-0000-0000-0000-0000000000a1'
SB='00000000-0000-0000-0000-0000000000a2'

class Registry:
    def __init__(self,factory,state_dir):self.root=state_dir;self.records={}
    def start(self,session,value):self.records[session]=value;return str(self.root/session)
    def get(self,session):return self.records.get(session)


def test_router_tenant_cache_and_same_session_isolation(tmp_path):
    router=TenantRegistryRouter(Registry,None,tmp_path,allow_file_sessions=True)
    app=FastAPI()
    @app.post('/sessions/{session}')
    def start(session:str,value:str):return {'root':router.start(session,value),'worker_root':str(worker_root(tmp_path))}
    @app.get('/sessions/{session}')
    def get(session:str):return {'value':router.get(session)}
    app.add_middleware(TenantWorkerMiddleware,base=tmp_path)
    with TestClient(app) as client:
        assert client.get('/sessions/same').status_code==401
        for owner,value in [(A,'private-a'),(B,'private-b')]:
            h={'X-Requi-Owner':owner,'X-Interview-Engine-Token':'local-interview-engine-token'}
            response=client.post('/sessions/same',params={'value':value},headers=h)
            assert response.json()['root']==str(tmp_path/'tenants'/owner/'sessions'/'same')
        assert client.get('/sessions/same',headers={'X-Requi-Owner':A,'X-Interview-Engine-Token':'local-interview-engine-token'}).json()['value']=='private-a'
        assert client.get('/sessions/same',headers={'X-Requi-Owner':B,'X-Interview-Engine-Token':'local-interview-engine-token'}).json()['value']=='private-b'
        assert client.get('/sessions/same',headers={'X-Requi-Owner':'../x'}).status_code==401
        assert client.get('/sessions/same').status_code==401


def test_namespace_symlink_denied(tmp_path):
    (tmp_path/'outside').mkdir();(tmp_path/'tenants').symlink_to(tmp_path/'outside')
    app=FastAPI();app.add_middleware(TenantWorkerMiddleware,base=tmp_path)
    with TestClient(app) as client:
        assert client.get('/sessions/same',headers={'X-Requi-Owner':A,'X-Interview-Engine-Token':'local-interview-engine-token'}).status_code==403


def test_restart_retains_same_owner_root(tmp_path):
    assert tmp_path.resolve()==tmp_path
    # Root selection is deterministic even with a fresh per-owner registry cache.
    for _ in range(2):
        router=TenantRegistryRouter(Registry,None,tmp_path,allow_file_sessions=True)
        app=FastAPI()
        @app.get('/')
        def root():return {'path':str(router._registry().root)}
        app.add_middleware(TenantWorkerMiddleware,base=tmp_path)
        with TestClient(app) as client:
            assert client.get('/',headers={'X-Requi-Owner':A,'X-Interview-Engine-Token':'local-interview-engine-token'}).json()['path']==str(tmp_path/'tenants'/A/'sessions')



def test_owner_header_without_service_token_cannot_read_or_create_paths(tmp_path):
    app=FastAPI();app.add_middleware(TenantWorkerMiddleware,base=tmp_path,service_token='real-token')
    with TestClient(app) as client:
        assert client.get('/',headers={'X-Requi-Owner':A}).status_code==401
        assert client.get('/',headers={'X-Requi-Owner':A,'X-Interview-Engine-Token':'wrong-token'}).status_code==401
    assert not (tmp_path/'tenants').exists()


def test_stop_same_session_id_is_owner_scoped(tmp_path):
    from uuid import uuid4
    from interview_engine.http_service import SessionRegistry,create_app
    from vmodel_contracts.interactive import ProcessStatus
    router=TenantRegistryRouter(SessionRegistry,None,tmp_path,allow_file_sessions=True)
    app=create_app(router,service_token='fake-service')
    app.add_middleware(TenantWorkerMiddleware,base=tmp_path,service_token='fake-service')
    sid=str(uuid4());ha={'X-Requi-Owner':A,'X-Interview-Engine-Token':'fake-service'};hb={'X-Requi-Owner':B,'X-Interview-Engine-Token':'fake-service'}
    with TestClient(app) as c:
        for headers,task in [(ha,'a'),(hb,'b')]:
            assert c.post('/v1/sessions',headers=headers,json={'task_id':task,'session_id':sid,'idea':'fake','model_route':'local'}).status_code==201
        assert c.post(f'/v1/sessions/{sid}/stop',headers=ha).json()['process_status']==ProcessStatus.STOPPED
        assert c.get(f'/v1/sessions/{sid}',headers=hb).json()['process_status']==ProcessStatus.RUNNING
        assert c.post(f'/v1/sessions/{uuid4()}/stop',headers=hb).status_code==404
        assert c.post(f'/v1/sessions/{sid}/stop',headers={'X-Requi-Owner':A}).status_code==401
    assert (tmp_path/'tenants'/A/'sessions'/sid/'interview'/'stop-request.json').exists()
    assert not (tmp_path/'tenants'/B/'sessions'/sid/'interview'/'stop-request.json').exists()


def test_middleware_session_scope_required_and_validated(tmp_path):
    app=FastAPI()
    @app.get('/')
    def root():return {'ok':True}
    app.add_middleware(TenantWorkerMiddleware,base=tmp_path,service_token='real-token',require_session=True)
    h={'X-Requi-Owner':A,'X-Interview-Engine-Token':'real-token'}
    with TestClient(app) as client:
        assert client.get('/',headers=h).status_code==401
        assert client.get('/',headers={**h,'X-Requi-Session-Id':'not-a-uuid'}).status_code==401
        assert client.get('/',headers={**h,'X-Requi-Session-Id':SA}).status_code==200


def test_middleware_session_scope_optional_when_file_persistence(tmp_path):
    app=FastAPI()
    @app.get('/')
    def root():return {'ok':True}
    app.add_middleware(TenantWorkerMiddleware,base=tmp_path,service_token='real-token')
    h={'X-Requi-Owner':A,'X-Interview-Engine-Token':'real-token'}
    with TestClient(app) as client:
        assert client.get('/',headers=h).status_code==200
        assert client.get('/',headers={**h,'X-Requi-Session-Id':SA}).status_code==200


class _StubRegistry:
    def __init__(self,factory,state_dir):self.root=state_dir;self._sessions={};self.closed=False
    def close(self):self.closed=True


def _active_session():
    import threading
    class S:pass
    session=S();session.pg_claim=object();session.pg_stopped=False;session.write_lock=threading.RLock()
    return session,threading.Event()


def test_same_scope_returns_cached_registry(tmp_path):
    router=TenantRegistryRouter(_StubRegistry,None,tmp_path,allow_file_sessions=True)
    from interview_engine import tenant_worker
    token=tenant_worker._owner.set(A)
    try:
        assert router._registry() is router._registry()
    finally:
        tenant_worker._owner.reset(token)


def test_idle_registry_evicted_active_work_never(tmp_path,monkeypatch):
    monkeypatch.setenv('INTERVIEW_ENGINE_REGISTRY_IDLE_TTL','0')
    router=TenantRegistryRouter(_StubRegistry,None,tmp_path,allow_file_sessions=True)
    from interview_engine import tenant_worker
    session,halt=_active_session()
    token=tenant_worker._owner.set(A)
    try:
        first=router._registry()
        # With an active session the registry is not evictable.
        first._sessions['live']=session
        assert not router._evictable(first)
        first._sessions.pop('live')
        assert router._evictable(first)
        # A different scope's lookup retires the idle registry through close().
        token_b=tenant_worker._owner.set(B)
        try:router._registry()
        finally:tenant_worker._owner.reset(token_b)
        assert first.closed
    finally:
        halt.set();tenant_worker._owner.reset(token)


def test_capacity_rejects_rather_than_closing_active_work(tmp_path,monkeypatch):
    monkeypatch.setenv('INTERVIEW_ENGINE_REGISTRY_CAP','1')
    monkeypatch.setenv('INTERVIEW_ENGINE_REGISTRY_IDLE_TTL','0')
    router=TenantRegistryRouter(_StubRegistry,None,tmp_path,allow_file_sessions=True)
    from interview_engine import tenant_worker
    session,halt=_active_session()
    token=tenant_worker._owner.set(A)
    try:
        first=router._registry()
        first._sessions['live']=session
        token_b=tenant_worker._owner.set(B)
        try:
            import pytest
            with pytest.raises(RuntimeError,match='tenant registry capacity reached'):
                router._registry()
        finally:tenant_worker._owner.reset(token_b)
        assert not first.closed  # active work was never closed to make room
    finally:
        halt.set();tenant_worker._owner.reset(token)


def test_injected_pool_is_used_and_never_closed(tmp_path):
    # Hosted in-process the host owns the process pool: the router must build
    # stores on the INJECTED pool and must not close it at router close.
    import contextlib, pytest
    class Marker(Exception):pass
    class FakePool:
        closed=False
        @contextlib.contextmanager
        def connection(self):
            raise Marker('injected pool used')
            yield
        def close(self):FakePool.closed=True
    router=TenantRegistryRouter(_StubRegistry,None,tmp_path,session_dsn='postgresql://unused',pool=FakePool())
    from interview_engine import tenant_worker
    token=tenant_worker._owner.set(A);stoken=tenant_worker._session.set('auth-session-1')
    try:
        with pytest.raises(RuntimeError) as caught:
            router._registry()
        assert isinstance(caught.value.__cause__, Marker)  # the INJECTED pool drove the startup probe
    finally:
        tenant_worker._session.reset(stoken);tenant_worker._owner.reset(token)
    router.close()
    assert FakePool.closed is False
