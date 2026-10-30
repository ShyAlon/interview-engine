import pytest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from interview_engine.tenant_worker import TenantWorkerMiddleware,TenantRegistryRouter,worker_root

A='00000000-0000-0000-0000-000000000001'
B='00000000-0000-0000-0000-000000000002'
SA='00000000-0000-0000-0000-0000000000a1'
SB='00000000-0000-0000-0000-0000000000a2'

@pytest.fixture
def cache_pg_port(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from interview_engine import tenant_worker
    monkeypatch.setattr(tenant_worker,'pg_session_store',lambda pool,owner,session:SimpleNamespace(root=tmp_path/'tenants'/owner/'sessions'))
    token=tenant_worker._session.set(SA)
    yield
    tenant_worker._session.reset(token)


class Registry:
    def __init__(self,factory,*,session_store,owner_id,**kwargs):
        self.root=session_store.root;self.records={}
    def start(self,session,value):self.records[session]=value;return str(self.root/session)
    def get(self,session):return self.records.get(session)


def test_router_tenant_cache_and_same_session_isolation(cache_pg_port,tmp_path):
    router=TenantRegistryRouter(Registry,None,tmp_path,session_dsn='postgresql://test-only',pool=object())
    app=FastAPI()
    @app.post('/sessions/{session}')
    def start(session:str,value:str):return {'root':router.start(session,value),'worker_root':str(worker_root(tmp_path))}
    @app.get('/sessions/{session}')
    def get(session:str):return {'value':router.get(session)}
    app.add_middleware(TenantWorkerMiddleware,base=tmp_path)
    with TestClient(app) as client:
        assert client.get('/sessions/same').status_code==401
        for owner,value in [(A,'private-a'),(B,'private-b')]:
            h={'X-Requi-Owner':owner,'X-Interview-Engine-Token':'local-interview-engine-token','X-Requi-Session-Id':SA}
            response=client.post('/sessions/same',params={'value':value},headers=h)
            assert response.json()['root']==str(tmp_path/'tenants'/owner/'sessions'/'same')
        assert client.get('/sessions/same',headers={'X-Requi-Owner':A,'X-Interview-Engine-Token':'local-interview-engine-token','X-Requi-Session-Id':SA}).json()['value']=='private-a'
        assert client.get('/sessions/same',headers={'X-Requi-Owner':B,'X-Interview-Engine-Token':'local-interview-engine-token','X-Requi-Session-Id':SA}).json()['value']=='private-b'
        assert client.get('/sessions/same',headers={'X-Requi-Owner':'../x'}).status_code==401
        assert client.get('/sessions/same').status_code==401


def test_namespace_symlink_denied(tmp_path):
    (tmp_path/'outside').mkdir();(tmp_path/'tenants').symlink_to(tmp_path/'outside')
    app=FastAPI();app.add_middleware(TenantWorkerMiddleware,base=tmp_path)
    with TestClient(app) as client:
        assert client.get('/sessions/same',headers={'X-Requi-Owner':A,'X-Interview-Engine-Token':'local-interview-engine-token','X-Requi-Session-Id':SA}).status_code==403


def test_restart_retains_same_owner_root(cache_pg_port,tmp_path):
    assert tmp_path.resolve()==tmp_path
    # Root selection is deterministic even with a fresh per-owner registry cache.
    for _ in range(2):
        router=TenantRegistryRouter(Registry,None,tmp_path,session_dsn='postgresql://test-only',pool=object())
        app=FastAPI()
        @app.get('/')
        def root():return {'path':str(router._registry().root)}
        app.add_middleware(TenantWorkerMiddleware,base=tmp_path)
        with TestClient(app) as client:
            assert client.get('/',headers={'X-Requi-Owner':A,'X-Interview-Engine-Token':'local-interview-engine-token','X-Requi-Session-Id':SA}).json()['path']==str(tmp_path/'tenants'/A/'sessions')



def test_owner_header_without_service_token_cannot_read_or_create_paths(tmp_path):
    app=FastAPI();app.add_middleware(TenantWorkerMiddleware,base=tmp_path,service_token='real-token')
    with TestClient(app) as client:
        assert client.get('/',headers={'X-Requi-Owner':A}).status_code==401
        assert client.get('/',headers={'X-Requi-Owner':A,'X-Interview-Engine-Token':'wrong-token'}).status_code==401
    assert not (tmp_path/'tenants').exists()


def test_stop_same_session_id_is_owner_scoped(tmp_path):
    from uuid import uuid4
    from interview_engine.http_service import SessionRegistry,create_app
    from test_pg_session_registry import make_conn,make_pool
    from vmodel_contracts.interactive import ProcessStatus
    conn=make_conn(tmp_path)
    router=TenantRegistryRouter(SessionRegistry,None,tmp_path,
        session_dsn=conn.info.dsn,pool=make_pool(conn._pg_server))
    app=create_app(router,service_token='fake-service')
    app.add_middleware(TenantWorkerMiddleware,base=tmp_path,service_token='fake-service',require_session=True)
    sid=str(uuid4())
    ha={'X-Requi-Owner':A,'X-Interview-Engine-Token':'fake-service','X-Requi-Session-Id':SA}
    hb={'X-Requi-Owner':B,'X-Interview-Engine-Token':'fake-service','X-Requi-Session-Id':SB}
    with TestClient(app) as c:
        assert c.post('/v1/sessions',headers=ha,json={'task_id':'a','session_id':sid,'idea':'fake','model_route':'local'}).status_code==201
        # PG has one global session identity: a foreign owner cannot read or
        # stop it, and cannot create a second session under that same identity.
        assert c.get(f'/v1/sessions/{sid}',headers=hb).status_code==404
        assert c.post(f'/v1/sessions/{sid}/stop',headers=hb).status_code==404
        assert c.get(f'/v1/sessions/{sid}',headers=ha).json()['process_status']==ProcessStatus.RUNNING
        assert c.post(f'/v1/sessions/{sid}/stop',headers=ha).json()['process_status']==ProcessStatus.STOPPED
        assert c.post(f'/v1/sessions/{uuid4()}/stop',headers=hb).status_code==404
        assert c.post(f'/v1/sessions/{sid}/stop',headers={'X-Requi-Owner':A}).status_code==401
    assert not list(tmp_path.rglob('stop-request.json'))
    conn.close()


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
    def __init__(self,factory,*,session_store,owner_id,**kwargs):
        self.root=session_store.root;self._sessions={};self.closed=False
    def close(self):self.closed=True


def _active_session():
    import threading
    class S:pass
    session=S();session.pg_claim=object();session.pg_stopped=False;session.write_lock=threading.RLock()
    return session,threading.Event()


def test_same_scope_returns_cached_registry(cache_pg_port,tmp_path):
    router=TenantRegistryRouter(_StubRegistry,None,tmp_path,session_dsn='postgresql://test-only',pool=object())
    from interview_engine import tenant_worker
    token=tenant_worker._owner.set(A)
    try:
        assert router._registry() is router._registry()
    finally:
        tenant_worker._owner.reset(token)


def test_idle_registry_evicted_active_work_never(cache_pg_port,tmp_path,monkeypatch):
    monkeypatch.setenv('INTERVIEW_ENGINE_REGISTRY_IDLE_TTL','0')
    router=TenantRegistryRouter(_StubRegistry,None,tmp_path,session_dsn='postgresql://test-only',pool=object())
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


def test_capacity_rejects_rather_than_closing_active_work(cache_pg_port,tmp_path,monkeypatch):
    monkeypatch.setenv('INTERVIEW_ENGINE_REGISTRY_CAP','1')
    monkeypatch.setenv('INTERVIEW_ENGINE_REGISTRY_IDLE_TTL','0')
    router=TenantRegistryRouter(_StubRegistry,None,tmp_path,session_dsn='postgresql://test-only',pool=object())
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


def test_installed_control_route_bypasses_interview_registry_and_rechecks_each_call(tmp_path):
    from interview_engine.tenant_worker import _owner,_session
    import pytest
    seen=[];active=[True]
    class NeverRegistry:
        def __init__(self,*a,**k):raise AssertionError('installed execution must not restore interview graph')
    class Control:
        def snapshot(self,sid):return {'sid':sid,'readonly':True}
        def respond(self,sid,response):raise ValueError('new interview input unavailable')
    def resolve(name,sid):
        seen.append((name,sid,_owner.get(),_session.get()))
        if not active[0]:raise PermissionError('revoked')
        return getattr(Control(),name) if sid=='installed' else None
    router=TenantRegistryRouter(NeverRegistry,None,tmp_path,session_dsn='unused',installed_resolver=resolve)
    owner=_owner.set(A);session=_session.set(SA)
    try:
        assert router.snapshot('installed')=={'sid':'installed','readonly':True}
        with pytest.raises(ValueError,match='new interview'):router.respond('installed',{})
        active[0]=False
        with pytest.raises(PermissionError,match='revoked'):router.snapshot('installed')
        assert len(seen)==3 and all(row[2:]==(A,SA) for row in seen)
        assert router._registries=={}
    finally:_owner.reset(owner);_session.reset(session)
