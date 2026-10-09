from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from interview_engine.tenant_worker import TenantWorkerMiddleware,TenantRegistryRouter,worker_root

A='00000000-0000-0000-0000-000000000001'
B='00000000-0000-0000-0000-000000000002'

class Registry:
    def __init__(self,factory,state_dir):self.root=state_dir;self.records={}
    def start(self,session,value):self.records[session]=value;return str(self.root/session)
    def get(self,session):return self.records.get(session)


def test_router_tenant_cache_and_same_session_isolation(tmp_path):
    router=TenantRegistryRouter(Registry,None,tmp_path)
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
        router=TenantRegistryRouter(Registry,None,tmp_path)
        app=FastAPI()
        @app.get('/')
        def root():return {'path':str(router._registry().root)}
        app.add_middleware(TenantWorkerMiddleware,base=tmp_path)
        with TestClient(app) as client:
            assert client.get('/',headers={'X-Requi-Owner':A,'X-Interview-Engine-Token':'local-interview-engine-token'}).json()['path']==str(tmp_path/'tenants'/A/'sessions')

