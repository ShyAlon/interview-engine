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
