import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest
from vmodel_contracts.interactive import StartSessionRequest,InputRequest,InputKind,InputResponse,ProcessStatus
from interview_engine.http_service import SessionRegistry,ProtocolFailure


def test_registry_stop_is_durable_nonblocking_and_discards_late_pending(pg_registry,tmp_path):
    entered=threading.Event();release=threading.Event()
    class Program:
        finished=False
        def advance(self,response=None):
            if response is not None:entered.set();release.wait(2)
            return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='fake')
    registry=pg_registry(lambda *a:Program())
    session,_=registry.start(StartSessionRequest(task_id='fake',idea='fake',model_route='local'))
    answer=InputResponse(input_id=session.open_input.input_id,value='fake',actor_id='fake')
    with ThreadPoolExecutor() as pool:
        future=pool.submit(registry.respond,session.session_id,answer)
        assert entered.wait(1)
        state=registry.stop(session.session_id)
        assert state.process_status==ProcessStatus.STOP_REQUESTED and state.open_input is None
        release.set();_,state=future.result()
    state=registry.snapshot(session.session_id)
    assert state.process_status==ProcessStatus.STOPPED and state.open_input is None
    assert registry.snapshot(session.session_id).process_status==ProcessStatus.STOPPED
    with pytest.raises(ProtocolFailure):registry.respond(session.session_id,answer)
    restored=pg_registry(lambda *a:pytest.fail('no auto restore'))
    # Restart itself does not prove the previous worker is gone.
    assert restored.snapshot(session.session_id).process_status==ProcessStatus.STOPPED



def test_stop_during_start_keeps_registry_and_never_publishes_late_input(pg_registry,tmp_path):
    entered=threading.Event();release=threading.Event();sid=str(uuid4())
    class Program:
        finished=False
        def advance(self,response=None):
            entered.set();release.wait(2)
            return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='late')
    r=pg_registry(lambda *a:Program())
    with ThreadPoolExecutor() as pool:
        future=pool.submit(r.start,StartSessionRequest(session_id=sid,task_id='fake',idea='fake',model_route='local'))
        assert entered.wait(1);assert r.stop(sid).process_status==ProcessStatus.STOP_REQUESTED
        release.set();session,_=future.result()
    assert session.snapshot().process_status==ProcessStatus.STOPPED
    assert r.snapshot(sid).open_input is None



def test_saved_http_state_unchanged_by_late_response_after_stop(pg_registry,tmp_path):
    entered=threading.Event();release=threading.Event()
    class Program:
        finished=False
        def advance(self,response=None):
            if response is not None:entered.set();release.wait(2)
            return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='fake')
    r=pg_registry(lambda *a:Program())
    s,_=r.start(StartSessionRequest(task_id='fake',idea='fake',model_route='local'))
    answer=InputResponse(input_id=s.open_input.input_id,value='fake',actor_id='fake')
    with ThreadPoolExecutor() as pool:
        future=pool.submit(r.respond,s.session_id,answer);assert entered.wait(1);r.stop(s.session_id)
        before=r._store.load_published(s.session_id,pg_registry.owner)['control']
        release.set();future.result()
    after=r._store.load_published(s.session_id,pg_registry.owner)['control']
    assert after==before|{'stop_quiescent':True}



def test_stop_in_post_advance_gap_does_not_commit_late_control_state(pg_registry,tmp_path):
    class Program:
        finished=False
        def advance(self,response=None):
            return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='fake')
    r=pg_registry(lambda *a:Program())
    s,_=r.start(StartSessionRequest(task_id='fake',idea='fake',model_route='local'))
    prior=(s.sequence,dict(s.responses));advance=r._advance
    def gap(*args,**kwargs):
        pending=advance(*args,**kwargs)
        assert r.stop(s.session_id).process_status==ProcessStatus.STOP_REQUESTED
        return pending
    r._advance=gap
    answer=InputResponse(input_id=s.open_input.input_id,value='fake',actor_id='fake')
    _,state=r.respond(s.session_id,answer)
    assert state.process_status==ProcessStatus.STOP_REQUESTED
    assert r.snapshot(s.session_id).process_status==ProcessStatus.STOPPED
    assert s.responses==prior[1] and s.open_input is None
    # Only initial acceptance increments; late result cannot publish another input.
    assert s.sequence==prior[0]+2



def test_direct_publish_after_stop_cannot_change_control_state(pg_registry,tmp_path):
    from vmodel_contracts.interactive import EventType,NextStep
    r=pg_registry()
    s,_=r.start(StartSessionRequest(task_id='fake',idea='fake',model_route='local'))
    r.stop(s.session_id)
    before=(s.sequence,s.phase_id,s.status,s.open_input)
    from interview_engine.stop_fence import StopRequested
    with pytest.raises(StopRequested):
        r.publish(s.session_id,EventType.INPUT_REQUESTED,phase_id=10,status=ProcessStatus.WAITING_INPUT,next_step=NextStep.WAIT_INPUT,input_request=InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='late'))
    assert (s.sequence,s.phase_id,s.status,s.open_input)==before



def test_http_stop_authenticated_idempotent_and_rejects_new_inputs(pg_registry,tmp_path):
    from fastapi.testclient import TestClient
    from interview_engine.http_service import create_app
    class Program:
        finished=False
        def advance(self,response=None):
            return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='fake')
    r=pg_registry(lambda *a:Program())
    s,_=r.start(StartSessionRequest(task_id='fake',idea='fake',model_route='local'))
    pending=s.open_input
    app=create_app(r,service_token='fake-token')
    assert TestClient(app).post(f'/v1/sessions/{s.session_id}/stop').status_code==401
    c=TestClient(app,headers={'X-Interview-Engine-Token':'fake-token'})
    for _ in range(2):
        response=c.post(f'/v1/sessions/{s.session_id}/stop')
        assert response.status_code==200 and response.json()['process_status']==ProcessStatus.STOPPED
        assert response.json()['open_input'] is None
    assert c.post(f'/v1/sessions/{uuid4()}/stop').status_code==404
    assert c.post('/v1/sessions/not-an-id/stop').status_code==422
    answer=c.post(f'/v1/sessions/{s.session_id}/inputs/{pending.input_id}',json={'input_id':str(pending.input_id),'value':'late','actor_id':'fake'})
    assert answer.status_code==422 and not s.responses



def test_registry_requires_durable_storage_before_start():
    with pytest.raises(ValueError,match='PG session store required'):SessionRegistry()


def test_http_stop_does_not_wait_for_inflight_advance(pg_registry,tmp_path):
    from fastapi.testclient import TestClient
    from interview_engine.http_service import create_app
    entered=threading.Event();release=threading.Event()
    class Program:
        finished=False
        def advance(self,response=None):
            if response is not None:entered.set();release.wait(2)
            return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='fake')
    r=pg_registry(lambda *a:Program())
    s,_=r.start(StartSessionRequest(task_id='fake',idea='fake',model_route='local'));pending=s.open_input
    c=TestClient(create_app(r,service_token='fake'),headers={'X-Interview-Engine-Token':'fake'})
    with ThreadPoolExecutor() as pool:
        future=pool.submit(c.post,f'/v1/sessions/{s.session_id}/inputs/{pending.input_id}',json={'input_id':str(pending.input_id),'value':'fake','actor_id':'fake'})
        assert entered.wait(1)
        response=c.post(f'/v1/sessions/{s.session_id}/stop')
        assert response.status_code==200 and response.json()['process_status']==ProcessStatus.STOP_REQUESTED
        assert not future.done()
        release.set();assert future.result().json()['snapshot']['process_status']==ProcessStatus.STOP_REQUESTED
        assert r.snapshot(s.session_id).process_status==ProcessStatus.STOPPED



def test_stop_waits_for_factory_quiescence_and_never_advances_late_program(pg_registry,tmp_path):
    entered=threading.Event();release=threading.Event();sid=str(uuid4())
    class Program:
        def advance(self,*a):pytest.fail('late factory program must not advance')
    def factory(*args):
        entered.set();release.wait(2);return Program()
    r=pg_registry(factory)
    with ThreadPoolExecutor() as pool:
        future=pool.submit(r.start,StartSessionRequest(session_id=sid,task_id='fake',idea='fake',model_route='local'))
        assert entered.wait(1)
        assert r.stop(sid).process_status==ProcessStatus.STOP_REQUESTED
        assert not future.done()
        release.set();session,_=future.result()
    assert session.snapshot().process_status==ProcessStatus.STOPPED
    assert session.program is None and session.open_input is None



def test_explicit_same_host_stop_reconcile_requires_writer_exit_and_never_restores(pg_registry,tmp_path):
    entered=threading.Event();release=threading.Event();sid=str(uuid4())
    class Program:
        def advance(self,*args):entered.set();release.wait(2);return None
    r=pg_registry(lambda *args:Program())
    with ThreadPoolExecutor() as pool:
        future=pool.submit(r.start,StartSessionRequest(session_id=sid,task_id='fake',idea='fake',model_route='local'))
        assert entered.wait(1);r.stop(sid)
        restored=pg_registry(lambda *a:pytest.fail('no restart'))
        assert restored.reconcile_stopped(sid).process_status==ProcessStatus.STOP_REQUESTED
        release.set();future.result()
    assert restored.snapshot(sid).process_status==ProcessStatus.STOPPED
    assert restored.reconcile_stopped(sid).process_status==ProcessStatus.STOPPED
    assert restored.snapshot(sid).open_input is None



def test_stop_reconcile_http_authenticated_and_no_unstopped_mutation(pg_registry,tmp_path):
    from fastapi.testclient import TestClient
    from interview_engine.http_service import create_app
    r=pg_registry()
    s,_=r.start(StartSessionRequest(task_id='fake',idea='fake',model_route='local'))
    app=create_app(r,service_token='fake')
    endpoint=f'/v1/sessions/{s.session_id}/stop/reconcile'
    assert TestClient(app).post(endpoint).status_code==401
    c=TestClient(app,headers={'X-Interview-Engine-Token':'fake'})
    assert c.post(endpoint).status_code==409
    assert not s.pg_stopped
    r.stop(s.session_id)
    assert c.post(endpoint).json()['process_status']==ProcessStatus.STOPPED
