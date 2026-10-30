from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from fastapi.testclient import TestClient
from interview_engine.pg_journal import ObservationalJournal
from interview_engine.http_service import SessionRegistry, create_app


def test_journal_independent_cursor_and_concurrent_read(pg_registry):
    from vmodel_contracts.interactive import StartSessionRequest
    registry=pg_registry(progress_mode='observational')
    session,_=registry.start(StartSessionRequest(task_id='task',idea='idea',model_route='m'))
    sid=session.session_id
    journal=ObservationalJournal(registry._store,session.pg_claim,sid,'task')
    first=journal.append(phase_key='intent',stage='probing',label='Probing',status='started')
    with ThreadPoolExecutor() as pool:
        pending=pool.submit(journal.append,phase_key='intent',stage='probing',label='Probing',status='heartbeat')
        assert registry.progress(sid,after=0).events[0]==first
        second=pending.result()
    assert second.cursor==first.cursor+1
    assert [row.cursor for row in registry.progress(sid,first.cursor).events]==[second.cursor]
    client=TestClient(create_app(registry,service_token='test'))
    response=client.get(f'/v1/sessions/{sid}/progress?after={first.cursor}',headers={'X-Interview-Engine-Token':'test'})
    assert response.status_code==200
    assert response.json()['events'][0]['status']=='heartbeat'
    assert client.get(f'/v1/sessions/{sid}/progress').status_code==401


def test_http_start_progress_is_readable_before_synchronous_response(pg_registry):
    import threading,time
    from vmodel_contracts.interactive import InputRequest,InputKind
    from interview_engine.http_service import SessionRegistry,create_app
    from interview_engine.pg_journal import ObservationalJournal
    gate=threading.Event();entered=threading.Event();sid=str(uuid4())
    class SlowProgram:
        model_map=None
        def advance(self):
            ObservationalJournal(registry._store,registry.get(sid).pg_claim,sid,'task').append(phase_key='intent',stage='probe',label='Probing',status='started')
            entered.set();assert gate.wait(3)
            return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Who?')
    registry=pg_registry(lambda _id,_request:SlowProgram(),progress_mode='observational')
    app=create_app(registry,service_token='test')
    def start():
        return TestClient(app,headers={'X-Interview-Engine-Token':'test'}).post('/v1/sessions',json={'task_id':'task','session_id':sid,'idea':'idea','model_route':'m'})
    with ThreadPoolExecutor() as pool:
        reply=pool.submit(start)
        assert entered.wait(2)
        client=TestClient(app,headers={'X-Interview-Engine-Token':'test'})
        try:page=pool.submit(client.get,f'/v1/sessions/{sid}/progress?after=0').result(timeout=1)
        finally:gate.set()
        assert page.status_code==200 and page.json()['events'][0]['status']=='started'
        gate.set()
        assert reply.result(timeout=3).status_code==201


def test_progress_never_invalidates_open_input_sequence(pg_registry):
    from vmodel_contracts.interactive import StartSessionRequest,InputRequest,InputKind,InputResponse,ProcessStatus
    class Program:
        model_map=None
        finished=False
        def advance(self,response=None):
            if response is None:return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Who?')
            self.finished=True;return None
    registry=pg_registry(lambda _sid,_req:Program(),progress_mode='observational')
    session,_=registry.start(StartSessionRequest(task_id='task',idea='idea',model_route='m'))
    before=session.snapshot()
    journal=ObservationalJournal(registry._store,session.pg_claim,session.session_id,'task')
    for _ in range(3):journal.append(phase_key='intent',stage='probe',label='Probing',status='heartbeat')
    after=registry.get(session.session_id).snapshot()
    assert after.sequence==before.sequence and after.open_input==before.open_input
    accepted,done=registry.respond(session.session_id,InputResponse(input_id=before.open_input.input_id,
        expected_sequence=before.sequence,value='founders',actor_id='tester'))
    assert done.process_status==ProcessStatus.COMPLETED
