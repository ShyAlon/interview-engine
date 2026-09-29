from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from fastapi.testclient import TestClient
from interview_engine.progress_journal import ProgressJournal
from interview_engine.http_service import SessionRegistry, create_app


def test_journal_independent_cursor_and_concurrent_read(tmp_path):
    sid=str(uuid4());journal=ProgressJournal(tmp_path,sid,'task')
    first=journal.append(phase_key='intent',stage='probing',label='Probing',status='started')
    with ThreadPoolExecutor() as pool:
        pending=pool.submit(journal.append,phase_key='intent',stage='probing',label='Probing',status='heartbeat')
        assert journal.read(after=0).events[0]==first
        second=pending.result()
    assert second.cursor==first.cursor+1
    assert [row.cursor for row in ProgressJournal(tmp_path,sid,'',create=False).read(first.cursor).events]==[second.cursor]
    registry=SessionRegistry(state_dir=tmp_path)
    client=TestClient(create_app(registry,service_token='test'))
    response=client.get(f'/v1/sessions/{sid}/progress?after={first.cursor}',headers={'X-Interview-Engine-Token':'test'})
    assert response.status_code==200
    assert response.json()['events'][0]['status']=='heartbeat'
    assert client.get(f'/v1/sessions/{sid}/progress').status_code==401


def test_http_start_progress_is_readable_before_synchronous_response(tmp_path):
    import threading,time
    from vmodel_contracts.interactive import InputRequest,InputKind
    from interview_engine.http_service import SessionRegistry,create_app
    from interview_engine.progress_journal import ProgressJournal
    gate=threading.Event();entered=threading.Event();sid=str(uuid4())
    class SlowProgram:
        model_map=None
        def advance(self):
            ProgressJournal(tmp_path,sid,'task').append(phase_key='intent',stage='probe',label='Probing',status='started')
            entered.set();assert gate.wait(3)
            return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Who?')
    registry=SessionRegistry(lambda _id,_request:SlowProgram(),state_dir=tmp_path)
    app=create_app(registry,service_token='test')
    def start():
        return TestClient(app,headers={'X-Interview-Engine-Token':'test'}).post('/v1/sessions',json={'task_id':'task','session_id':sid,'idea':'idea','model_route':'m'})
    with ThreadPoolExecutor() as pool:
        reply=pool.submit(start)
        assert entered.wait(2)
        with TestClient(app,headers={'X-Interview-Engine-Token':'test'}) as client:
            page=client.get(f'/v1/sessions/{sid}/progress?after=0',timeout=1)
        assert page.status_code==200 and page.json()['events'][0]['status']=='started'
        gate.set()
        assert reply.result(timeout=3).status_code==201


def test_progress_never_invalidates_open_input_sequence(tmp_path):
    from vmodel_contracts.interactive import StartSessionRequest,InputRequest,InputKind,InputResponse,ProcessStatus
    class Program:
        model_map=None
        finished=False
        def advance(self,response=None):
            if response is None:return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Who?')
            self.finished=True;return None
    registry=SessionRegistry(lambda _sid,_req:Program(),state_dir=tmp_path)
    session,_=registry.start(StartSessionRequest(task_id='task',idea='idea',model_route='m'))
    before=session.snapshot()
    journal=ProgressJournal(tmp_path,session.session_id,'task')
    for _ in range(3):journal.append(phase_key='intent',stage='probe',label='Probing',status='heartbeat')
    after=registry.get(session.session_id).snapshot()
    assert after.sequence==before.sequence and after.open_input==before.open_input
    accepted,done=registry.respond(session.session_id,InputResponse(input_id=before.open_input.input_id,
        expected_sequence=before.sequence,value='founders',actor_id='tester'))
    assert done.process_status==ProcessStatus.COMPLETED
