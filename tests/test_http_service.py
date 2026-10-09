from uuid import uuid4
from fastapi.testclient import TestClient
from vmodel_contracts.interactive import (
    EventType,InputKind,InputOption,InputRequest,NextStep,ProcessStatus,ProtocolErrorCode,
)
from interview_engine.http_service import SessionRegistry,create_app

TOKEN='test-service-token'
def authed(app):return TestClient(app,headers={'X-Interview-Engine-Token':TOKEN})


def test_start_session_returns_typed_http_contract():
    client=authed(create_app(service_token=TOKEN));r=client.post('/v1/sessions',json={
        'protocol_version':1,'task_id':'task-1','idea':'A virtual dungeon master','model_route':'deepseek'})
    assert r.status_code==201
    body=r.json();assert body['accepted_event']['event_type']==EventType.RUN_STARTED
    assert body['snapshot']['process_status']==ProcessStatus.RUNNING
    assert body['snapshot']['next_step']==NextStep.RUN


def test_approval_is_single_idempotent_input_boundary():
    registry=SessionRegistry();client=authed(create_app(registry,service_token=TOKEN));started=client.post('/v1/sessions',json={
        'protocol_version':1,'task_id':'task-1','idea':'idea','model_route':'local'}).json();sid=started['snapshot']['session_id']
    pending=InputRequest(input_id=uuid4(),kind=InputKind.APPROVAL,prompt='Approve intent?',options=[
        InputOption(value=1,label='Approve'),InputOption(value=0,label='Reject')])
    registry.publish(sid,EventType.INPUT_REQUESTED,phase_id=10,status=ProcessStatus.WAITING_INPUT,
        next_step=NextStep.WAIT_INPUT,input_request=pending)
    body={'protocol_version':1,'input_id':str(pending.input_id),'value':1,'actor_id':'founder'}
    first=client.post(f'/v1/sessions/{sid}/inputs/{pending.input_id}',json=body)
    assert first.status_code==200 and first.json()['snapshot']['next_step']==NextStep.RUN
    replay=client.post(f'/v1/sessions/{sid}/inputs/{pending.input_id}',json=body)
    assert replay.status_code==200


def test_stale_input_uses_error_enum_and_http_conflict():
    registry=SessionRegistry();client=authed(create_app(registry,service_token=TOKEN));sid=client.post('/v1/sessions',json={
        'protocol_version':1,'task_id':'t','idea':'i','model_route':'m'}).json()['snapshot']['session_id']
    current=InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Who uses it?')
    registry.publish(sid,EventType.INPUT_REQUESTED,phase_id=10,status=ProcessStatus.WAITING_INPUT,
        next_step=NextStep.WAIT_INPUT,input_request=current)
    stale=uuid4();r=client.post(f'/v1/sessions/{sid}/inputs/{stale}',json={
        'protocol_version':1,'input_id':str(stale),'value':'players','actor_id':'founder'})
    assert r.status_code==409 and r.json()['detail']['code']==ProtocolErrorCode.STALE_INPUT

class FakeProgram:
    def __init__(self):self.finished=False;self.calls=0;self.pending=InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Who uses it?')
    def advance(self,response=None):
        self.calls+=1
        if response is None:return self.pending
        self.finished=True;return None

def test_http_session_drives_program_to_typed_input_and_completion():
    program=FakeProgram();registry=SessionRegistry(lambda session_id,request:program);client=authed(create_app(registry,service_token=TOKEN))
    started=client.post('/v1/sessions',json={'protocol_version':1,'task_id':'t','idea':'i','model_route':'m'}).json()
    assert started['snapshot']['process_status']==ProcessStatus.WAITING_INPUT
    pending=started['snapshot']['open_input'];reply={'protocol_version':1,'input_id':pending['input_id'],'value':'players','actor_id':'founder'}
    done=client.post(f"/v1/sessions/{started['snapshot']['session_id']}/inputs/{pending['input_id']}",json=reply).json()
    assert done['snapshot']['process_status']==ProcessStatus.COMPLETED


def test_service_rejects_missing_and_incorrect_tokens():
 app=create_app(service_token=TOKEN);payload={'protocol_version':1,'task_id':'t','idea':'i','model_route':'m'}
 assert TestClient(app).post('/v1/sessions',json=payload).status_code==401
 assert TestClient(app,headers={'X-Interview-Engine-Token':'wrong'}).post('/v1/sessions',json=payload).status_code==401
 assert authed(app).post('/v1/sessions',json=payload).status_code==201


class FailOnceProgram(FakeProgram):
    def advance(self,response=None):
        if response is None:return self.pending
        self.calls+=1
        if self.calls==1:raise RuntimeError('model unavailable')
        self.finished=True;return None

def test_failed_program_advance_keeps_typed_input_retryable():
    program=FailOnceProgram();registry=SessionRegistry(lambda session_id,request:program)
    session,_=registry.start(__import__('vmodel_contracts.interactive',fromlist=['StartSessionRequest']).StartSessionRequest(task_id='t',idea='i',model_route='m'))
    pending=session.open_input;response=__import__('vmodel_contracts.interactive',fromlist=['InputResponse']).InputResponse(input_id=pending.input_id,value='players',actor_id='founder')
    import pytest
    with pytest.raises(RuntimeError,match='model unavailable'):registry.respond(session.session_id,response)
    restored=registry.get(session.session_id)
    assert restored.status==ProcessStatus.WAITING_INPUT and restored.next_step==NextStep.WAIT_INPUT
    assert restored.open_input==pending and pending.input_id not in restored.responses
    _,snapshot=registry.respond(session.session_id,response)
    assert snapshot.process_status==ProcessStatus.COMPLETED


def test_failed_advance_returns_typed_retryable_service_error():
    program=FailOnceProgram();client=authed(create_app(SessionRegistry(lambda session_id,request:program),service_token=TOKEN))
    started=client.post('/v1/sessions',json={'protocol_version':1,'task_id':'t','idea':'i','model_route':'m'}).json();pending=started['snapshot']['open_input']
    response={'protocol_version':1,'input_id':pending['input_id'],'value':'players','actor_id':'founder'}
    failed=client.post(f"/v1/sessions/{started['snapshot']['session_id']}/inputs/{pending['input_id']}",json=response)
    assert failed.status_code==503 and failed.json()['detail']['code']==ProtocolErrorCode.INTERNAL_ERROR and failed.json()['detail']['retryable'] is True
    retried=client.post(f"/v1/sessions/{started['snapshot']['session_id']}/inputs/{pending['input_id']}",json=response)
    assert retried.status_code==200 and retried.json()['snapshot']['process_status']==ProcessStatus.COMPLETED


def test_registry_restores_open_input_after_engine_restart(tmp_path):
    first_program=FakeProgram();first=SessionRegistry(lambda session_id,request:first_program,state_dir=tmp_path)
    request=__import__('vmodel_contracts.interactive',fromlist=['StartSessionRequest']).StartSessionRequest(task_id='t',idea='i',model_route='m')
    session,_=first.start(request);pending=session.open_input
    second_program=FakeProgram();restarted=SessionRegistry(lambda session_id,request:second_program,state_dir=tmp_path)
    restored=restarted.get(session.session_id)
    assert restored.open_input==pending and restored.status==ProcessStatus.WAITING_INPUT
    response=__import__('vmodel_contracts.interactive',fromlist=['InputResponse']).InputResponse(input_id=pending.input_id,value='players',actor_id='founder')
    _,snapshot=restarted.respond(session.session_id,response)
    assert snapshot.process_status==ProcessStatus.COMPLETED


def test_conops_tradeoff_options_cross_typed_http_contract():
    from vmodel_contracts.interactive import InputOption
    option={'value':'1','label':'Progressive web app','summary':'Installable web experience',
        'advantages':['Fast rollout','One codebase'],'disadvantages':['Platform limits']}
    typed=InputOption.model_validate(option)
    assert typed.advantages==option['advantages'] and typed.disadvantages==option['disadvantages']


class UserActionProgram(FakeProgram):
    def advance(self,response=None):
        if response is None:return self.pending
        from interview_engine.errors import UserFacingError
        raise UserFacingError('stage conops needs a stronger model.','local model returned invalid output.','switch CONOPS to a stronger model and retry.')

def test_user_facing_error_keeps_actionable_structure():
    program=UserActionProgram();client=authed(create_app(SessionRegistry(lambda session_id,request:program),service_token=TOKEN))
    started=client.post('/v1/sessions',json={'protocol_version':1,'task_id':'t','idea':'i','model_route':'m'}).json();pending=started['snapshot']['open_input']
    response={'protocol_version':1,'input_id':pending['input_id'],'value':'players','actor_id':'founder'}
    failed=client.post(f"/v1/sessions/{started['snapshot']['session_id']}/inputs/{pending['input_id']}",json=response)
    detail=failed.json()['detail']
    assert failed.status_code==422 and detail['code']==ProtocolErrorCode.INVALID_RESPONSE and detail['retryable'] is False
    assert 'Error: stage conops needs a stronger model.' in detail['detail']
    assert 'Likely cause: local model returned invalid output.' in detail['detail']
    assert 'Next: switch CONOPS to a stronger model and retry.' in detail['detail']



def test_http_application_subscribes_console_progress():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'src/interview_engine/application.py').read_text()
    assert 'progress=ProgressBus();progress.subscribe(stderr_printer())' in source
    assert 'build_graph(log,checkpointer=saver,progress=progress' in source


class RationaleProgram(FakeProgram):
    def __init__(self):
        super().__init__();self.pending=self.pending.model_copy(update={'kind':InputKind.CHOICE,'options':[InputOption(value='1',label='One')],'rationale_required':True});self.received=None
    def advance(self,response=None):
        if response is None:return self.pending
        self.received=response;self.finished=True;return None

def test_http_input_preserves_separate_typed_rationale():
    program=RationaleProgram();client=authed(create_app(SessionRegistry(lambda session_id,request:program),service_token=TOKEN))
    started=client.post('/v1/sessions',json={'protocol_version':1,'task_id':'t','idea':'i','model_route':'m'}).json();pending=started['snapshot']['open_input']
    response={'protocol_version':1,'input_id':pending['input_id'],'value':'1','rationale':'because it fits','actor_id':'founder'}
    done=client.post(f"/v1/sessions/{started['snapshot']['session_id']}/inputs/{pending['input_id']}",json=response)
    assert done.status_code==200 and program.received.rationale=='because it fits'

class SwitchableProgram(FakeProgram):
    def __init__(self,route='ollama',calls=None):
        super().__init__();self.model_route=route;self.model_map={stage:(route,'phi4' if route=='ollama' else 'deepseek-flash') for stage in __import__('interview_engine.session_models',fromlist=['STAGES']).STAGES};self.observed=calls if calls is not None else []
    def candidate_for_route(self,route,models):
        other=SwitchableProgram(route,self.observed);other.model_map=models;other.pending=self.pending;return other
    def advance(self,response=None):
        if response is None:return self.pending
        self.observed.append(self.model_route);self.finished=True;return None

def test_explicit_route_switch_is_typed_idempotent_and_preserves_input(monkeypatch,tmp_path):
    from interview_engine import session_models
    monkeypatch.setattr(session_models,'validate',lambda models,route=None:None)
    calls=[];factory=lambda sid,request:SwitchableProgram(request.model_route,calls)
    registry=SessionRegistry(factory,state_dir=tmp_path);client=authed(create_app(registry,service_token=TOKEN))
    start=client.post('/v1/sessions',json={'task_id':'t','idea':'idea','model_route':'ollama'}).json();state=start['snapshot'];sid=state['session_id']
    body={'model_route':'deepseek','model':'deepseek-flash','expected_sequence':state['sequence'],'expected_input_id':state['open_input']['input_id'],'idempotency_key':str(uuid4()),'actor_id':'founder'}
    route=client.post(f'/v1/sessions/{sid}/model-route',json=body)
    assert route.status_code==200,route.text
    current=route.json()['snapshot'];assert current['model_route']=='deepseek' and current['route_version']==1
    assert current['open_input']['input_id']==state['open_input']['input_id']
    assert route.json()['accepted_event']['event_type']==EventType.ROUTE_CHANGED
    replay=client.post(f'/v1/sessions/{sid}/model-route',json=body)
    assert replay.status_code==200 and len(replay.json()['snapshot']['route_history'])==1
    assert client.post(f'/v1/sessions/{sid}/model-route',json={**body,'idempotency_key':str(uuid4())}).status_code==409
    assert client.post(f'/v1/sessions/{sid}/inputs/{state["open_input"]["input_id"]}',json={'input_id':state['open_input']['input_id'],'value':'answer','actor_id':'founder'}).status_code==200
    assert calls==['deepseek']
    saved=__import__('json').loads((tmp_path/sid/'interview'/'http-session.json').read_text())
    assert saved['route_version']==1 and len(saved['route_history'])==1


def test_route_switch_rejected_while_not_waiting(monkeypatch):
    from interview_engine import session_models
    monkeypatch.setattr(session_models,'validate',lambda models,route=None:None)
    registry=SessionRegistry();client=authed(create_app(registry,service_token=TOKEN))
    start=client.post('/v1/sessions',json={'task_id':'t','idea':'idea','model_route':'ollama'}).json()['snapshot'];sid=start['session_id']
    body={'model_route':'deepseek','model':'deepseek-flash','expected_sequence':start['sequence'],'expected_input_id':str(uuid4()),'idempotency_key':str(uuid4()),'actor_id':'founder'}
    response=client.post(f'/v1/sessions/{sid}/model-route',json=body)
    assert response.status_code==409 and response.json()['detail']['code']==ProtocolErrorCode.ROUTE_SWITCH_CONFLICT

def test_switched_session_restores_route_input_and_idempotency(monkeypatch,tmp_path):
    from interview_engine import session_models
    monkeypatch.setattr(session_models,'validate',lambda models,route=None:None)
    def factory(sid,request):
        program=SwitchableProgram(request.model_route)
        path=tmp_path/sid/'interview'/'http-session.json'
        if path.exists():
            stored=__import__('json').loads(path.read_text())
            program.model_map=stored['model_map']
        return program
    first=SessionRegistry(factory,state_dir=tmp_path);client=authed(create_app(first,service_token=TOKEN))
    started=client.post('/v1/sessions',json={'task_id':'task','idea':'idea','model_route':'ollama'}).json()['snapshot'];sid=started['session_id']
    body={'model_route':'deepseek','model':'deepseek-flash','expected_sequence':started['sequence'],'expected_input_id':started['open_input']['input_id'],
          'idempotency_key':str(uuid4()),'actor_id':'founder'}
    changed=client.post(f'/v1/sessions/{sid}/model-route',json=body)
    assert changed.status_code==200
    second=SessionRegistry(factory,state_dir=tmp_path);restored=second.get(sid)
    assert restored.snapshot().model_route=='deepseek' and restored.snapshot().route_version==1
    assert restored.open_input.input_id==__import__('uuid').UUID(started['open_input']['input_id'])
    retry=authed(create_app(second,service_token=TOKEN)).post(f'/v1/sessions/{sid}/model-route',json=body)
    assert retry.status_code==200
    assert retry.json()['accepted_event']['sequence']==changed.json()['accepted_event']['sequence']
    assert len(retry.json()['snapshot']['route_history'])==1


def test_legacy_qwen_pin_can_switch_to_phi4_same_provider(monkeypatch,tmp_path):
    from interview_engine import session_models
    monkeypatch.setattr(session_models,'validate',lambda models,route=None:None)
    class OldProgram(SwitchableProgram):
        def __init__(self,route='ollama'):
            super().__init__(route)
            self.model_map={stage:('ollama','qwen2.5-coder:14b') for stage in session_models.STAGES}
        def candidate_for_route(self,route,models):
            other=SwitchableProgram(route);other.model_map=models;other.pending=self.pending;return other
    first=SessionRegistry(lambda sid,request:OldProgram(),state_dir=tmp_path);client=authed(create_app(first,service_token=TOKEN))
    old=client.post('/v1/sessions',json={'task_id':'pre-existing','idea':'idea','model_route':'ollama'}).json()['snapshot']
    sid=old['session_id'];path=tmp_path/sid/'interview'/'http-session.json'
    saved=__import__('json').loads(path.read_text());saved.pop('route_version',None);saved.pop('route_history',None);path.write_text(__import__('json').dumps(saved))
    def factory(sid,request):
        value=__import__('json').loads((tmp_path/sid/'interview'/'http-session.json').read_text()) if (tmp_path/sid/'interview'/'http-session.json').exists() else {}
        program=OldProgram()
        if value.get('model_map'):program.model_map=value['model_map']
        return program
    restarted=SessionRegistry(factory,state_dir=tmp_path);other=authed(create_app(restarted,service_token=TOKEN))
    before=other.get(f'/v1/sessions/{sid}').json()
    assert before['route_version']==0 and before['model_map']['probing']==['ollama','qwen2.5-coder:14b']
    response=other.post(f'/v1/sessions/{sid}/model-route',json={'model_route':'ollama','model':'phi4','expected_sequence':before['sequence'],
      'expected_input_id':before['open_input']['input_id'],'idempotency_key':str(uuid4()),'actor_id':'founder'})
    assert response.status_code==200,response.text
    assert response.json()['snapshot']['model_map']['probing']==['ollama','phi4']
    assert response.json()['snapshot']['open_input']['input_id']==before['open_input']['input_id']

def test_failed_legacy_input_stays_switchable_after_restart(monkeypatch,tmp_path):
    from interview_engine import session_models
    monkeypatch.setattr(session_models,'validate',lambda models,route=None:None)
    class LegacyError(SwitchableProgram):
        def __init__(self):
            super().__init__('ollama');self.model_map={stage:('ollama','qwen2.5-coder:14b') for stage in session_models.STAGES}
        def advance(self,response=None):
            if response is None:return self.pending
            raise RuntimeError('qwen returned a template literal')
        def candidate_for_route(self,route,models):
            new=SwitchableProgram(route);new.pending=self.pending;new.model_map=models;return new
    factory=lambda sid,request:LegacyError()
    registry=SessionRegistry(factory,state_dir=tmp_path);client=authed(create_app(registry,service_token=TOKEN))
    before=client.post('/v1/sessions',json={'task_id':'old','idea':'idea','model_route':'ollama'}).json()['snapshot'];sid=before['session_id']
    bad=client.post(f'/v1/sessions/{sid}/inputs/{before["open_input"]["input_id"]}',json={
      'input_id':before['open_input']['input_id'],'value':'answer','actor_id':'founder'})
    assert bad.status_code==503
    restored=SessionRegistry(factory,state_dir=tmp_path);state=authed(create_app(restored,service_token=TOKEN)).get(f'/v1/sessions/{sid}').json()
    assert state['process_status']==ProcessStatus.WAITING_INPUT and state['open_input']['input_id']==before['open_input']['input_id']
    switched=restored.switch_route(sid,__import__('vmodel_contracts.interactive',fromlist=['ModelRouteSwitchRequest']).ModelRouteSwitchRequest(
      model_route='ollama',model='phi4',expected_sequence=state['sequence'],expected_input_id=state['open_input']['input_id'],
      idempotency_key=uuid4(),actor_id='founder'))
    assert switched[1].model_map['probing']==('ollama','phi4')


def test_restored_old_pin_does_not_preflight_retired_model(monkeypatch,tmp_path):
    from interview_engine import session_models
    from interview_engine.errors import UserFacingError
    from interview_engine.http_service import default_program_factory
    from vmodel_contracts.interactive import StartSessionRequest
    old={stage:["ollama","qwen2.5-coder:14b"] for stage in session_models.STAGES}
    path=tmp_path/"legacy"/"interview"/"http-session.json";path.parent.mkdir(parents=True);path.write_text(__import__("json").dumps({"model_map":old}))
    monkeypatch.setattr("interview_engine.http_service.artifacts_root",lambda arg=None:tmp_path)
    monkeypatch.setattr("interview_engine.application.artifacts_root",lambda arg=None:tmp_path)
    calls=[]
    def preflight(cfg,stage):
        calls.append(cfg.model)
        raise UserFacingError("retired model", "not installed", "choose another model")
    monkeypatch.setattr("interview_engine.model_config._preflight",preflight)
    class StubApplication:
        def __init__(self,*args,**kwargs):
            session_models.validate(kwargs["pinned_models"],kwargs["model_route"],preflight=False)
            self.model_map=kwargs["pinned_models"]
    monkeypatch.setattr("interview_engine.application.InterviewApplication",StubApplication)
    restored=default_program_factory("legacy",StartSessionRequest(task_id="old",idea="idea",model_route="ollama"))
    assert restored.model_map==old and calls==[]


def test_expected_unpinned_legacy_snapshot_logs_one_info_without_traceback(tmp_path,caplog):
 import json,logging
 from interview_engine.errors import UserFacingError
 from interview_engine.http_service import default_program_factory
 from vmodel_contracts.interactive import StartSessionRequest
 path=tmp_path/'old'/'interview'/'http-session.json';path.parent.mkdir(parents=True)
 path.write_text(json.dumps({'session_id':'old','task_id':'legacy-task','phase_id':1,'sequence':2,
  'status':40,'next_step':40,'open_input':None,'responses':[],
  'request':StartSessionRequest(task_id='legacy-task',idea='old idea',model_route='ollama').model_dump(mode='json')}))
 def factory(sid,request):
  from interview_engine import http_service
  # The real restore guard is in default_program_factory, but it needs only
  # the persisted session file, no model provider or graph to reject safely.
  return default_program_factory(sid,request)
 from interview_engine import http_service
 original=http_service.artifacts_root
 http_service.artifacts_root=lambda arg=None:tmp_path
 try:
  client=authed(create_app(SessionRegistry(factory,state_dir=tmp_path),service_token=TOKEN))
  with caplog.at_level(logging.INFO,logger='interview_engine.http'):
   response=client.get('/v1/sessions/old')
 finally:http_service.artifacts_root=original
 assert response.status_code==422
 assert 'legacy session has no pinned model route' in response.json()['detail']['detail']
 lines=[record for record in caplog.records if record.name=='interview_engine.http']
 assert len(lines)==1 and lines[0].levelno==logging.INFO
 assert 'legacy_unpinned' in lines[0].getMessage() and not lines[0].exc_info
def test_sparse_conops_failure_is_typed_nonretryable_with_way_forward():
    from interview_engine.errors import UserFacingError
    class SparseProgram(FakeProgram):
        def advance(self,response=None):
            if response is None:return self.pending
            raise UserFacingError('CONOPS cannot start: the interview lacks stakeholder and scenario evidence.',
                'validated intent has 0 stakeholder context seeds.',
                'Start a new interview and answer who uses the product and a real-world workflow.')
    program=SparseProgram()
    client=authed(create_app(SessionRegistry(lambda session_id,request:program),service_token=TOKEN))
    started=client.post('/v1/sessions',json={'protocol_version':1,'task_id':'t','idea':'i','model_route':'m'}).json()
    pending=started['snapshot']['open_input']
    failed=client.post(f"/v1/sessions/{started['snapshot']['session_id']}/inputs/{pending['input_id']}",
                       json={'protocol_version':1,'input_id':pending['input_id'],'value':'finish','actor_id':'founder'})
    detail=failed.json()['detail']
    assert failed.status_code==422 and detail['code']==ProtocolErrorCode.INVALID_RESPONSE
    assert detail['retryable'] is False and 'Start a new interview' in detail['detail']

class BlockingProgram(FakeProgram):
    def __init__(self,entered,release):
        super().__init__();self.entered=entered;self.release=release
    def advance(self,response=None):
        if response is None:return self.pending
        self.entered.set()
        if not self.release.wait(5):raise RuntimeError('test model call never released')
        self.finished=True;return None


def test_model_advance_does_not_block_snapshot_other_sessions_or_progress(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from vmodel_contracts.interactive import StartSessionRequest,InputResponse
    entered=Event();release=Event();program=BlockingProgram(entered,release)
    registry=SessionRegistry(lambda sid,request:program,state_dir=tmp_path)
    first,_=registry.start(StartSessionRequest(task_id='first',idea='first',model_route='deepseek'))
    other,_=registry.start(StartSessionRequest(task_id='second',idea='second',model_route='deepseek'))
    prior=registry.snapshot(first.session_id);pending=prior.open_input
    response=InputResponse(input_id=pending.input_id,expected_sequence=prior.sequence,value='yes',actor_id='founder')
    with ThreadPoolExecutor(max_workers=3) as executor:
        submitted=executor.submit(registry.respond,first.session_id,response)
        try:
            assert entered.wait(2)
            # No control read can expose the old question while it is being answered.
            current=executor.submit(registry.snapshot,first.session_id).result(timeout=.5)
            assert current.process_status==ProcessStatus.RUNNING and current.open_input is None
            assert current.sequence==prior.sequence+1
            assert registry.snapshot(other.session_id).task_id=='second'
            # Concurrent duplicate writes serialize behind the advance.
            duplicate=executor.submit(registry.respond,first.session_id,response)
            import time
            time.sleep(.03);assert not duplicate.done()
            from interview_engine.progress_journal import ProgressJournal
            ProgressJournal(tmp_path,first.session_id,'first').append(phase_key='intent',stage='probe',label='still working',status='heartbeat')
            page=executor.submit(registry.progress,first.session_id,0).result(timeout=.5)
            assert page.events
        finally:release.set()
        accepted,done=submitted.result(timeout=2)
        _,replayed=duplicate.result(timeout=2)
    assert done.process_status==ProcessStatus.COMPLETED and replayed.sequence==done.sequence
    saved=__import__('json').loads((tmp_path/first.session_id/'interview'/'http-session.json').read_text())
    assert len(saved['responses'])==1 and saved['sequence']==done.sequence
    assert registry.snapshot(first.session_id).sequence==done.sequence


def test_blocked_model_start_exposes_running_snapshot_and_other_session(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from vmodel_contracts.interactive import StartSessionRequest
    entered=Event();release=Event()
    class BlockingStart(FakeProgram):
        def advance(self,response=None):
            entered.set()
            if not release.wait(5):raise RuntimeError('test start never released')
            return self.pending
    registry=SessionRegistry(lambda sid,request:BlockingStart(),state_dir=tmp_path)
    request=StartSessionRequest(task_id='first',idea='first',model_route='deepseek',session_id=uuid4())
    with ThreadPoolExecutor(max_workers=2) as executor:
        started=executor.submit(registry.start,request)
        try:
            assert entered.wait(2)
            snapshot=executor.submit(registry.snapshot,str(request.session_id)).result(timeout=.5)
            assert snapshot.process_status==ProcessStatus.RUNNING and snapshot.open_input is None
            other_request=StartSessionRequest(task_id='second',idea='second',model_route='deepseek',session_id=uuid4())
            other=executor.submit(registry.start,other_request)
            # Other start can reserve and publish without waiting for the first model.
            import time
            time.sleep(.03)
            assert registry.snapshot(str(other_request.session_id)).process_status==ProcessStatus.RUNNING
        finally:release.set()
        started.result(timeout=2);other.result(timeout=2)

def test_failed_blocked_advance_restores_published_input_and_sequence(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from vmodel_contracts.interactive import StartSessionRequest,InputResponse
    entered=Event();release=Event()
    class FailingBlocked(FakeProgram):
        def advance(self,response=None):
            if response is None:return self.pending
            entered.set()
            if not release.wait(5):raise RuntimeError('test model did not finish')
            raise RuntimeError('provider rejected the call')
    registry=SessionRegistry(lambda sid,request:FailingBlocked(),state_dir=tmp_path)
    session,_=registry.start(StartSessionRequest(task_id='first',idea='idea',model_route='deepseek'))
    before=registry.snapshot(session.session_id)
    response=InputResponse(input_id=before.open_input.input_id,expected_sequence=before.sequence,value='reply',actor_id='founder')
    with ThreadPoolExecutor(max_workers=2) as pool:
        work=pool.submit(registry.respond,session.session_id,response)
        try:
            assert entered.wait(2)
            running=pool.submit(registry.snapshot,session.session_id).result(timeout=.5)
            assert running.process_status==ProcessStatus.RUNNING and running.open_input is None
        finally:release.set()
        import pytest
        with pytest.raises(RuntimeError,match='provider rejected'):work.result(timeout=2)
    after=registry.snapshot(session.session_id)
    assert after.sequence==before.sequence and after.open_input==before.open_input
    assert after.process_status==ProcessStatus.WAITING_INPUT
    assert not session.responses
    assert __import__('json').loads((tmp_path/session.session_id/'interview'/'http-session.json').read_text())['sequence']==before.sequence


def test_snapshot_read_cannot_mutate_published_control():
    from vmodel_contracts.interactive import StartSessionRequest
    registry=SessionRegistry(lambda sid,request:FakeProgram())
    session,_=registry.start(StartSessionRequest(task_id='first',idea='idea',model_route='deepseek'))
    read=registry.snapshot(session.session_id)
    read.open_input.options.append(InputOption(value='bogus',label='Bogus'))
    assert not registry.snapshot(session.session_id).open_input.options

def test_blocked_http_submit_allows_concurrent_http_snapshot_and_progress(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from interview_engine.progress_journal import ProgressJournal
    entered=Event();release=Event();registry=SessionRegistry(lambda sid,req:BlockingProgram(entered,release),state_dir=tmp_path)
    app=create_app(registry,service_token=TOKEN)
    started=authed(app).post('/v1/sessions',json={'task_id':'t','idea':'i','model_route':'deepseek'}).json()['snapshot']
    sid=started['session_id'];input_id=started['open_input']['input_id']
    ProgressJournal(tmp_path,sid,'t').append(phase_key='intent',stage='probing',label='Started',status='started')
    with ThreadPoolExecutor(max_workers=3) as pool:
        work=pool.submit(lambda:authed(app).post(f'/v1/sessions/{sid}/inputs/{input_id}',json={
            'input_id':input_id,'expected_sequence':started['sequence'],'value':'yes','actor_id':'founder'}))
        try:
            assert entered.wait(2)
            state=pool.submit(lambda:authed(app).get(f'/v1/sessions/{sid}')).result(timeout=.75)
            page=pool.submit(lambda:authed(app).get(f'/v1/sessions/{sid}/progress')).result(timeout=.75)
            assert state.status_code==200 and state.json()['process_status']==ProcessStatus.RUNNING
            assert state.json()['open_input'] is None
            assert page.status_code==200 and page.json()['events']
        finally:release.set()
        assert work.result(timeout=2).status_code==200
