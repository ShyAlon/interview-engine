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
    body={'model_route':'deepseek','expected_sequence':state['sequence'],'expected_input_id':state['open_input']['input_id'],'idempotency_key':str(uuid4()),'actor_id':'founder'}
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
    body={'model_route':'deepseek','expected_sequence':start['sequence'],'expected_input_id':str(uuid4()),'idempotency_key':str(uuid4()),'actor_id':'founder'}
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
    body={'model_route':'deepseek','expected_sequence':started['sequence'],'expected_input_id':started['open_input']['input_id'],
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
