from uuid import uuid4
from fastapi.testclient import TestClient
from vmodel_contracts.interactive import (
    EventType,InputKind,InputOption,InputRequest,NextStep,ProcessStatus,ProtocolErrorCode,
)
from interview_engine.http_service import SessionRegistry,create_app


def test_start_session_returns_typed_http_contract():
    client=TestClient(create_app());r=client.post('/v1/sessions',json={
        'protocol_version':1,'task_id':'task-1','idea':'A virtual dungeon master','model_route':'deepseek'})
    assert r.status_code==201
    body=r.json();assert body['accepted_event']['event_type']==EventType.RUN_STARTED
    assert body['snapshot']['process_status']==ProcessStatus.RUNNING
    assert body['snapshot']['next_step']==NextStep.RUN


def test_approval_is_single_idempotent_input_boundary():
    registry=SessionRegistry();client=TestClient(create_app(registry));started=client.post('/v1/sessions',json={
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
    registry=SessionRegistry();client=TestClient(create_app(registry));sid=client.post('/v1/sessions',json={
        'protocol_version':1,'task_id':'t','idea':'i','model_route':'m'}).json()['snapshot']['session_id']
    current=InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Who uses it?')
    registry.publish(sid,EventType.INPUT_REQUESTED,phase_id=10,status=ProcessStatus.WAITING_INPUT,
        next_step=NextStep.WAIT_INPUT,input_request=current)
    stale=uuid4();r=client.post(f'/v1/sessions/{sid}/inputs/{stale}',json={
        'protocol_version':1,'input_id':str(stale),'value':'players','actor_id':'founder'})
    assert r.status_code==409 and r.json()['detail']['code']==ProtocolErrorCode.STALE_INPUT
