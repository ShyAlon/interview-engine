"""Actual typed start/answer registry operations with fake graph programs."""
import json,time
from uuid import uuid4
import pytest
from requi_child_contract import decode_operation
from interview_engine.child_operations import execute_operation
from interview_engine.child_step import published_digest
from test_pg_session_registry import make_conn,make_pool,OWNER,_cleanup_pg
from interview_engine.pg_session_store import PgSessionStore
from vmodel_contracts.interactive import StartSessionRequest,InputRequest,InputKind,InputResponse


class Program:
    finished=False
    def advance(self,response=None):return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='next fake question')


def op(sid,body,epoch=1):
    return decode_operation(json.dumps({'version':1,'operation_id':'op-'+str(uuid4()),'binding':{
        'owner_id':OWNER,'auth_session_id':'00000000-0000-0000-0000-000000000002',
        'task_id':'fake','session_id':sid,'job_id':'job-1','generation':1,'lease_epoch':epoch,
        'provider':'ollama','model':'fake'},'operation':body}).encode()+b'\n')


def test_typed_start_then_durable_answer_uses_real_registry_and_releases(tmp_path):
    conn=make_conn(tmp_path);store=PgSessionStore(make_pool(conn._pg_server),claims=None);sid=str(uuid4())
    store.create(sid,'fake',OWNER);bootstrap=store.claim(sid,OWNER,'bootstrap',60);store.release(bootstrap)
    store.claims=(OWNER,'00000000-0000-0000-0000-000000000002')
    request=StartSessionRequest(session_id=sid,task_id='fake',idea='fake',model_route='local')
    start=op(sid,{'kind':'interview.start','request_ref':'request-1'})
    snapshot=execute_operation(start,trusted_binding=start.binding,store=store,
        load_request=lambda ref:request,load_answer=lambda ref:None,program_factory=lambda *a:Program())
    assert snapshot.open_input and snapshot.sequence==2
    row=store.load_published(sid,OWNER);wait=snapshot.open_input
    answer=InputResponse(input_id=wait.input_id,value='fake answer',actor_id='fake',expected_sequence=2)
    answer_op=op(sid,{'kind':'interview.answer','checkpoint_sha256':published_digest(row),'expected_sequence':2,
        'wait_id':str(wait.input_id),'review_revision':'engine:'+sid+':2','answer_ref':'answer-1'},epoch=2)
    result=execute_operation(answer_op,trusted_binding=answer_op.binding,store=store,
        load_request=lambda ref:None,load_answer=lambda ref:answer,program_factory=lambda *a:Program())
    assert result.sequence==4 and result.open_input.input_id!=wait.input_id
    assert store.claim(sid,OWNER,'after-answer',60) is not None


def test_start_rejects_changed_preclaim_epoch_before_graph(tmp_path):
    conn=make_conn(tmp_path);store=PgSessionStore(make_pool(conn._pg_server),claims=None);sid=str(uuid4())
    store.create(sid,'fake',OWNER);claim=store.claim(sid,OWNER,'bootstrap',60);store.release(claim)
    store.claims=(OWNER,'00000000-0000-0000-0000-000000000002')
    operation=op(sid,{'kind':'interview.start','request_ref':'request-1'},epoch=99);calls=[]
    with pytest.raises(PermissionError):execute_operation(operation,trusted_binding=operation.binding,store=store,
        load_request=lambda ref:StartSessionRequest(session_id=sid,task_id='fake',idea='fake',model_route='local'),
        load_answer=lambda ref:None,program_factory=lambda *a:calls.append(1))
    assert calls==[]
