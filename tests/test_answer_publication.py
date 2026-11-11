from types import SimpleNamespace
import pytest
from test_pg_session_store import make_store,OWNER_A
from interview_engine.pg_session_store import answer_publication
from interview_engine.stop_fence import StopRequested

def test_final_answer_atomic_identity_fence_and_same_answer_noop(tmp_path):
    store=make_store(tmp_path);store.create('s1','task1',OWNER_A)
    claim=store.claim('s1',OWNER_A,'child',60)
    store.publish_step(claim,{'sequence':1,'open_input':{'input_id':'wait'}})
    op=SimpleNamespace(operation_id='answer1',binding=SimpleNamespace(session_id='s1',task_id='task1',owner_id=OWNER_A,generation=1),operation=SimpleNamespace(kind='interview.answer',expected_sequence=1,wait_id='wait'))
    with answer_publication(op):
        store.publish_step(claim,{'sequence':2,'open_input':{'input_id':'next'}},published_checkpoint_id='cp2')
        # Same logical answer is a no-op, not permission to replace its result.
        store.publish_step(claim,{'sequence':2,'open_input':None},published_checkpoint_id='different')
    row=store.load_published('s1',OWNER_A)
    assert row['control']['open_input']['input_id']=='next'
    assert row['published_checkpoint_id']=='cp2'
    op.operation_id='answer-other'
    with answer_publication(op),pytest.raises(StopRequested):
        store.publish_step(claim,{'sequence':3},published_checkpoint_id='bad')
    op.operation.expected_sequence=2;op.operation.wait_id='wrong'
    with answer_publication(op),pytest.raises(StopRequested):store.publish_step(claim,{'sequence':3})
    store.stop('s1',OWNER_A)
    op.operation.wait_id='next'
    with answer_publication(op),pytest.raises(StopRequested):store.publish_step(claim,{'sequence':3})
    assert store.load_published('s1',OWNER_A)['published_checkpoint_id']=='cp2'
