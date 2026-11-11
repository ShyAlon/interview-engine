"""The answer checkpoint and its pointer have one commit boundary."""
from types import SimpleNamespace
import pytest
from test_pg_checkpoint import make_saver,server,cfg,cp,META
from interview_engine.pg_checkpoint import answer_checkpoint_memory
from interview_engine.pg_session_store import PgSessionStore,answer_publication
from langgraph.checkpoint.postgres import PostgresSaver

def test_checkpoint_defers_until_fenced_publication_and_rolls_back_with_it(tmp_path):
    from interview_engine.pg_checkpoint import answer_checkpoint_publication
    with answer_checkpoint_memory(),answer_checkpoint_publication() as batch:
        ctx,saver,admin=make_saver(server(tmp_path).get_uri())
        store=PgSessionStore(saver.conn)
        try:
            admin.execute("UPDATE engine_sessions SET control='{}'::jsonb,lease_expires_at=0 WHERE session_id='s1'")
            claim=store.claim('s1','00000000-0000-0000-0000-000000000001','child',60)
            saver._lease.epoch=claim.epoch
            store.publish_step(claim,{'sequence':1,'open_input':{'input_id':'wait'}})
            written=saver.put(cfg('t'),cp(1),META,{'messages':'1'})
            assert saver.get_tuple(written).checkpoint['id']==cp(1)['id']
            assert PostgresSaver.get_tuple(saver,written) is None,'not durable before final publication'
            op=SimpleNamespace(operation_id='a1',binding=SimpleNamespace(session_id='s1',task_id='task1',owner_id=claim.owner_id,generation=1),operation=SimpleNamespace(kind='interview.answer',expected_sequence=1,wait_id='wrong'))
            from interview_engine.stop_fence import StopRequested
            with answer_publication(op),pytest.raises(StopRequested):
                store.publish_step(claim,{'sequence':2},published_checkpoint_id=cp(1)['id'])
            assert PostgresSaver.get_tuple(saver,written) is None,'failed pointer commit must roll back checkpoint too'
            op.operation.wait_id='wait'
            with answer_publication(op):store.publish_step(claim,{'sequence':2},published_checkpoint_id=cp(1)['id'])
            assert PostgresSaver.get_tuple(saver,written) is not None
            assert store.load_published('s1',claim.owner_id)['published_checkpoint_id']==cp(1)['id']
        finally:ctx.__exit__(None,None,None)


def test_staged_pending_writes_preserve_same_channel_slots(tmp_path):
    from interview_engine.pg_checkpoint import answer_checkpoint_publication
    from langgraph.checkpoint.base import INTERRUPT
    with answer_checkpoint_memory(),answer_checkpoint_publication() as batch:
        ctx,saver,admin=make_saver(server(tmp_path).get_uri())
        try:
            one=saver.put(cfg('t'),cp(1),META,{'messages':'1'})
            saver.put_writes(one,[('messages','first'),('messages','second')],'A')
            saver.put_writes(one,[('messages','overwrite')],'A')
            saver.put_writes(one,[(INTERRUPT,'before')],'B')
            saver.put_writes(one,[(INTERRUPT,'after')],'B')
            expected=saver.get_tuple(one)
            with saver._guarded_conn() as conn:batch.flush(conn)
            assert expected==PostgresSaver.get_tuple(saver,one)
            assert [w[2] for w in expected.pending_writes]==['first','second','after']
        finally:ctx.__exit__(None,None,None)


def test_staged_blob_collision_rejects_instead_of_publishing_proposed_bytes(tmp_path):
    from interview_engine.pg_checkpoint import answer_checkpoint_publication
    with answer_checkpoint_memory():ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        first=saver.put(cfg('t'),cp(1),META,{'messages':'1'})
        with answer_checkpoint_publication(saver) as batch:
            proposed=cp(2);proposed['channel_versions']['messages']='1';proposed['channel_values']['messages']=['bad']
            second=saver.put(first,proposed,META,{'messages':'1'})
            with pytest.raises(ValueError,match='immutable blob collision'):
                with saver._guarded_conn() as conn:batch.flush(conn)
        assert PostgresSaver.get_tuple(saver,second) is None
    finally:ctx.__exit__(None,None,None)


def test_staged_mixed_reserved_regular_writes_keep_actual_indices(tmp_path):
    from interview_engine.pg_checkpoint import answer_checkpoint_publication
    from langgraph.checkpoint.base import INTERRUPT
    with answer_checkpoint_memory(),answer_checkpoint_publication() as batch:
        ctx,saver,admin=make_saver(server(tmp_path).get_uri())
        try:
            one=saver.put(cfg('t'),cp(1),META,{'messages':'1'})
            saver.put_writes(one,[(INTERRUPT,'signal'),('messages','slot-one')],'A')
            saver.put_writes(one,[('messages','slot-zero')],'A')
            expected=saver.get_tuple(one)
            with saver._guarded_conn() as conn:batch.flush(conn)
            assert expected==PostgresSaver.get_tuple(saver,one)
            assert [w[2] for w in expected.pending_writes]==['signal','slot-zero','slot-one']
        finally:ctx.__exit__(None,None,None)


def test_staged_restore_pending_indices_and_replay_do_not_create_orphans(tmp_path):
    from interview_engine.pg_checkpoint import answer_checkpoint_publication
    from interview_engine.stop_fence import StopRequested
    from langgraph.checkpoint.base import INTERRUPT
    with answer_checkpoint_memory():ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    store=PgSessionStore(saver.conn)
    try:
        admin.execute("UPDATE engine_sessions SET control='{}'::jsonb,lease_expires_at=0 WHERE session_id='s1'")
        claim=store.claim('s1','00000000-0000-0000-0000-000000000001','child',60)
        saver._lease.epoch=claim.epoch
        first=saver.put(cfg('t'),cp(1),META,{'messages':'1'})
        saver.put_writes(first,[(INTERRUPT,'signal'),('messages','slot-one')],'A')
        # Force a real durable restore with a reserved-channel gap.
        saver._answer_memory.clear()
        with answer_checkpoint_publication(saver) as batch:
            saver.put_writes(first,[('messages','slot-zero')],'A')
            expected=saver.get_tuple(first)
            with saver._guarded_conn() as conn:batch.flush(conn)
            assert expected==PostgresSaver.get_tuple(saver,first)
        store.publish_step(claim,{'sequence':1,'open_input':{'input_id':'wait'}},published_checkpoint_id=cp(1)['id'])
        op=SimpleNamespace(operation_id='a1',binding=SimpleNamespace(session_id='s1',task_id='task1',owner_id=claim.owner_id,generation=1),operation=SimpleNamespace(kind='interview.answer',expected_sequence=1,wait_id='wait'))
        with answer_checkpoint_publication(saver),answer_publication(op):
            second=saver.put(first,cp(2),META,{'messages':'2'})
            store.publish_step(claim,{'sequence':2},published_checkpoint_id=cp(2)['id'])
            store.publish_step(claim,{'sequence':2},published_checkpoint_id='ignored')
        with answer_checkpoint_publication(saver),answer_publication(op):
            third=saver.put(second,cp(3),META,{'messages':'3'})
            with pytest.raises(StopRequested):
                store.publish_step(claim,{'sequence':2},published_checkpoint_id=cp(3)['id'])
        assert PostgresSaver.get_tuple(saver,third) is None,'fresh replay may not leave unpublished staged checkpoints'
        assert store.load_published('s1',claim.owner_id)['published_checkpoint_id']==cp(2)['id']
    finally:ctx.__exit__(None,None,None)


def test_committed_warm_turns_keep_only_latest_and_parent_in_memory(tmp_path):
    from interview_engine.pg_checkpoint import answer_checkpoint_publication
    with answer_checkpoint_memory():ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    store=PgSessionStore(saver.conn)
    try:
        admin.execute("UPDATE engine_sessions SET control='{}'::jsonb,lease_expires_at=0 WHERE session_id='s1'")
        claim=store.claim('s1','00000000-0000-0000-0000-000000000001','child',60)
        saver._lease.epoch=claim.epoch
        store.publish_step(claim,{'sequence':1,'open_input':{'input_id':'wait1'}})
        previous=cfg('t');configs=[]
        for turn in range(1,7):
            op=SimpleNamespace(operation_id='a'+str(turn),binding=SimpleNamespace(session_id='s1',task_id='task1',owner_id=claim.owner_id,generation=1),operation=SimpleNamespace(kind='interview.answer',expected_sequence=turn,wait_id='wait'+str(turn)))
            with answer_checkpoint_publication(saver),answer_publication(op):
                previous=saver.put(previous,cp(turn),META,{'messages':str(turn)})
                configs.append(previous)
                store.publish_step(claim,{'sequence':turn+1,'open_input':{'input_id':'wait'+str(turn+1)}},published_checkpoint_id=cp(turn)['id'])
            assert len(saver._answer_memory)<=2,'warm child must not retain the full checkpoint history'
            assert saver.get_tuple(cfg('t')).checkpoint['id']==cp(turn)['id']
        assert saver.get_tuple(configs[0])==PostgresSaver.get_tuple(saver,configs[0]),'evicted history still resolves durably'
    finally:ctx.__exit__(None,None,None)
