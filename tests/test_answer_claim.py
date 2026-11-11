from types import SimpleNamespace
import pytest
from test_pg_checkpoint import make_saver,server
from interview_engine.pg_session_store import PgSessionStore,answer_publication
from interview_engine.pg_checkpoint import FencedPostgresSaver

def test_cold_claim_atomic_input_epoch_and_startup_schema(tmp_path):
    ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    store=PgSessionStore(saver.conn)
    try:
        admin.execute("UPDATE engine_sessions SET lease_expires_at=0,control=%s::jsonb WHERE session_id='s1'",('{"sequence":1,"open_input":{"input_id":"wait"}}',))
        binding=SimpleNamespace(session_id='s1',owner_id='00000000-0000-0000-0000-000000000001',task_id='task1',generation=1,lease_epoch=1)
        body=SimpleNamespace(expected_sequence=1,wait_id='wrong',review_revision='engine:s1:1')
        with pytest.raises(PermissionError):store._claim_answer_expected(binding,body,'child',60)
        assert admin.execute("SELECT lease_epoch FROM engine_sessions WHERE session_id='s1'").fetchone()['lease_epoch']==1
        body.wait_id='wait'
        claim,row=store._claim_answer_expected(binding,body,'child',60)
        assert claim.epoch==2 and row['control']['sequence']==1
        assert store._checkpoint_schema_version>=len(FencedPostgresSaver.MIGRATIONS)-1
        with pytest.raises(PermissionError):store._claim_answer_expected(binding,body,'racer',60)
        store.release(claim)
        binding.lease_epoch=2
        admin.execute('DELETE FROM checkpoint_migrations')
        with pytest.raises(RuntimeError,match='schema is not prepared'):store._claim_answer_expected(binding,body,'child2',60)
        assert admin.execute("SELECT lease_epoch FROM engine_sessions WHERE session_id='s1'").fetchone()['lease_epoch']==2
    finally:ctx.__exit__(None,None,None)


def test_claim_returns_exact_published_checkpoint_never_saver_latest(tmp_path):
    from test_pg_checkpoint import cfg,cp,META
    from langgraph.checkpoint.postgres import PostgresSaver
    ctx,saver,admin=make_saver(server(tmp_path).get_uri());store=PgSessionStore(saver.conn)
    try:
        one=saver.put(cfg('s1'),cp(1),META,{'messages':'1'})
        saver.put(one,cp(2),META,{'messages':'2'})
        admin.execute("UPDATE engine_sessions SET lease_expires_at=0,control=%s::jsonb,published_thread_id='s1',published_checkpoint_id=%s WHERE session_id='s1'",
            ('{"sequence":1,"open_input":{"input_id":"wait"}}',cp(1)['id']))
        binding=SimpleNamespace(session_id='s1',owner_id='00000000-0000-0000-0000-000000000001',task_id='task1',generation=1,lease_epoch=1)
        body=SimpleNamespace(expected_sequence=1,wait_id='wait',review_revision='engine:s1:1')
        claim,row=store._claim_answer_expected(binding,body,'child',60)
        returned=saver._load_checkpoint_tuple(store._answer_checkpoint_row)
        assert returned==PostgresSaver.get_tuple(saver,one)
        assert returned.checkpoint['id']==cp(1)['id']
    finally:ctx.__exit__(None,None,None)
