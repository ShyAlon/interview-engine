import time
import pytest

pgserver=pytest.importorskip('pgserver',reason='pgserver not installed: PG checkpoint tests skip')
from conftest import managed_pg_server
pgcp=pytest.importorskip('langgraph.checkpoint.postgres',reason='langgraph-checkpoint-postgres not installed')

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from interview_engine.pg_checkpoint import bootstrap_checkpoint_schema, FencedPostgresSaver,LeaseFence
from interview_engine.stop_fence import StopRequested

from test_pg_session_store import MIRRORED_REGISTRY_DDL as LEASE_DDL

def server(tmp_path):
    return managed_pg_server(str(tmp_path/'pgdata'))

def make_saver(uri,session='s1',generation=1,epoch=1,expire=3600):
    from conftest import managed_pool
    ctx=PostgresSaver.from_conn_string(uri)
    saver=ctx.__enter__()
    bootstrap_checkpoint_schema(saver.conn)
    admin=saver.conn
    admin.execute(LEASE_DDL)
    admin.execute("INSERT INTO engine_sessions(session_id,owner_id,task_id,generation,created_at,lease_epoch,lease_expires_at)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (session,'00000000-0000-0000-0000-000000000001','task1',generation,time.time(),epoch,time.time()+expire))
    pool=managed_pool(uri)
    fenced=FencedPostgresSaver(pool,lease=LeaseFence(session,generation,epoch),
        serde=JsonPlusSerializer())
    return ctx,fenced,saver.conn

def cp(i):
    return {'v':4,'ts':f'2026-10-22T08:0{i}:00+00:00','id':f'00000000-0000-4000-8000-{i:012d}',
            'channel_values':{'messages':[f'm{i}']},'channel_versions':{'messages':str(i)},
            'versions_seen':{},'updated_channels':['messages']}
META={'source':'update','step':1,'parents':{}}
def cfg(tid,cid=''):
    return {'configurable':{'thread_id':tid,'checkpoint_ns':'','checkpoint_id':cid}}

def test_fenced_writes_round_trip(tmp_path):
    ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        c1=saver.put(cfg('t'),cp(1),META,{'messages':1})
        saver.put_writes(c1,[('messages','w1')],'taskA')
        got=saver.get_tuple(c1)
        assert got is not None and [w[2] for w in got.pending_writes]==['w1']
    finally:ctx.__exit__(None,None,None)

def test_stale_generation_raises_and_persists_nothing(tmp_path):
    ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        c1=saver.put(cfg('t'),cp(1),META,{'messages':1})
        admin.execute("UPDATE engine_sessions SET generation=2 WHERE session_id='s1'")
        with pytest.raises(StopRequested):
            saver.put_writes(c1,[('messages','stale')],'taskB')
        rows=admin.execute("SELECT count(*) AS n FROM checkpoint_writes WHERE task_id='taskB'").fetchone()
        assert rows['n']==0
    finally:ctx.__exit__(None,None,None)

def test_same_key_collision_first_writer_wins(tmp_path):
    ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        c1=saver.put(cfg('t'),cp(1),META,{'messages':1})
        saver.put_writes(c1,[('messages','legit')],'taskA')
        saver.put_writes(c1,[('messages','overwrite-attempt')],'taskA')
        got=saver.get_tuple(c1)
        assert [w[2] for w in got.pending_writes]==['legit']
    finally:ctx.__exit__(None,None,None)

def test_reserved_channel_guarded_upsert(tmp_path):
    from langgraph.checkpoint.base import INTERRUPT
    ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        c1=saver.put(cfg('t'),cp(1),META,{'messages':1})
        saver.put_writes(c1,[(INTERRUPT,'first')],'taskA')
        saver.put_writes(c1,[(INTERRUPT,'updated')],'taskA')
        got=saver.get_tuple(c1)
        assert [w[2] for w in got.pending_writes if w[1]==INTERRUPT]==['updated']
    finally:ctx.__exit__(None,None,None)

def test_reserved_upsert_still_fenced(tmp_path):
    from langgraph.checkpoint.base import INTERRUPT
    ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        c1=saver.put(cfg('t'),cp(1),META,{'messages':1})
        saver.put_writes(c1,[(INTERRUPT,'first')],'taskA')
        admin.execute("UPDATE engine_sessions SET generation=2 WHERE session_id='s1'")
        with pytest.raises(StopRequested):
            saver.put_writes(c1,[(INTERRUPT,'stale-update')],'taskA')
        got=saver.get_tuple(c1)
        assert [w[2] for w in got.pending_writes if w[1]==INTERRUPT]==['first']
    finally:ctx.__exit__(None,None,None)

def test_takeover_keeps_prior_epoch_work(tmp_path):
    srv=server(tmp_path);ctx,saver,admin=make_saver(srv.get_uri())
    from conftest import managed_pool as _mp; pool=_mp(srv.get_uri())
    try:
        c1=saver.put(cfg('t'),cp(1),META,{'messages':1})
        saver.put_writes(c1,[('messages','epoch1-result')],'taskA')
        # takeover: bump + new lease identity, new saver handle
        admin.execute("UPDATE engine_sessions SET generation=2,lease_epoch=2 WHERE session_id='s1'")
        take=FencedPostgresSaver(pool,lease=LeaseFence('s1',2,2),serde=JsonPlusSerializer())
        take.put_writes(c1,[('messages','epoch2-result')],'taskB')
        got=take.get_tuple(c1)
        vals={w[0]:w[2] for w in got.pending_writes}
        assert vals=={'taskA':'epoch1-result','taskB':'epoch2-result'}
    finally:ctx.__exit__(None,None,None)

def test_expired_lease_raises(tmp_path):
    ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        c1=saver.put(cfg('t'),cp(1),META,{'messages':1})
        admin.execute("UPDATE engine_sessions SET lease_expires_at=%s WHERE session_id='s1'",(time.time()-1,))
        with pytest.raises(StopRequested):
            saver.put_writes(c1,[('messages','x')],'taskA')
        with pytest.raises(StopRequested):
            saver.put(cfg('t'),cp(2),META,{'messages':2})
    finally:ctx.__exit__(None,None,None)

def test_graph_resume_through_fenced_saver(tmp_path):
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph,START,END
    from langgraph.types import interrupt
    class S(TypedDict):steps:list[str]
    def first(s):return {'steps':s['steps']+['first']}
    def second(s):
        answer=interrupt({'ask':'name?'})
        return {'steps':s['steps']+['second:'+answer]}
    g=StateGraph(S);g.add_node('first',first);g.add_node('second',second)
    g.add_edge(START,'first');g.add_edge('first','second');g.add_edge('second',END)
    ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        graph=g.compile(checkpointer=saver)
        c=cfg('session-x')
        hit=graph.invoke({'steps':[]},c)
        assert '__interrupt__' in hit
        from langgraph.types import Command
        done=graph.invoke(Command(resume='shy'),c)
        assert done['steps']==['first','second:shy']
        latest=saver.get_tuple(c)  # completed thread keeps its fenced history
        assert latest is not None
    finally:ctx.__exit__(None,None,None)

def test_stale_same_id_reput_cannot_mutate_metadata_or_payload(tmp_path):
    ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        c1=saver.put(cfg('t'),cp(1),META,{'messages':1})
        admin.execute("UPDATE engine_sessions SET generation=2 WHERE session_id='s1'")
        forged=dict(cp(1));forged['ts']='2030-01-01T00:00:00+00:00'
        with pytest.raises(StopRequested):
            saver.put(cfg('t'),forged,{'source':'update','step':9,'parents':{'x':'y'}},{'messages':1})
        got=PostgresSaver(admin,serde=JsonPlusSerializer()).get_tuple(c1)
        assert got.checkpoint['ts']=='2026-10-22T08:01:00+00:00'
        assert got.metadata==META
    finally:ctx.__exit__(None,None,None)
