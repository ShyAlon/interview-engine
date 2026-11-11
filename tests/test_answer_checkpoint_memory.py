from copy import deepcopy

def test_memory_checkpoint_reuses_cold_read_and_own_writes(monkeypatch):
    from interview_engine.pg_checkpoint import FencedPostgresSaver,answer_checkpoint_memory
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.base import CheckpointTuple
    calls=[]
    config={'configurable':{'thread_id':'s','checkpoint_ns':'','checkpoint_id':'old'}}
    row=CheckpointTuple(config,{'id':'old','channel_values':{'a':[1]}},{'step':1},None,[])
    def cold(self,cfg):calls.append(cfg);return deepcopy(row)
    monkeypatch.setattr(PostgresSaver,'get_tuple',cold)
    saver=FencedPostgresSaver.__new__(FencedPostgresSaver)
    with answer_checkpoint_memory():
        assert saver.get_tuple(config).checkpoint['id']=='old'
        got=saver.get_tuple(config);got.checkpoint['channel_values']['a'].append(2)
        assert saver.get_tuple(config).checkpoint['channel_values']['a']==[1]
        assert len(calls)==1
        saver.remember_answer_checkpoint(config,{'id':'new','channel_values':{'a':[3]}},{'step':2})
        assert saver.get_tuple({'configurable':{'thread_id':'s','checkpoint_ns':''}}).checkpoint['id']=='new'
        assert len(calls)==1
    saver.get_tuple(config)
    assert len(calls)==2


def test_memory_matches_real_database_and_invalidates_pending_writes(tmp_path):
    from test_pg_checkpoint import make_saver,server,cfg,cp,META
    from interview_engine.pg_checkpoint import answer_checkpoint_memory
    from langgraph.checkpoint.postgres import PostgresSaver
    with answer_checkpoint_memory():ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        written=saver.put(cfg('t'),cp(1),META,{'messages':'1'})
        local=saver.get_tuple(written)
        durable=PostgresSaver.get_tuple(saver,written)
        assert local==durable
        saver.put_writes(written,[('messages','new pending')],'taskA')
        assert saver.get_tuple(written)==PostgresSaver.get_tuple(saver,written)
        assert saver.get_tuple(written).pending_writes==[('taskA','messages','new pending')]
    finally:ctx.__exit__(None,None,None)


def test_cold_latest_pending_writes_failure_and_new_saver(tmp_path):
    import pytest
    from test_pg_checkpoint import make_saver,server,cfg,cp,META
    from interview_engine.pg_checkpoint import FencedPostgresSaver,answer_checkpoint_memory,LeaseFence
    from interview_engine.stop_fence import StopRequested
    from langgraph.checkpoint.postgres import PostgresSaver
    with answer_checkpoint_memory():ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        one=saver.put(cfg('t'),cp(1),META,{'messages':'1'})
        # A fresh saver must cold-read, including the latest (unpinned) form.
        with answer_checkpoint_memory():
            cold=FencedPostgresSaver(saver.conn,lease=LeaseFence('s1',1,1),serde=saver.serde)
        assert cold.get_tuple(cfg('t'))==PostgresSaver.get_tuple(cold,cfg('t'))
        cold.put_writes(one,[('messages','pending')],'A')
        assert cold.get_tuple(cfg('t'))==PostgresSaver.get_tuple(cold,cfg('t'))
        assert cold.get_tuple(cfg('t')).pending_writes==[('A','messages','pending')]
        two=cold.put(one,cp(2),META,{'messages':'2'})
        assert cold.get_tuple(cfg('t'))==PostgresSaver.get_tuple(cold,cfg('t'))
        assert cold.get_tuple(one).checkpoint['id']==cp(1)['id']
        admin.execute("UPDATE engine_sessions SET generation=2 WHERE session_id='s1'")
        with pytest.raises(StopRequested):cold.put(two,cp(3),META,{'messages':'3'})
        assert cold.get_tuple(cfg('t')).checkpoint['id']==cp(2)['id']
        # No scope on construction: legacy saver always goes to durable state.
        fresh=FencedPostgresSaver(saver.conn,lease=LeaseFence('s1',2,1),serde=saver.serde)
        assert fresh.get_tuple(cfg('t'))==PostgresSaver.get_tuple(fresh,cfg('t'))
        assert not hasattr(fresh,'_answer_memory')
    finally:ctx.__exit__(None,None,None)


def test_delta_snapshot_memory_matches_installed_saver(tmp_path):
    from test_pg_checkpoint import make_saver,server,cfg,cp,META
    from interview_engine.pg_checkpoint import answer_checkpoint_memory
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.serde.types import _DeltaSnapshot
    with answer_checkpoint_memory():ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        checkpoint=cp(1);checkpoint['channel_values']['messages']=_DeltaSnapshot(['m1'])
        written=saver.put(cfg('t'),checkpoint,META,{'messages':'1'})
        assert saver.get_tuple(written)==PostgresSaver.get_tuple(saver,written)
    finally:ctx.__exit__(None,None,None)


def test_immutable_blob_collision_cannot_poison_answer_memory(tmp_path):
    from test_pg_checkpoint import make_saver,server,cfg,cp,META
    from interview_engine.pg_checkpoint import answer_checkpoint_memory
    from langgraph.checkpoint.postgres import PostgresSaver
    with answer_checkpoint_memory():ctx,saver,admin=make_saver(server(tmp_path).get_uri())
    try:
        first=saver.put(cfg('t'),cp(1),META,{'messages':'1'})
        conflict=cp(2);conflict['channel_versions']['messages']='1'
        conflict['channel_values']['messages']=['not the stored version']
        second=saver.put(first,conflict,META,{'messages':'1'})
        assert saver.get_tuple(second)==PostgresSaver.get_tuple(saver,second)
        assert saver.get_tuple(second).checkpoint['channel_values']['messages']==['m1']
    finally:ctx.__exit__(None,None,None)


def test_answer_local_latest_uses_published_anchor_not_unpublished_orphan(tmp_path):
    from test_pg_checkpoint import make_saver,server,cfg,cp,META
    from interview_engine.pg_checkpoint import FencedPostgresSaver,LeaseFence,answer_checkpoint_memory
    from langgraph.checkpoint.postgres import PostgresSaver
    ctx,writer,admin=make_saver(server(tmp_path).get_uri())
    try:
        published=writer.put(cfg('t'),cp(1),META,{'messages':'1'})
        writer.put(published,cp(2),META,{'messages':'2'}) # An unpublished crash orphan.
        with answer_checkpoint_memory():
            reader=FencedPostgresSaver(writer.conn,lease=LeaseFence('s1',1,1),serde=writer.serde)
        reader.use_published_answer_checkpoint(published)
        assert reader.get_tuple(published).checkpoint['id']==cp(1)['id']
        assert reader.get_tuple(cfg('t')).checkpoint['id']==cp(1)['id']
        assert PostgresSaver.get_tuple(reader,cfg('t')).checkpoint['id']==cp(2)['id']
    finally:ctx.__exit__(None,None,None)
