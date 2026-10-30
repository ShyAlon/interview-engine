"""Live progress is observational, independently fenced, and not step evidence."""
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest
from test_pg_session_store import make_store, MIRRORED_JOURNAL_DDL, OWNER_A, OWNER_B
from interview_engine.pg_session_store import PublicationConflict
from interview_engine.stop_fence import StopRequested


def setup(tmp_path):
    store=make_store(tmp_path)
    with store._borrow() as c:
        c.execute(MIRRORED_JOURNAL_DDL);c.commit()
    store.create('live','task',OWNER_A)
    claim=store.claim('live',OWNER_A,'worker',ttl=60)
    return store,claim


def journal(store,claim):
    from interview_engine.pg_journal import ObservationalJournal
    return ObservationalJournal(store,claim,'live','task')


def append(j):
    return j.append(phase_key='intent',stage='probe',label='Building',status='heartbeat')


def test_concurrent_independent_append_paging_and_restart(tmp_path):
    store,claim=setup(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as p:
        signals=list(p.map(lambda _:append(journal(store,claim)),range(20)))
    assert sorted(s.cursor for s in signals)==list(range(1,21))
    assert len({s.event_id for s in signals})==20
    rows=store.read_journal('live',OWNER_A,10,3)
    assert [r['cursor'] for r in rows]==[11,12,13]
    assert all(r['page']['journal_mode']=='observational' for r in rows)
    assert append(journal(store,claim)).cursor==21
    assert store.read_journal('live',OWNER_B,0,100)==[]
    assert store.read_journal('other',OWNER_A,0,100)==[]
    assert store.load_published('live',OWNER_A)['control'] is None


def test_invalid_signal_and_fence_fail_closed(tmp_path):
    from dataclasses import replace
    store,claim=setup(tmp_path)
    with pytest.raises(ValueError):journal(store,claim).append(phase_key='intent',stage='probe',label='x',status='invalid')
    assert store.journal_head('live',OWNER_A)==0
    for bad in [replace(claim,owner_id=OWNER_B),replace(claim,generation=99),replace(claim,epoch=99)]:
        with pytest.raises(StopRequested):append(journal(store,bad))
    store.stop('live',OWNER_A)
    with pytest.raises(StopRequested):append(journal(store,claim))
    assert store.journal_head('live',OWNER_A)==0


def test_journal_modes_cannot_mix_in_either_order(tmp_path):
    store,claim=setup(tmp_path)
    append(journal(store,claim))
    with pytest.raises(PublicationConflict):store.publish_step(claim,{'sequence':1},journal_pages=[(2,{'legacy':'buffered'})])
    # Control-only commits remain valid and do not erase live heartbeats.
    store.publish_step(claim,{'sequence':1})
    assert store.journal_head('live',OWNER_A)==1
    store.create('buffered','task',OWNER_A)
    other=store.claim('buffered',OWNER_A,'worker',ttl=60)
    store.publish_step(other,{'sequence':1},journal_pages=[(1,{'legacy':'buffered'})])
    from interview_engine.pg_journal import ObservationalJournal
    with pytest.raises(PublicationConflict):append(ObservationalJournal(store,other,'buffered','task'))


def test_live_progress_readable_during_blocked_advance_without_control_lock(tmp_path):
    import threading
    from interview_engine.http_service import SessionRegistry
    from interview_engine.pg_journal import ObservationalJournal
    from vmodel_contracts.interactive import StartSessionRequest,InputRequest,InputKind,InputResponse
    store,_=setup(tmp_path)
    entered,release=threading.Event(),threading.Event()
    class Program:
        finished=False
        def advance(self,response=None):
            if response is None:return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Who?')
            append(self.progress_journal)
            entered.set()
            assert release.wait(5)
            self.finished=True
    registry=SessionRegistry(lambda *_:Program(),session_store=store,owner_id=OWNER_A,progress_mode='observational')
    try:
        session,_=registry.start(StartSessionRequest(session_id=uuid4(),task_id='task',idea='idea',model_route='m'))
        session.program.progress_journal=ObservationalJournal(store,session.pg_claim,session.session_id,'task')
        snapshot=registry.snapshot(session.session_id)
        response=InputResponse(input_id=snapshot.open_input.input_id,expected_sequence=snapshot.sequence,value='answer',actor_id='tester')
        with ThreadPoolExecutor(max_workers=2) as pool:
            work=pool.submit(registry.respond,session.session_id,response)
            try:
                assert entered.wait(2)
                page=pool.submit(registry.progress,session.session_id).result(timeout=.75)
                assert page.events[0].status=='heartbeat' and page.next_cursor==1
                assert store.load_published(session.session_id,OWNER_A)['control']['open_input']['input_id'] == str(snapshot.open_input.input_id)
                assert registry.snapshot(session.session_id).open_input is None
                assert not work.done(), 'live progress must not wait for step completion'
            finally:release.set()
            work.result(timeout=2)
    finally:registry.close()


def test_expired_lease_and_takeover_reject_live_append(tmp_path):
    store,claim=setup(tmp_path)
    with store._borrow() as c:
        c.execute("UPDATE engine_sessions SET lease_expires_at=0 WHERE session_id='live'");c.commit()
    with pytest.raises(StopRequested):append(journal(store,claim))
    new=store.claim('live',OWNER_A,'new-worker',ttl=60)
    with pytest.raises(StopRequested):append(journal(store,claim))
    assert append(journal(store,new)).cursor==1


def test_signal_heavy_benchmark_records_query_and_checkout_cost(tmp_path):
    import json,time
    from contextlib import contextmanager
    store,claim=setup(tmp_path)
    original=store._borrow
    costs={'checkouts':0,'queries':0,'guard_roundtrip_ms':[],'append_ms':[],'lock_held_ms':[]}
    class Probe:
        def __init__(self,conn):self.conn=conn;self.lock_at=None
        def commit(self):
            try:return self.conn.commit()
            finally:
                if self.lock_at is not None:costs['lock_held_ms'].append((time.perf_counter()-self.lock_at)*1000)
        def execute(self,query,*args,**kwargs):
            start=time.perf_counter()
            try:return self.conn.execute(query,*args,**kwargs)
            finally:
                costs['queries']+=1
                if 'FOR UPDATE' in query:
                    costs['guard_roundtrip_ms'].append((time.perf_counter()-start)*1000)
                    self.lock_at=time.perf_counter()
        def __getattr__(self,k):return getattr(self.conn,k)
    @contextmanager
    def borrow():
        costs['checkouts']+=1
        with original() as conn:yield Probe(conn)
    store._borrow=borrow
    j=journal(store,claim)
    for _ in range(200):
        start=time.perf_counter();append(j);costs['append_ms'].append((time.perf_counter()-start)*1000)
    assert costs['checkouts']==200 and costs['queries']==600
    def summary(values):
        values=sorted(values)
        return {'mean':sum(values)/len(values),'p95':values[int(len(values)*.95)-1],'max':max(values)}
    metrics={'signals':200,'checkouts':costs['checkouts'],'data_queries':costs['queries'],
        'guard_roundtrip_ms_includes_lock_wait':summary(costs['guard_roundtrip_ms']),
        'lock_hold_ms_after_guard_until_commit_return':summary(costs['lock_held_ms']),
        'append_ms':summary(costs['append_ms'])}
    print('LIVE_PROGRESS_BENCHMARK '+json.dumps(metrics))


def test_actual_application_observational_mode_is_explicit(tmp_path,monkeypatch):
    from test_pg_application_wiring import make_conn,FakeModel,PgSessionStore,OWNER
    from interview_engine.application import InterviewApplication
    from interview_engine.pg_journal import ObservationalJournal
    from conftest import managed_pool
    from interview_engine import model_config
    monkeypatch.setattr('httpx.get',lambda *a,**k:type('Response',(),{'raise_for_status':lambda self:None,'json':lambda self:{'data':[{'id':'phi4'}]}})())
    monkeypatch.setattr(model_config,'build_chat_model',lambda *a,**k:FakeModel([]))
    store=PgSessionStore(managed_pool(make_conn(tmp_path).info.dsn))
    store.create('live','task',OWNER)
    claim=store.claim('live',OWNER,'worker',ttl=60)
    app=InterviewApplication('live','task','idea',model_route='ollama',initial_model='phi4',
        artifact_root=str(tmp_path/'artifacts'),pg_store=store,pg_claim=claim,progress_mode='observational')
    try:
        assert isinstance(app.progress_journal,ObservationalJournal)
        append(app.progress_journal)
        assert len(store.read_journal('live',OWNER,0,100))==1
    finally:
        if hasattr(app,'close'):app.close()
