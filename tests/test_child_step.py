"""Typed claim/restore seam on disposable Postgres; fake programs only."""
from contextlib import contextmanager
import hashlib
import json
from uuid import uuid4

import pytest
from requi_child_contract import decode_operation
from interview_engine.child_step import restored_child_step, published_digest
from interview_engine.http_service import SessionRegistry,default_program_factory
from interview_engine.pg_session_store import PgSessionStore
from test_pg_session_registry import make_conn,make_registry,make_pool,start,OWNER,_cleanup_pg


def operation(sid,row,*,epoch=1,kind='interview.advance'):
    body={'kind':kind,'checkpoint_sha256':published_digest(row),'expected_sequence':row['control']['sequence']}
    if kind=='interview.answer':body.update(wait_id=row['control']['open_input']['input_id'],review_revision='engine:'+sid+':2',answer_ref='answer-1')
    return decode_operation(json.dumps({'version':1,'operation_id':'op-1','binding':{
        'owner_id':OWNER,'auth_session_id':'00000000-0000-0000-0000-000000000002',
        'task_id':'fake','session_id':sid,'job_id':'job-1','generation':1,'lease_epoch':epoch,
        'provider':'ollama','model':'fake'},'operation':body}).encode()+b'\n')


def setup(tmp_path):
    conn=make_conn(tmp_path);first=make_registry(conn);session,_=start(first,uuid4())
    store=PgSessionStore(make_pool(conn._pg_server),claims=None)
    row=store.load_published(session.session_id,OWNER)
    first.close()
    # Mirror production per-transaction Auth claims, not the execution session.
    store.claims=(OWNER,'00000000-0000-0000-0000-000000000002')
    return conn,store,session.session_id,row


class Program:
    finished=False
    def advance(self,response=None):raise AssertionError('restore must not advance')


def test_restores_only_published_checkpoint_and_releases_lease(tmp_path):
    conn,store,sid,row=setup(tmp_path);op=operation(sid,row)
    with restored_child_step(op,trusted_binding=op.binding,store=store,program_factory=lambda *a:Program()) as step:
        assert step.session.sequence==row['control']['sequence']
        assert step.session.pg_claim.epoch==2
        assert step.session.task_id=='fake'
    lease=conn.execute('SELECT placement_instance,lease_expires_at FROM engine_sessions WHERE session_id=%s',(sid,)).fetchone()
    assert lease['lease_expires_at']<=__import__('time').time()


@pytest.mark.parametrize('change', ['epoch','generation','digest','sequence','auth','task','stop','live-lease','binding'])
def test_stale_or_foreign_authority_never_constructs_program(tmp_path,change):
    conn,store,sid,row=setup(tmp_path);op=operation(sid,row);expected=op.binding;calls=[]
    value=op.model_dump(mode='json')
    if change=='epoch':value['binding']['lease_epoch']=99
    if change=='generation':
        value['binding']['generation']=2;value['binding']['execution_anchor_sha256']='d'*64
    if change=='digest':value['operation']['checkpoint_sha256']='f'*64
    if change=='sequence':value['operation']['expected_sequence']=99
    if change=='auth':store.claims=(OWNER,'00000000-0000-0000-0000-000000000099')
    if change=='task':value['binding']['task_id']='other'
    if change=='stop':store.stop(sid,OWNER)
    if change=='live-lease':assert store.claim(sid,OWNER,'other-live',60) is not None
    if change=='binding':expected=expected.model_copy(update={'job_id':'other'})
    op=decode_operation(json.dumps(value).encode()+b'\n')
    if change!='binding':expected=op.binding
    with pytest.raises((ValueError,PermissionError,KeyError)):
        with restored_child_step(op,trusted_binding=expected,store=store,program_factory=lambda *a:calls.append(1)):pass
    assert calls==[]


def test_production_factory_refused_before_database_or_program(tmp_path):
    conn,store,sid,row=setup(tmp_path);op=operation(sid,row)
    with pytest.raises(PermissionError,match='money boundary'):
        with restored_child_step(op,trusted_binding=op.binding,store=store,program_factory=default_program_factory):pass


def test_exception_during_restore_releases_claim(tmp_path):
    conn,store,sid,row=setup(tmp_path);op=operation(sid,row)
    def fail(*a):raise RuntimeError('fake restore failed')
    with pytest.raises(RuntimeError,match='fake restore failed'):
        with restored_child_step(op,trusted_binding=op.binding,store=store,program_factory=fail):pass
    assert store.claim(sid,OWNER,'after-error',60) is not None


def test_real_process_restores_and_releases_published_step(tmp_path):
    import asyncio,os,sys
    from pathlib import Path
    delivery=Path(__file__).parents[2]/'delivery-api/src'
    if not delivery.is_dir():pytest.skip('explicit sibling delivery source required')
    sys.path.insert(0,str(delivery))
    from delivery_api.child_supervisor import ChildSupervisor
    conn,store,sid,row=setup(tmp_path);op=operation(sid,row)
    events=[]
    # The supervisor intentionally strips PYTHONPATH. Use an explicit local source
    # bootstrap for this uninstalled development checkout, never loosen its env.
    bootstrap="import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    supervisor=ChildSupervisor([sys.executable,'-c',bootstrap,str(Path(__file__).parents[1]/'src'),str(Path(__file__).with_name('fixtures_child_restore.py')),
        conn._pg_server.get_uri(),json.dumps(op.binding.model_dump(mode='json'))])
    result=asyncio.run(supervisor.run(op,authorize=lambda op:None,on_event=events.append,timeout_seconds=5))
    assert result.reaped and result.returncode==0
    assert [e.state for e in events]==['ready','running','completed']
    assert store.claim(sid,OWNER,'next-child',60) is not None


def test_stale_answer_review_or_wait_never_constructs(tmp_path):
    conn,store,sid,row=setup(tmp_path);op=operation(sid,row,kind='interview.answer')
    for field,value in [('wait_id',str(uuid4())),('review_revision','stale')]:
        data=op.model_dump(mode='json');data['operation'][field]=value
        bad=decode_operation(json.dumps(data).encode()+b'\n');calls=[]
        with pytest.raises(PermissionError):
            with restored_child_step(bad,trusted_binding=bad.binding,store=store,program_factory=lambda *a:calls.append(1)):pass
        assert calls==[]


def test_parent_descriptor_loss_stops_action_and_releases_claim(tmp_path):
    import socket,threading
    from interview_engine.child_channel import serve_restored,ParentGone
    from requi_child_contract import encode_operation
    conn,store,sid,row=setup(tmp_path);op=operation(sid,row)
    parent,child=socket.socketpair();failures=[];action_started=threading.Event()
    def action(step,stopped):
        action_started.set()
        assert __import__('time').monotonic()>0
        for _ in range(100):
            if stopped():return
            __import__('time').sleep(.01)
        raise AssertionError('parent descriptor loss was ignored')
    def worker():
        try:serve_restored(child,trusted_binding=op.binding,store=store,program_factory=lambda *a:Program(),action=action)
        except BaseException as exc:failures.append(exc)
    thread=threading.Thread(target=worker);thread.start()
    parent.sendall(encode_operation(op))
    assert action_started.wait(3)
    parent.close();thread.join(3)
    assert not thread.is_alive()
    assert len(failures)==1 and isinstance(failures[0],ParentGone)
    assert store.claim(sid,OWNER,'after-parent-loss',60) is not None


def test_same_schema_wheel_and_fixture_as_delivery():
    from pathlib import Path
    import zipfile,requi_child_contract
    root=Path(__file__).parents[1]
    wheel=root/'vendor/requi_child_contract-0.4.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as z:assert z.read('requi_child_contract/__init__.py')==Path(requi_child_contract.__file__).read_bytes()
    delivery=root.parent/'delivery-api'
    if not delivery.is_dir():pytest.skip('explicit sibling delivery source required')
    assert wheel.read_bytes()==(delivery/'vendor'/wheel.name).read_bytes()
    fixture=decode_operation((delivery/'tests/fixtures/child-execution-v1-answer.json').read_bytes())
    assert fixture.operation.kind=='interview.answer'


def test_compare_and_claim_is_atomic_between_two_children(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    conn,store,sid,row=setup(tmp_path);op=operation(sid,row)
    barrier=threading.Barrier(2)
    def claim(instance):
        barrier.wait()
        try:return store.claim_published_expected(op.binding,op.operation,instance,60)[0]
        except PermissionError:return None
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(claim,['child-a','child-b']))
    assert sum(result is not None for result in results)==1
    assert next(result for result in results if result is not None).epoch==2


def test_corrupt_published_payload_releases_before_any_graph(tmp_path):
    conn,store,sid,row=setup(tmp_path)
    broken=dict(row['control']);broken['request']['session_id']=str(uuid4())
    conn.execute('UPDATE engine_sessions SET control=%s::jsonb WHERE session_id=%s',(json.dumps(broken),sid));conn.commit()
    row=store.load_published(sid,OWNER);op=operation(sid,row)
    with pytest.raises(KeyError,match='persisted session binding'):
        with restored_child_step(op,trusted_binding=op.binding,store=store,program_factory=lambda *a:Program()):pass
    assert store.claim(sid,OWNER,'after-corrupt',60) is not None
