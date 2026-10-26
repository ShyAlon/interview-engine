"""Journal pages in the database: fenced append, immutable cursors, tenant reads."""
import time

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed: PG journal tests skip')
from conftest import managed_pg_server, managed_pool

import psycopg
import psycopg.rows
from vmodel_contracts.progress_signal import ProgressSignal

from interview_engine.pg_session_store import PgSessionStore, PublicationConflict
from interview_engine.http_service import SessionRegistry
from interview_engine.stop_fence import StopRequested
from test_pg_session_store import MIRRORED_REGISTRY_DDL, MIRRORED_JOURNAL_DDL

OWNER = '00000000-0000-0000-0000-000000000001'


def make_store(tmp_path):
    server = managed_pg_server(str(tmp_path / 'pgdata'))
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    conn.execute(MIRRORED_REGISTRY_DDL)
    conn.execute(MIRRORED_JOURNAL_DDL)
    conn.commit()
    return PgSessionStore(managed_pool(conn.info.dsn))


def signal(session, cursor, label):
    return ProgressSignal(event_id=f'e{cursor}', cursor=cursor, task_id='task1', session_id=session,
                          phase_key='interview', stage='s', label=label, status='started').model_dump(mode='json')


def test_fenced_append_and_paged_read(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=60)
    store.append_journal(claim, 1, signal('s1', 1, 'first'))
    store.append_journal(claim, 2, signal('s1', 2, 'second'))
    page = store.read_journal('s1', OWNER, after=0, limit=10)
    assert [r['cursor'] for r in page] == [1, 2]
    assert [r['cursor'] for r in store.read_journal('s1', OWNER, after=1, limit=10)] == [2]
    assert [r['cursor'] for r in store.read_journal('s1', OWNER, after=0, limit=1)] == [1]


def test_duplicate_cursor_is_kept_not_overwritten(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=60)
    store.append_journal(claim, 1, signal('s1', 1, 'original'))
    forged = signal('s1', 1, 'forged')
    store.append_journal(claim, 1, forged)
    assert store.read_journal('s1', OWNER, after=0, limit=10)[0]['page']['label'] == 'original'


def test_stale_claim_append_rejected_and_nothing_persisted(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    stale = store.claim('s1', OWNER, 'host#1', ttl=0.05)
    time.sleep(0.1)
    live = store.claim('s1', OWNER, 'host#2', ttl=60)
    store.append_journal(live, 1, signal('s1', 1, 'live'))
    with pytest.raises(StopRequested):
        store.append_journal(stale, 2, signal('s1', 2, 'stale'))
    assert [r['cursor'] for r in store.read_journal('s1', OWNER, after=0, limit=10)] == [1]


def test_cross_owner_journal_read_is_invisible(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=60)
    store.append_journal(claim, 1, signal('s1', 1, 'private'))
    assert store.read_journal('s1', '00000000-0000-0000-0000-000000000002', after=0, limit=10) == []


def test_registry_progress_reads_journal_in_db_mode(tmp_path, request):
    from uuid import uuid4
    from vmodel_contracts.interactive import StartSessionRequest, InputRequest, InputKind
    class Program:
        finished = False
        def advance(self, response=None):
            return InputRequest(input_id=uuid4(), kind=InputKind.TEXT, prompt='fake')
    store = make_store(tmp_path)
    registry = SessionRegistry(lambda *a: Program(), session_store=store, owner_id=OWNER,
                               instance_id='test#1', lease_ttl=60.0)
    request.addfinalizer(registry.close)
    session, _ = registry.start(StartSessionRequest(task_id='fake', idea='fake', model_route='local'))
    store.append_journal(session.pg_claim, 1, signal(session.session_id, 1, 'step'))
    page = registry.progress(session.session_id, after=0, limit=10)
    assert [e.label for e in page.events] == ['step'] and page.next_cursor == 1


def test_publish_step_commits_journal_control_and_pointer_atomically(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=60)
    store.publish_step(claim, {'status': 'step-1', 'sequence': 1}, journal_cursor=1, journal_page=signal('s1', 1, 'one'),
                       published_thread_id='s1', published_checkpoint_id='cp-1')
    published = store.load_published('s1', OWNER)
    assert published['control'] == {'status': 'step-1', 'sequence': 1} and published['published_checkpoint_id'] == 'cp-1'
    assert [r['cursor'] for r in store.read_journal('s1', OWNER, 0, 10)] == [1]


def test_publish_step_fence_failure_leaves_nothing_behind(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    stale = store.claim('s1', OWNER, 'host#1', ttl=0.05)
    time.sleep(0.1)
    store.claim('s1', OWNER, 'host#2', ttl=60)
    with pytest.raises(StopRequested):
        store.publish_step(stale, {'status': 'stale', 'sequence': 1}, journal_cursor=1, journal_page=signal('s1', 1, 'x'),
                           published_thread_id='s1', published_checkpoint_id='cp-x')
    published = store.load_published('s1', OWNER)
    assert published['control'] is None and published['published_checkpoint_id'] is None
    assert store.read_journal('s1', OWNER, 0, 10) == []


def test_publish_step_exact_replay_is_a_noop(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=60)
    control = {'status': 'step-1', 'sequence': 1}
    page = signal('s1', 1, 'original')
    store.publish_step(claim, control, journal_cursor=1, journal_page=page,
                       published_thread_id='s1', published_checkpoint_id='cp-1')
    store.publish_step(claim, dict(control), journal_cursor=1, journal_page=dict(page),
                       published_thread_id='s1', published_checkpoint_id='cp-1')  # exact replay, same bytes
    published = store.load_published('s1', OWNER)
    assert published['control'] == control and published['published_checkpoint_id'] == 'cp-1'
    assert len(store.read_journal('s1', OWNER, 0, 10)) == 1


def test_conflicting_replay_same_cursor_different_page_rejected(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=60)
    store.publish_step(claim, {'status': 'step-1', 'sequence': 1}, journal_cursor=1,
                       journal_page=signal('s1', 1, 'original'),
                       published_thread_id='s1', published_checkpoint_id='cp-1')
    with pytest.raises(PublicationConflict):
        store.publish_step(claim, {'status': 'changed', 'sequence': 1}, journal_cursor=1,
                           journal_page=signal('s1', 1, 'forged'),
                           published_thread_id='s1', published_checkpoint_id='cp-2')
    published = store.load_published('s1', OWNER)   # nothing advanced
    assert published['control'] == {'status': 'step-1', 'sequence': 1}
    assert published['published_checkpoint_id'] == 'cp-1'
    assert store.read_journal('s1', OWNER, 0, 10)[0]['page']['label'] == 'original'


def test_delayed_old_retry_after_newer_step_rejected(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=60)
    store.publish_step(claim, {'status': 'step-1', 'sequence': 1}, journal_cursor=1,
                       journal_page=signal('s1', 1, 'one'),
                       published_thread_id='s1', published_checkpoint_id='cp-1')
    store.publish_step(claim, {'status': 'step-2', 'sequence': 2}, journal_cursor=2,
                       journal_page=signal('s1', 2, 'two'),
                       published_thread_id='s1', published_checkpoint_id='cp-2')
    with pytest.raises(PublicationConflict):
        store.publish_step(claim, {'status': 'step-1', 'sequence': 1}, journal_cursor=3,
                           journal_page=signal('s1', 3, 'late-old'),
                           published_thread_id='s1', published_checkpoint_id='cp-1')
    published = store.load_published('s1', OWNER)   # state did not regress
    assert published['control'] == {'status': 'step-2', 'sequence': 2}
    assert published['published_checkpoint_id'] == 'cp-2'
    assert [r['cursor'] for r in store.read_journal('s1', OWNER, 0, 10)] == [1, 2]


def test_newer_step_reusing_old_cursor_rejected(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=60)
    store.publish_step(claim, {'status': 'step-1', 'sequence': 1}, journal_cursor=1,
                       journal_page=signal('s1', 1, 'one'),
                       published_thread_id='s1', published_checkpoint_id='cp-1')
    with pytest.raises(PublicationConflict):
        store.publish_step(claim, {'status': 'step-2', 'sequence': 2}, journal_cursor=1,
                           journal_page=signal('s1', 1, 'different'),
                           published_thread_id='s1', published_checkpoint_id='cp-2')
    assert store.load_published('s1', OWNER)['control'] == {'status': 'step-1', 'sequence': 1}


def test_mid_transaction_failure_leaves_no_partial_publication(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=60)
    def boom():
        raise RuntimeError('injected after journal insert, before pointer commit')
    with pytest.raises(RuntimeError):
        store.publish_step(claim, {'status': 'step-1', 'sequence': 1}, journal_cursor=1,
                           journal_page=signal('s1', 1, 'one'),
                           published_thread_id='s1', published_checkpoint_id='cp-1',
                           fault_hook=boom)
    # no explicit rollback: the pool borrow guard rolls back anything a failed call left open
    published = store.load_published('s1', OWNER)
    assert published['control'] is None and published['published_checkpoint_id'] is None
    assert store.read_journal('s1', OWNER, 0, 10) == []
