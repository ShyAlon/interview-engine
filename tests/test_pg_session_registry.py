"""SessionRegistry in database persistence mode (PgSessionStore).

Same registry contract as the file-backed mode - claim-before-serve, durable
Stop without the execution lease, restore from the staged publication only -
proven on a throwaway real Postgres (pgserver). No provider calls.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed: PG registry tests skip')

import psycopg
import psycopg.rows
from vmodel_contracts.interactive import StartSessionRequest, InputRequest, InputKind, InputResponse, ProcessStatus

from interview_engine.http_service import SessionRegistry, ProtocolFailure
from interview_engine.pg_session_store import PgSessionStore
from test_pg_session_store import MIRRORED_REGISTRY_DDL

OWNER = 'owner-a'


def make_conn(tmp_path, name='pgdata'):
    server = pgserver.get_server(str(tmp_path / name))
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    conn.execute(MIRRORED_REGISTRY_DDL)
    conn.commit()
    return conn


def make_registry(conn, program=None, instance='test#1', ttl=60.0):
    return SessionRegistry(program or (lambda *a: Program()),
                           session_store=PgSessionStore(conn), owner_id=OWNER,
                           instance_id=instance, lease_ttl=ttl)


class Program:
    finished = False
    def advance(self, response=None):
        return InputRequest(input_id=uuid4(), kind=InputKind.TEXT, prompt='fake')


def start(registry, sid=None):
    return registry.start(StartSessionRequest(session_id=sid, task_id='fake', idea='fake', model_route='local'))


def test_start_respond_persist_round_trip(tmp_path):
    conn = make_conn(tmp_path)
    registry = make_registry(conn)
    session, _ = start(registry)
    answer = InputResponse(input_id=session.open_input.input_id, value='fake', actor_id='fake')
    _, snapshot = registry.respond(session.session_id, answer)
    assert snapshot.open_input is not None
    row = PgSessionStore(conn).load_published(session.session_id, OWNER)
    assert row['control']['task_id'] == 'fake' and row['control']['sequence'] == snapshot.sequence


def test_live_lease_blocks_second_instance_then_takeover_after_expiry(tmp_path):
    conn = make_conn(tmp_path)
    first = make_registry(conn, instance='test#1', ttl=0.2)
    session, _ = start(first)
    second = make_registry(conn, instance='test#2', ttl=60.0)
    with pytest.raises(KeyError):
        second.get(session.session_id)  # live lease: no claim, no serve
    import time; time.sleep(0.25)
    restored = second.get(session.session_id)  # expired: takeover + restore
    assert restored.pg_claim.instance == 'test#2'
    assert restored.task_id == 'fake' and restored.published is not None


def test_takeover_keeps_prior_epoch_control_and_stale_saves_fail_closed(tmp_path):
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    first = make_registry(conn, instance='test#1', ttl=0.2)
    session, _ = start(first)
    stale_claim = session.pg_claim
    import time; time.sleep(0.25)
    second = make_registry(conn, instance='test#2', ttl=60.0)
    restored = second.get(session.session_id)
    assert restored.published.sequence == session.published.sequence
    # The stale instance's writes die at the fence; nothing is silently kept.
    from interview_engine.stop_fence import StopRequested
    with pytest.raises(StopRequested):
        store.fenced_save(stale_claim, {'status': 'stale'})
    assert store.load_published(session.session_id, OWNER)['control']['task_id'] == 'fake'


def test_stop_during_advance_discards_late_result_and_stays_stopped(tmp_path):
    entered = threading.Event(); release = threading.Event()
    class Slow:
        finished = False
        def advance(self, response=None):
            if response is not None: entered.set(); release.wait(2)
            return InputRequest(input_id=uuid4(), kind=InputKind.TEXT, prompt='late')
    conn = make_conn(tmp_path)
    registry = make_registry(conn, program=lambda *a: Slow())
    session, _ = start(registry)
    answer = InputResponse(input_id=session.open_input.input_id, value='fake', actor_id='fake')
    with ThreadPoolExecutor() as pool:
        future = pool.submit(registry.respond, session.session_id, answer)
        assert entered.wait(2)
        state = registry.stop(session.session_id)
        assert state.process_status == ProcessStatus.STOP_REQUESTED and state.open_input is None
        release.set(); _, state = future.result()
    assert state.open_input is None and state.process_status == ProcessStatus.STOP_REQUESTED
    with pytest.raises(ProtocolFailure):
        registry.respond(session.session_id, answer)
    # Durable across a fresh instance: restore shows stopped, never resumes.
    fresh = make_registry(conn, instance='test#2', ttl=60.0)
    import time; time.sleep(0.05)
    restored = fresh.get(session.session_id)
    assert restored.pg_stopped is True and restored.open_input is None


def test_cross_owner_stop_and_read_are_invisible(tmp_path):
    conn = make_conn(tmp_path)
    registry = make_registry(conn)  # owner-a
    session, _ = start(registry)
    foreign = SessionRegistry(lambda *a: Program(), session_store=PgSessionStore(conn),
                              owner_id='owner-b', instance_id='test#9', lease_ttl=60.0)
    with pytest.raises(KeyError):
        foreign.stop(session.session_id)
    with pytest.raises(KeyError):
        foreign.get(session.session_id)
    # The real owner's session is untouched.
    assert registry.snapshot(session.session_id).process_status != ProcessStatus.STOP_REQUESTED


def test_session_id_reused_after_commit_rejected(tmp_path):
    conn = make_conn(tmp_path)
    registry = make_registry(conn)
    sid = str(uuid4())
    start(registry, sid)
    other = make_registry(conn, instance='test#2', ttl=0.0)
    with pytest.raises(ProtocolFailure):
        start(other, sid)  # row exists; no overwrite even after lease expiry
