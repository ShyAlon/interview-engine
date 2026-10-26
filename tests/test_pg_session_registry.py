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
from conftest import managed_pg_server

import psycopg
import psycopg.rows
from vmodel_contracts.interactive import StartSessionRequest, InputRequest, InputKind, InputResponse, ProcessStatus

from interview_engine.http_service import SessionRegistry, ProtocolFailure
from interview_engine.pg_session_store import PgSessionStore
from test_pg_session_store import MIRRORED_REGISTRY_DDL

OWNER = '00000000-0000-0000-0000-000000000001'


# Every connection and registry a test opens is closed during teardown, BEFORE
# the throwaway server goes away: no heartbeat thread or socket outlives its
# Postgres.
_OPEN = {'servers': [], 'conns': [], 'registries': []}


@pytest.fixture(autouse=True)
def _cleanup_pg():
    yield
    for r in _OPEN['registries']:
        r.close()
    for c in _OPEN['conns']:
        try: c.close()
        except Exception: pass
    _OPEN['registries'].clear(); _OPEN['conns'].clear(); _OPEN['servers'].clear()


def make_conn(tmp_path, name='pgdata'):
    server = managed_pg_server(str(tmp_path / name))
    _OPEN['servers'].append(server)
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    conn.execute(MIRRORED_REGISTRY_DDL)
    conn.commit()
    _OPEN['conns'].append(conn)
    return conn


def make_registry(conn, program=None, instance='test#1', ttl=60.0):
    registry = SessionRegistry(program or (lambda *a: Program()),
                               session_store=PgSessionStore(conn), owner_id=OWNER,
                               instance_id=instance, lease_ttl=ttl)
    _OPEN['registries'].append(registry)
    return registry


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
    first.close()  # simulated instance death: heartbeats halt, lease can expire
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
    first.close()  # simulated instance death before takeover
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
                              owner_id='00000000-0000-0000-0000-000000000002', instance_id='test#9', lease_ttl=60.0)
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


def test_heartbeat_holds_lease_through_long_advance(tmp_path):
    import time
    entered = threading.Event(); release = threading.Event()
    class Slow:
        finished = False
        def advance(self, response=None):
            if response is not None: entered.set(); release.wait(3)
            return InputRequest(input_id=uuid4(), kind=InputKind.TEXT, prompt='late')
    conn = make_conn(tmp_path)
    first = make_registry(conn, program=lambda *a: Slow(), instance='test#1', ttl=0.4)
    session, _ = start(first)
    answer = InputResponse(input_id=session.open_input.input_id, value='fake', actor_id='fake')
    with ThreadPoolExecutor() as pool:
        future = pool.submit(first.respond, session.session_id, answer)
        assert entered.wait(2)
        time.sleep(0.9)  # over 2x the lease ttl; only heartbeats keep it alive
        second = make_registry(conn, instance='test#2', ttl=60.0)
        with pytest.raises(KeyError):
            second.get(session.session_id)  # still served by the first instance
        release.set(); future.result(timeout=5)


def test_lost_heartbeat_marks_session_stopped_and_fails_closed(tmp_path):
    import time
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    first = make_registry(conn, instance='test#1', ttl=0.3)
    session, _ = start(first)
    store.stop(session.session_id, OWNER)  # owner Stop lands from outside
    deadline = time.time() + 5
    while not session.pg_stopped and time.time() < deadline:
        time.sleep(0.05)
    assert session.pg_stopped is True  # heartbeat observed the loss
    # Later fenced writes from this instance are rejected; the session stays stopped.
    assert store.heartbeat(session.pg_claim, ttl=60) is False
    fresh = make_registry(conn, instance='test#2', ttl=60.0)
    restored = fresh.get(session.session_id)
    assert restored.pg_stopped is True


def test_close_halts_and_joins_heartbeats_before_takeover(tmp_path):
    import time
    conn = make_conn(tmp_path)
    first = make_registry(conn, instance='test#1', ttl=0.3)
    session, _ = start(first)
    first.close()
    thread = session.pg_heartbeat_thread
    assert thread is not None and not thread.is_alive()  # bounded join, no straggler
    time.sleep(0.35)  # lease lapses only after heartbeats actually stopped
    second = make_registry(conn, instance='test#2', ttl=60.0)
    assert second.get(session.session_id).pg_claim.instance == 'test#2'


def test_snapshot_read_never_claims_constructs_or_publishes(tmp_path):
    # The 2026-10-26 pool exhaustion: viewing a parked task resumed its session
    # through the READ path, which claimed the lease, started a heartbeat and
    # constructed the full application - pinning ~4 pool connections and the
    # lease forever (a live heartbeat defeats idle eviction). A read must only
    # read the staged publication.
    conn = make_conn(tmp_path)
    writer = make_registry(conn)
    session, _ = start(writer)
    sid = session.session_id
    writer.close()
    _OPEN['registries'].remove(writer)
    reader = make_registry(conn, instance='test#reader')
    claimed = []
    real_claim = reader._store.claim
    def spy_claim(*a, **k):
        claimed.append(a); return real_claim(*a, **k)
    reader._store.claim = spy_claim
    snap = reader.snapshot(sid)
    assert snap.session_id == sid
    assert claimed == [], 'snapshot is a read: it must never claim the lease'
    cached = reader._sessions.get(sid)
    assert cached is None or cached.pg_claim is None, \
        'a read-restored session must not sit in the serve cache (a later serve must claim first)'


def test_saver_write_conn_opens_lazily(tmp_path, monkeypatch):
    # The saver's dedicated guarded-write connection must open on the first
    # fenced write, not at construction: an idle session should not hold it.
    from interview_engine import pg_checkpoint
    from interview_engine.pg_checkpoint import FencedPostgresSaver, LeaseFence
    conn = make_conn(tmp_path, name='pgdata-lazy')
    opened = []
    real_connect = psycopg.Connection.connect
    def counting_connect(*a, **k):
        opened.append(a); return real_connect(*a, **k)
    monkeypatch.setattr(pg_checkpoint.psycopg.Connection, 'connect', staticmethod(counting_connect))
    saver = FencedPostgresSaver(conn, lease=LeaseFence(str(uuid4()), 1, 1), claims=None)
    try:
        assert opened == [], 'guarded-write connection opened at construction; it must be lazy'
        saver._write_conn  # first touch connects
        assert len(opened) == 1
    finally:
        saver.close()


def test_connection_budget_fits_pooler():
    # Supabase pooler session mode: max 15 clients. delivery-api pins
    # POOL_OPEN_MAX=6 (asserted in delivery-api's own budget test); this
    # service peaks at ENGINE_PEAK_SESSION_CONNS per served session (registry
    # serving, lease heartbeat, application base, lazy saver write conn).
    from interview_engine.http_service import ENGINE_PEAK_SESSION_CONNS
    assert ENGINE_PEAK_SESSION_CONNS == 4
    assert 6 + 2 * ENGINE_PEAK_SESSION_CONNS <= 15, \
        'delivery pool + two concurrent engine sessions must fit the pooler budget'
