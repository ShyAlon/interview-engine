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


# Every pool, connection and registry a test opens is closed during teardown,
# BEFORE the throwaway server goes away: no heartbeat loop or socket outlives
# its Postgres.
_OPEN = {'servers': [], 'conns': [], 'registries': [], 'pools': []}


@pytest.fixture(autouse=True)
def _cleanup_pg():
    yield
    for r in _OPEN['registries']:
        r.close()
    for c in _OPEN['conns']:
        try: c.close()
        except Exception: pass
    for pl in _OPEN['pools']:
        try: pl.close()
        except Exception: pass
    _OPEN['registries'].clear(); _OPEN['conns'].clear(); _OPEN['servers'].clear(); _OPEN['pools'].clear()


def make_conn(tmp_path, name='pgdata'):
    server = managed_pg_server(str(tmp_path / name))
    _OPEN['servers'].append(server)
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    conn._pg_server = server  # so make_pool can build the pool on the same DSN
    conn.execute(MIRRORED_REGISTRY_DDL)
    conn.commit()
    _OPEN['conns'].append(conn)
    return conn


def make_pool(server, max_size=4):
    from interview_engine.pg_pool import build_pool
    pool = build_pool(server.get_uri(), max_size=max_size, name='test')
    _OPEN['pools'].append(pool)
    return pool


def make_registry(conn, program=None, instance='test#1', ttl=60.0, heartbeat_interval=None):
    # heartbeat_interval defaults to the production ratio (ttl/3): the shared
    # loop beats every held lease on one borrowed connection per tick.
    pool = make_pool(conn._pg_server) if hasattr(conn, '_pg_server') else None
    if pool is None:
        raise AssertionError('make_registry needs a conn from make_conn (server-tracked)')
    registry = SessionRegistry(program or (lambda *a: Program()),
                               session_store=PgSessionStore(pool), owner_id=OWNER,
                               instance_id=instance, lease_ttl=ttl,
                               heartbeat_interval=heartbeat_interval if heartbeat_interval is not None else max(ttl/3.0, 0.05))
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
    row = PgSessionStore(make_pool(conn._pg_server)).load_published(session.session_id, OWNER)
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
    store = PgSessionStore(make_pool(conn._pg_server))
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
    foreign = SessionRegistry(lambda *a: Program(), session_store=PgSessionStore(make_pool(conn._pg_server)),
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
    store = PgSessionStore(make_pool(conn._pg_server))
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
    loop = first._hb_thread
    assert loop is not None and not loop.is_alive()  # shared loop halted and joined, no straggler
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


def test_saver_borrows_per_write_and_returns_to_pool(tmp_path, monkeypatch):
    # The saver holds NO connection of its own: each fenced write batch checks
    # one out of the shared pool and returns it, so an idle session holds zero.
    from interview_engine import pg_checkpoint
    from interview_engine.pg_checkpoint import FencedPostgresSaver, LeaseFence
    conn = make_conn(tmp_path, name='pgdata-lazy')
    pool = make_pool(conn._pg_server, max_size=2)
    pool.wait(timeout=10)
    opened = []
    real_connect = psycopg.Connection.connect
    def counting_connect(*a, **k):
        opened.append(a); return real_connect(*a, **k)
    monkeypatch.setattr(pg_checkpoint.psycopg.Connection, 'connect', staticmethod(counting_connect))
    from interview_engine.pg_checkpoint import bootstrap_checkpoint_schema
    conn.autocommit = True  # CREATE INDEX CONCURRENTLY needs a non-transaction conn
    bootstrap_checkpoint_schema(conn)  # superuser test bootstrap; the saver itself never runs DDL
    conn.autocommit = False
    import time as _time
    sid = str(uuid4())
    conn.execute("INSERT INTO engine_sessions(session_id,owner_id,task_id,generation,created_at,lease_epoch,lease_expires_at)"
                 " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                 (sid, OWNER, 'task1', 1, _time.time(), 1, _time.time()+3600))
    conn.commit()
    saver = FencedPostgresSaver(pool, lease=LeaseFence(sid, 1, 1), claims=None)
    saver.setup()
    cfg = {'configurable': {'thread_id': 't1', 'checkpoint_ns': '', 'checkpoint_id': 'c1'}}
    state = {'v': 4, 'ts': '2026-10-26T08:00:00+00:00', 'id': '00000000-0000-4000-8000-000000000001',
             'channel_values': {}, 'channel_versions': {}, 'versions_seen': {}}
    saver.put(cfg, state, {'source': 'test'}, {})
    saver.put(cfg, state, {'source': 'test'}, {})
    assert len(opened) <= 2, f'saver writes must draw on the bounded pool, not open connections beyond it: {len(opened)} opened'
    stats = pool.get_stats()
    assert stats['pool_available'] == stats['pool_size'], \
        f'saver leaked a checked-out connection between writes: {stats}'


def test_connection_budget_fits_pooler():
    # Supabase pooler session mode: max 15 clients. ONE pool per process now
    # bounds this service's entire footprint regardless of session count:
    # delivery-api pins POOL_OPEN_MAX=6 (its own budget test) and injects the
    # pool when hosting the engine in-process; a standalone engine builds its
    # own bounded at ENGINE_POOL_MAX.
    from interview_engine.pg_pool import ENGINE_POOL_MAX
    assert 6 + ENGINE_POOL_MAX <= 15, \
        'delivery pool + one engine pool must fit the pooler budget with headroom'
