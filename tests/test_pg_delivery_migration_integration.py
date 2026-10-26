from conftest import managed_pool
"""PR A integration: engine store + fenced checkpoint saver against the REAL
delivery-api migrated schema - every migration applied by delivery-api's own
pg_harness - connected as delivery_app (NOSUPERUSER, NOBYPASSRLS) with tenant
JWT claims. No DDL mirror, no superuser, no provider call. Skips unless the
delivery-api checkout is provided via DELIVERY_TEST_DELIVERY_SOURCE.
"""
import os
import sys
import time
from pathlib import Path

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed: integration tests skip')
psycopg = pytest.importorskip('psycopg', reason='psycopg not installed')
import psycopg.rows

DELIVERY = os.getenv('DELIVERY_TEST_DELIVERY_SOURCE')
pytestmark = pytest.mark.skipif(not DELIVERY, reason='requires DELIVERY_TEST_DELIVERY_SOURCE (delivery-api checkout)')

A = '00000000-0000-0000-0000-000000000001'
B = '00000000-0000-0000-0000-000000000002'
_STATE = {}


@pytest.fixture(scope='module', autouse=True)
def migrated():
    if not DELIVERY:
        return
    sys.path.insert(0, str(Path(DELIVERY) / 'tests'))
    import pg_harness
    pg_harness.start()
    dsn = pg_harness.STATE.get('dsn')
    if not dsn:
        pytest.skip('pg_harness unavailable: ' + pg_harness.STATE.get('reason', ''))
    admin = psycopg.connect(dsn, autocommit=True)
    admin.execute("INSERT INTO tasks(task_id,title,state,created_at,updated_at,owner_id)"
                  " VALUES ('ita','Integration','pending',0,0,%s) ON CONFLICT DO NOTHING", (A,))
    admin.close()
    _STATE['dsn'] = dsn
    yield
    _STATE.clear()


# The harness seeds one ACTIVE session per owner, deliberately distinct from the
# owner uuid: the engine_sessions RLS policy requires requi_session_active(),
# i.e. request.jwt.claim.session_id must name a live session owned by
# request.jwt.claim.sub. Claiming session_id == owner is the vacuous
# session==user trap and fails under FORCE RLS.
SESSION = {A: '00000000-0000-0000-0000-0000000000a1',
           B: '00000000-0000-0000-0000-0000000000a2'}


def app_conn(owner):
    # The engine's exact production claim shape: tenant claims ride each
    # transaction (pg_claims), never the conninfo - poolers drop startup-option
    # GUCs, so the conninfo options payload is not a claim transport.
    from interview_engine.pg_claims import connect_claimed
    return connect_claimed(f"{_STATE['dsn']} user=delivery_app",
                           owner, SESSION[owner], autocommit=False)


def test_store_claim_publish_stop_under_app_role():
    from interview_engine.pg_session_store import PgSessionStore
    from interview_engine.stop_fence import StopRequested

    store = PgSessionStore(managed_pool(app_conn(A).info.dsn))
    sid = 'ita-' + time.strftime('%H%M%S')
    store.create(sid, 'ita', A)
    claim = store.claim(sid, A, 'integration#1', ttl=60)
    assert claim is not None and claim.epoch == 1

    # One publish transaction: control + journal pages + checkpoint pointer.
    store.publish_step(claim, {'session_id': sid, 'sequence': 1, 'status': 1},
                       journal_pages=[(1, {'cursor': 1, 'stage': 'scope_gate', 'status': 'completed'}),
                                      (2, {'cursor': 2, 'stage': 'interview_turn', 'status': 'started'})],
                       published_thread_id=sid, published_checkpoint_id='cp-1')
    row = store.load_published(sid, A)
    assert row['control']['sequence'] == 1 and row['published_checkpoint_id'] == 'cp-1'
    assert store.journal_head(sid, A) == 2
    # Pagination: after/limit reads committed pages in cursor order.
    assert [p['cursor'] for p in store.read_journal(sid, A, after=0, limit=1)] == [1]
    assert [p['cursor'] for p in store.read_journal(sid, A, after=1, limit=10)] == [2]

    # Owner-authorized stop lands under the app role; the stale claim is fenced.
    store.stop(sid, A)
    assert store.load_published(sid, A)['control']['stopped'] is True
    with pytest.raises(StopRequested):
        store.publish_step(claim, {'session_id': sid, 'sequence': 2})


def test_fenced_checkpoint_roundtrip_under_app_role():
    from interview_engine.pg_checkpoint import FencedPostgresSaver, LeaseFence
    from interview_engine.pg_session_store import PgSessionStore

    conn = app_conn(A)
    store = PgSessionStore(managed_pool(conn.info.dsn))
    sid = 'ita-cp-' + time.strftime('%H%M%S')
    store.create(sid, 'ita', A)
    claim = store.claim(sid, A, 'integration#1', ttl=60)
    saver = FencedPostgresSaver(conn, lease=LeaseFence(sid, claim.generation, claim.epoch))
    saver.setup()  # migration pre-recorded v10: a SELECT, no DDL, under the app role
    cfg = {'configurable': {'thread_id': sid, 'checkpoint_ns': '', 'checkpoint_id': ''}}
    checkpoint = {'v': 4, 'ts': '2026-10-22T14:00:00+00:00', 'id': '00000000-0000-4000-8000-0000000000c1',
                  'channel_values': {'messages': ['m1']}, 'channel_versions': {'messages': '1'},
                  'versions_seen': {}, 'updated_channels': ['messages']}
    written = saver.put(cfg, checkpoint, {'source': 'update', 'step': 1, 'parents': {}}, {'messages': '1'})
    cid = written['configurable']['checkpoint_id']
    got = saver.get_tuple({'configurable': {'thread_id': sid, 'checkpoint_ns': '', 'checkpoint_id': cid}})
    assert got is not None and got.checkpoint['channel_values']['messages'] == ['m1']


def test_cross_owner_invisible_under_rls():
    from interview_engine.pg_session_store import PgSessionStore

    store_a = PgSessionStore(managed_pool(app_conn(A).info.dsn))
    sid = 'ita-xo-' + time.strftime('%H%M%S')
    store_a.create(sid, 'ita', A)
    claim = store_a.claim(sid, A, 'integration#1', ttl=60)
    store_a.publish_step(claim, {'session_id': sid, 'sequence': 1},
                         journal_pages=[(1, {'cursor': 1, 'stage': 's'})], published_thread_id=sid)

    other = PgSessionStore(managed_pool(app_conn(B).info.dsn))
    assert other.load_published(sid, B) is None
    assert other.claim(sid, B, 'integration#2', ttl=60) is None
    assert other.read_journal(sid, B, after=0, limit=10) == []
    with pytest.raises(KeyError):
        other.stop(sid, B)
    # The adoption path is not a cross-owner oracle either: B's create against
    # A's session id conflicts (the row is invisible under RLS, never adopted),
    # and B's heartbeat against A's claim matches no row.
    with pytest.raises(psycopg.errors.UniqueViolation):
        other.create(sid, 'ita', B)
    assert other.heartbeat(claim, ttl=60) is False
    # A's own connection is unharmed by B's failed attempts.
    assert store_a.claim(sid, A, 'integration#1', ttl=60) is not None
