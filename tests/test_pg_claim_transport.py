from conftest import managed_pool
"""Pooler transport regression: startup-option GUCs are not a claim transport.

2026-10-25 live probe (user-run against the preview DSN): host
aws-0-us-east-1.pooler.supabase.com, read_only_requested=on but
read_only_verified=off, and startup claims visible in neither of two sampled
transactions - the pooler drops the conninfo options payload. The engine
previously carried tenant claims exactly there, and every FORCE-RLS engine
write then failed with SQLSTATE 42501 (the live interview-start failure).
The exact live statement was never captured; this file pins the mechanism.

Against the REAL delivery-api migrated schema as delivery_app (NOSUPERUSER,
NOBYPASSRLS):
1. mechanism: with startup options dropped at connect (the pooler behavior),
   the old claim transport fails with the live error class;
2. fix: the production store and saver paths pass through the same
   option-dropping transport, because claims now ride each transaction.
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
SESSION_A = '00000000-0000-0000-0000-0000000000a1'
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
                  " VALUES ('ict','ClaimTransport','pending',0,0,%s) ON CONFLICT DO NOTHING", (A,))
    admin.close()
    _STATE['dsn'] = dsn
    yield
    _STATE.clear()


def _dropping_connect(dsn, *args, **kwargs):
    """The pooler behavior: the conninfo options payload never reaches the backend."""
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    parts = conninfo_to_dict(dsn)
    parts.pop('options', None)
    return psycopg.Connection.connect(make_conninfo(**parts), *args, **kwargs)


def test_startup_option_claims_dropped_reproduces_live_rls_failure():
    # The engine's OLD transport: claims in the conninfo options payload.
    dsn = (f"{_STATE['dsn']} user=delivery_app options='-c request.jwt.claim.sub={A}"
           f" -c request.jwt.claim.session_id={SESSION_A}'")
    with _dropping_connect(dsn, autocommit=False, row_factory=psycopg.rows.dict_row) as conn:
        row = conn.execute("SELECT current_setting('request.jwt.claim.sub', true) AS sub").fetchone()
        assert row['sub'] is None  # startup options never took effect
        with pytest.raises(psycopg.errors.InsufficientPrivilege) as failure:
            conn.execute(
                "INSERT INTO engine_sessions(session_id,owner_id,task_id,generation,created_at,lease_epoch,lease_expires_at)"
                " VALUES (%s,%s,%s,1,%s,0,0)",
                ('ict-' + time.strftime('%H%M%S'), A, 'ict', time.time()))
        assert failure.value.sqlstate == '42501'


def test_production_store_paths_pass_through_option_dropping_transport(monkeypatch):
    # Every engine connection created through pg_claims goes through a
    # transport that drops startup options: claims must still arrive, because
    # they ride each transaction.
    from interview_engine import pg_claims
    monkeypatch.setattr(pg_claims.psycopg, 'connect', _dropping_connect)
    from interview_engine.tenant_worker import pg_session_store

    from conftest import managed_pool
    store = pg_session_store(managed_pool(f"{_STATE['dsn']} user=delivery_app"), A, SESSION_A)
    sid = 'ict-' + time.strftime('%H%M%S')
    store.create(sid, 'ict', A)
    claim = store.claim(sid, A, 'transport#1', ttl=60)
    assert claim is not None
    # Heartbeat path: one claimed transaction on a borrowed connection per call.
    assert store.heartbeat(claim, ttl=60) is True
    # Staging path: dedicated connection, one claimed transaction per call.
    store.record_dispatch(claim, execution_point='ep', node='n', stage='s', item='i',
                          ordinal=0, request_sha256='sha', provider='p', model='m')
    rows = store.staged_log(sid, claim.generation, execution_point='ep', node='n', stage='s', item='i')
    assert len(rows) == 1
    # Publish + journal + stop: the serving connection under the same transport.
    store.publish_step(claim, {'session_id': sid, 'sequence': 1},
                       journal_pages=[(1, {'cursor': 1, 'stage': 's'})], published_thread_id=sid)
    assert store.load_published(sid, A)['control']['sequence'] == 1
    store.stop(sid, A)


def test_checkpoint_saver_paths_pass_through_option_dropping_transport(monkeypatch):
    from interview_engine import pg_claims
    monkeypatch.setattr(pg_claims.psycopg, 'connect', _dropping_connect)
    from interview_engine.pg_claims import connect_claimed
    from interview_engine.pg_checkpoint import FencedPostgresSaver, LeaseFence
    from interview_engine.pg_session_store import PgSessionStore

    conn = connect_claimed(f"{_STATE['dsn']} user=delivery_app", A, SESSION_A, autocommit=False)
    store = PgSessionStore(managed_pool(conn.info.dsn))
    sid = 'ict-cp-' + time.strftime('%H%M%S')
    store.create(sid, 'ict', A)
    claim = store.claim(sid, A, 'transport#1', ttl=60)
    saver = FencedPostgresSaver(managed_pool(conn.info.dsn), lease=LeaseFence(sid, claim.generation, claim.epoch))
    saver.setup()  # migration pre-recorded: a SELECT, no DDL, under the app role
    cfg = {'configurable': {'thread_id': sid, 'checkpoint_ns': '', 'checkpoint_id': ''}}
    checkpoint = {'v': 4, 'ts': '2026-10-22T14:00:00+00:00', 'id': '00000000-0000-4000-8000-0000000000c2',
                  'channel_values': {'messages': ['m1']}, 'channel_versions': {'messages': '1'},
                  'versions_seen': {}, 'updated_channels': ['messages']}
    written = saver.put(cfg, checkpoint, {'source': 'update', 'step': 1, 'parents': {}}, {'messages': '1'})
    cid = written['configurable']['checkpoint_id']
    got = saver.get_tuple({'configurable': {'thread_id': sid, 'checkpoint_ns': '', 'checkpoint_id': cid}})
    assert got is not None and got.checkpoint['channel_values']['messages'] == ['m1']
