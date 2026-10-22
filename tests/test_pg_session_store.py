"""PgSessionStore contract tests on a throwaway real Postgres (pgserver).

The harness DDL below MIRRORS the authoritative delivery-api migration
20261022090000_session_placement_leases.sql column-for-column on
engine_sessions, minus the cross-schema foreign keys (auth.users/tasks do not
exist in this throwaway server). The store itself never creates tables.
"""
import time

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed: PG session store tests skip')

import psycopg
import psycopg.rows

from interview_engine.pg_session_store import PgSessionStore
from interview_engine.stop_fence import StopRequested

# Mirror of delivery-api supabase/migrations/20261007150000_tenant_control_plane.sql
# (engine_sessions) + 20261022090000_session_placement_leases.sql, without the
# auth/tasks foreign keys. Keep column names, types and defaults identical.
MIRRORED_REGISTRY_DDL = """CREATE TABLE engine_sessions(
 session_id TEXT PRIMARY KEY, owner_id UUID NOT NULL, task_id TEXT NOT NULL,
 generation INTEGER NOT NULL CHECK(generation > 0), created_at DOUBLE PRECISION NOT NULL,
 placement_instance TEXT, lease_epoch BIGINT NOT NULL DEFAULT 0,
 lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0, updated_at DOUBLE PRECISION,
 control JSONB, published_thread_id TEXT,
 published_checkpoint_ns TEXT NOT NULL DEFAULT '', published_checkpoint_id TEXT)"""

# Mirror of 20261022090000_session_placement_leases.sql (engine_session_journal).
MIRRORED_JOURNAL_DDL = """CREATE TABLE engine_session_journal(
 session_id TEXT NOT NULL REFERENCES engine_sessions(session_id),
 cursor BIGINT NOT NULL, page JSONB NOT NULL, created_at DOUBLE PRECISION NOT NULL,
 PRIMARY KEY(session_id, cursor))"""

# Mirror of 20261022160000_staged_provider_responses.sql (PR B staged provider
# responses), without the auth FK/RLS. Provider calls in DB mode stage responses here.
MIRRORED_STAGING_DDL = """CREATE TABLE staged_provider_responses(
 owner_id UUID NOT NULL, session_id TEXT NOT NULL, generation INTEGER NOT NULL,
 execution_point TEXT NOT NULL DEFAULT 'bootstrap', node TEXT NOT NULL DEFAULT '',
 stage TEXT NOT NULL, item TEXT NOT NULL DEFAULT '', ordinal INTEGER NOT NULL,
 attempt_id TEXT, request_sha256 TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
 dispatched_at DOUBLE PRECISION NOT NULL, response JSONB, usage JSONB,
 completed_at DOUBLE PRECISION,
 PRIMARY KEY (session_id, generation, execution_point, node, stage, item, ordinal))"""

OWNER_A = '00000000-0000-0000-0000-000000000001'
OWNER_B = '00000000-0000-0000-0000-000000000002'


def make_store(tmp_path):
    server = pgserver.get_server(str(tmp_path / 'pgdata'))
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    conn.execute(MIRRORED_REGISTRY_DDL)
    conn.commit()
    return PgSessionStore(conn)


def test_claim_before_serve_single_winner(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER_A)
    won = store.claim('s1', OWNER_A, 'host-a#1', ttl=60)
    assert won is not None and won.epoch == 1 and won.generation == 1
    # A second live claim from another instance loses - even on the same host.
    assert store.claim('s1', OWNER_A, 'host-a#2', ttl=60) is None


def test_same_instance_reclaim_is_idempotent_no_epoch_bump(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER_A)
    first = store.claim('s1', OWNER_A, 'host-a#1', ttl=60)
    again = store.claim('s1', OWNER_A, 'host-a#1', ttl=60)
    assert again is not None and again.epoch == first.epoch and again.generation == first.generation
    # The idempotent re-claim does not invalidate in-flight fenced writes.
    store.fenced_save(first, {'status': 'still-valid'})
    assert store.load_published('s1', OWNER_A)['control'] == {'status': 'still-valid'}


def test_expired_lease_takeover_bumps_epoch(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER_A)
    first = store.claim('s1', OWNER_A, 'host-a#1', ttl=0.05)
    time.sleep(0.1)
    second = store.claim('s1', OWNER_A, 'host-b#1', ttl=60)
    assert second is not None and second.epoch == first.epoch + 1 and second.instance == 'host-b#1'


def test_stale_epoch_save_rejected_and_persists_nothing(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER_A)
    stale = store.claim('s1', OWNER_A, 'host-a#1', ttl=0.05)
    time.sleep(0.1)
    live = store.claim('s1', OWNER_A, 'host-b#1', ttl=60)
    with pytest.raises(StopRequested):
        store.fenced_save(stale, {'status': 'stale'})
    store.fenced_save(live, {'status': 'live'}, published_thread_id='s1',
                      published_checkpoint_id='cp-1')
    published = store.load_published('s1', OWNER_A)
    assert published['control'] == {'status': 'live'}
    assert published['published_checkpoint_id'] == 'cp-1'


def test_expired_lease_save_rejected(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER_A)
    claim = store.claim('s1', OWNER_A, 'host-a#1', ttl=0.05)
    time.sleep(0.1)
    with pytest.raises(StopRequested):
        store.fenced_save(claim, {'status': 'late'})
    assert store.load_published('s1', OWNER_A)['control'] is None


def test_stop_without_lease_blocks_later_saves(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER_A)
    claim = store.claim('s1', OWNER_A, 'host-a#1', ttl=3600)
    new_generation = store.stop('s1', OWNER_A)
    assert new_generation == claim.generation + 1
    with pytest.raises(StopRequested):
        store.fenced_save(claim, {'status': 'after-stop'})
    assert store.heartbeat(claim, ttl=60) is False
    assert store.load_published('s1', OWNER_A)['generation'] == new_generation


def test_heartbeat_extends_and_lost_lease_reports(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER_A)
    claim = store.claim('s1', OWNER_A, 'host-a#1', ttl=0.2)
    time.sleep(0.1)
    assert store.heartbeat(claim, ttl=60) is True
    time.sleep(0.15)  # past the original ttl; the extension holds
    store.fenced_save(claim, {'x': 1})
    assert store.load_published('s1', OWNER_A)['control'] == {'x': 1}
    stale = claim
    time.sleep(0.0)
    store.stop('s1', OWNER_A)
    assert store.heartbeat(stale, ttl=60) is False


def test_cross_owner_mutations_match_no_row(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER_A)
    assert store.claim('s1', OWNER_B, 'host-b#1', ttl=60) is None
    assert store.load_published('s1', OWNER_B) is None
    with pytest.raises(KeyError):
        store.stop('s1', OWNER_B)
    claim = store.claim('s1', OWNER_A, 'host-a#1', ttl=60)
    forged = type(claim)('s1', OWNER_B, claim.generation, claim.epoch, claim.instance)
    with pytest.raises(StopRequested):
        store.fenced_save(forged, {'status': 'forged'})
    assert store.heartbeat(forged, ttl=60) is False
    # The real owner's session is untouched by every forged attempt.
    assert store.load_published('s1', OWNER_A)['generation'] == claim.generation


def test_restore_uses_published_pointer_not_latest_write(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1', OWNER_A)
    claim = store.claim('s1', OWNER_A, 'host-a#1', ttl=60)
    store.fenced_save(claim, {'status': 'step-1'}, published_thread_id='s1',
                      published_checkpoint_id='cp-1')
    published = store.load_published('s1', OWNER_A)
    assert published['control'] == {'status': 'step-1'}
    assert published['published_thread_id'] == 's1'
    assert published['published_checkpoint_id'] == 'cp-1'
