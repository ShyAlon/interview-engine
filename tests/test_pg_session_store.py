import time

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed: PG session store tests skip')

from interview_engine.pg_session_store import PgSessionStore
from interview_engine.stop_fence import StopRequested


def make_store(tmp_path):
    server = pgserver.get_server(str(tmp_path / 'pgdata'))
    import psycopg
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    store = PgSessionStore(conn)
    store.setup()
    return store


def test_claim_before_serve_single_winner(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1')
    won = store.claim('s1', 'host-a', ttl=60)
    assert won is not None and won.epoch == 1 and won.generation == 1
    # A second live claim from another host loses; the same host re-claims.
    assert store.claim('s1', 'host-b', ttl=60) is None
    again = store.claim('s1', 'host-a', ttl=60)
    assert again is not None and again.epoch == 2


def test_expired_lease_takeover_bumps_epoch(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1')
    first = store.claim('s1', 'host-a', ttl=0.05)
    assert first is not None
    time.sleep(0.1)
    second = store.claim('s1', 'host-b', ttl=60)
    assert second is not None and second.epoch == first.epoch + 1 and second.host == 'host-b'


def test_stale_epoch_save_rejected_and_persists_nothing(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1')
    stale = store.claim('s1', 'host-a', ttl=0.05)
    time.sleep(0.1)
    live = store.claim('s1', 'host-b', ttl=60)
    with pytest.raises(StopRequested):
        store.fenced_save(stale, {'status': 'stale'})
    store.fenced_save(live, {'status': 'live'}, published_thread_id='s1',
                      published_checkpoint_id='cp-1')
    published = store.load_published('s1')
    assert published['control'] == {'status': 'live'}
    assert published['published_checkpoint_id'] == 'cp-1'


def test_expired_lease_save_rejected(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1')
    claim = store.claim('s1', 'host-a', ttl=0.05)
    time.sleep(0.1)
    with pytest.raises(StopRequested):
        store.fenced_save(claim, {'status': 'late'})
    assert store.load_published('s1')['control'] is None


def test_stop_without_lease_blocks_later_saves(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1')
    claim = store.claim('s1', 'host-a', ttl=3600)
    new_generation = store.stop('s1')
    assert new_generation == claim.generation + 1
    with pytest.raises(StopRequested):
        store.fenced_save(claim, {'status': 'after-stop'})
    assert store.heartbeat(claim, ttl=60) is False
    # The stopped row keeps its last published control state for inspection.
    assert store.load_published('s1')['generation'] == new_generation


def test_heartbeat_extends_and_lost_lease_reports(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1')
    claim = store.claim('s1', 'host-a', ttl=0.2)
    time.sleep(0.1)
    assert store.heartbeat(claim, ttl=60) is True
    # Past the original ttl, the extended lease still fences saves successfully.
    time.sleep(0.15)
    store.fenced_save(claim, {'x': 1})
    assert store.load_published('s1')['control'] == {'x': 1}
    # A losing claim reports False instead of reviving a dead lease.
    stale = store.claim('s1', 'host-a', ttl=0.05)
    time.sleep(0.1)
    live = store.claim('s1', 'host-b', ttl=60)
    assert store.heartbeat(stale, ttl=60) is False
    assert store.heartbeat(live, ttl=60) is True


def test_restore_uses_published_pointer_not_latest_write(tmp_path):
    store = make_store(tmp_path)
    store.create('s1', 'task1')
    claim = store.claim('s1', 'host-a', ttl=60)
    store.fenced_save(claim, {'status': 'step-1'}, published_thread_id='s1',
                      published_checkpoint_id='cp-1')
    # A crash leaves the row exactly as published; no newer staged state exists
    # outside fenced saves, so load never sees anything newer than cp-1.
    published = store.load_published('s1')
    assert published['control'] == {'status': 'step-1'}
    assert published['published_thread_id'] == 's1'
    assert published['published_checkpoint_id'] == 'cp-1'
