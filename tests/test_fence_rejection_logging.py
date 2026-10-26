"""Hermetic tests for the fence-rejection triage logging (2026-10-26).

Live evidence: a start advance died at 07:54:46Z with only
"stop.swallowed ... reason=stop-requested" in the logs - the swallow hid
WHICH fence tripped, and the six StopRequested raise sites all carried no
lease state. These tests pin the new logging: every rejection names its
site and the failed predicate (generation / epoch / expiry), read from a
best-effort diagnostic row, with no behavior change on the raise itself.
"""
import time
import pytest
from interview_engine.stop_fence import StopFence, StopRequested
from interview_engine.pg_session_store import PgSessionStore, SessionClaim
from interview_engine.pg_checkpoint import FencedPostgresSaver, LeaseFence


class _Cur:
    def __init__(self, row):
        self._row = row
    def fetchone(self):
        return self._row


class _FakeConn:
    """Serves a scripted sequence of fetchone rows; records nothing else."""
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = 0
        self.rollbacks = 0
    def execute(self, *args, **kwargs):
        row = self.rows[min(self.calls, len(self.rows) - 1)]
        self.calls += 1
        return _Cur(row)
    def rollback(self):
        self.rollbacks += 1


def test_stop_fence_check_logs_requested(tmp_path, caplog):
    f = StopFence(tmp_path / 'stop.json')
    f.request_stop()
    with caplog.at_level('WARNING', logger='interview_engine.stop_fence'):
        with pytest.raises(StopRequested):
            f.check(1)
    msgs = [r.getMessage() for r in caplog.records]
    assert any('stop.fence.tripped' in m and 'site=check' in m and 'requested=True' in m for m in msgs), msgs


def test_stop_fence_check_logs_generation_mismatch(tmp_path, caplog):
    f = StopFence(tmp_path / 'stop.json')
    with caplog.at_level('WARNING', logger='interview_engine.stop_fence'):
        with pytest.raises(StopRequested):
            f.check(99)
    msgs = [r.getMessage() for r in caplog.records]
    assert any('expected_generation=1' in m and 'seen_generation=99' in m and 'requested=False' in m for m in msgs), msgs


def test_session_store_fence_logs_expired_lease(caplog):
    claim = SessionClaim('sid-1', 'owner-1', 1, 1, 'inst-1')
    conn = _FakeConn([None, {'generation': 1, 'lease_epoch': 1,
                             'lease_expires_at': time.time() - 5, 'placement_instance': 'inst-1'}])
    store = PgSessionStore(conn)
    with caplog.at_level('WARNING', logger='interview_engine.pg_session_store'):
        with pytest.raises(StopRequested, match='rejected the save'):
            store.fenced_save(claim, {'sequence': 1})
    msgs = [r.getMessage() for r in caplog.records]
    assert any('lease.fence.rejected' in m and 'what=save' in m and 'expired' in m for m in msgs), msgs
    assert conn.rollbacks == 1


def test_session_store_fence_logs_epoch_and_generation_mismatch(caplog):
    claim = SessionClaim('sid-1', 'owner-1', 1, 1, 'inst-1')
    future = time.time() + 60
    conn = _FakeConn([None, {'generation': 2, 'lease_epoch': 3,
                             'lease_expires_at': future, 'placement_instance': 'inst-2'}])
    store = PgSessionStore(conn)
    with caplog.at_level('WARNING', logger='interview_engine.pg_session_store'):
        with pytest.raises(StopRequested):
            store.fenced_save(claim, {'sequence': 1})
    msgs = [r.getMessage() for r in caplog.records]
    assert any('generation 2 != 1' in m and 'lease_epoch 3 != 1' in m and 'expired' not in m for m in msgs), msgs


def test_session_store_fence_diagnostic_failure_never_masks_rejection(caplog):
    class FailingConn:
        def __init__(self):
            self.calls = 0
        def execute(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                return _Cur(None)
            raise RuntimeError('diagnostic read failed')
        def rollback(self):
            pass
    claim = SessionClaim('sid-1', 'owner-1', 1, 1, 'inst-1')
    with caplog.at_level('WARNING', logger='interview_engine.pg_session_store'):
        with pytest.raises(StopRequested, match='rejected the save'):
            PgSessionStore(FailingConn()).fenced_save(claim, {'sequence': 1})
    msgs = [r.getMessage() for r in caplog.records]
    assert any('diagnostic-read-failed' in m for m in msgs), msgs


def test_checkpoint_fence_logs_expired_lease(caplog):
    saver = FencedPostgresSaver.__new__(FencedPostgresSaver)
    saver._lease = LeaseFence('sid-1', 1, 1)
    saver._lease_table = 'engine_sessions'
    saver._write_conn = _FakeConn([None, {'generation': 1, 'lease_epoch': 1,
                                          'lease_expires_at': time.time() - 9}])
    with caplog.at_level('WARNING', logger='interview_engine.pg_checkpoint'):
        with pytest.raises(StopRequested, match='lease lost'):
            saver._fenced('SELECT count(*) FROM guard', {})
    msgs = [r.getMessage() for r in caplog.records]
    assert any('lease.fence.rejected' in m and 'what=checkpoint write' in m and 'expired' in m for m in msgs), msgs
