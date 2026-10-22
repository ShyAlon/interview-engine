"""Postgres checkpoint saver with DB-atomic fenced writes.

Single-host file fencing (StopFence + fcntl) cannot extend across hosts: the
guard and the write must be one database statement. Every put_writes row here
is a conditional insert: a guard CTE locks the session's lease row
(FOR UPDATE) and requires the writer's claimed (generation, epoch) with an
unexpired lease; a stop/generation bump updates the same row, so bump and
write serialize on the row lock with no epsilon window. A failed guard raises
StopRequested - nothing is silently dropped.

put() is fenced too: the library's checkpoints upsert is ON CONFLICT DO
UPDATE, so a same-id re-put OVERWRITES the checkpoint payload and metadata
(blob bytes are safe - blobs are DO NOTHING and channel values live there).
A stale writer that knows a published checkpoint id could otherwise rewrite
its metadata/structure. The guard is the first statement of an explicit
transaction holding the lease row lock to commit, so a stop/takeover bump
serializes against the whole put.

Channel semantics mirror the base saver exactly: reserved channels (those in
WRITES_IDX_MAP - error/interrupt/resume) guarded-UPSERT, regular task writes
guarded-INSERT first-writer-wins.

Same-key collisions are first-writer-wins (ON CONFLICT DO NOTHING), so a stale
or retried batch can never overwrite a legitimate recorded write. Checkpoint
and blob payloads are immutable by the library's own upsert semantics
(verified against langgraph-checkpoint-postgres 3.1.2).

Restore needs no filter: only legitimately-fenced writes exist. A takeover
(new generation/epoch) therefore keeps valid prior-epoch pending work instead
of rerunning it.
"""
from __future__ import annotations

import threading

import psycopg
from langgraph.checkpoint.base import WRITES_IDX_MAP
from langgraph.checkpoint.postgres import PostgresSaver

from .stop_fence import StopRequested

LEASE_TABLE = 'engine_sessions'

# One statement: lock the lease row, verify the claimed fence, insert only
# while it holds. The bump path takes the same row lock, so a stop either
# lands first (guard fails, we raise) or after (the write was legitimate).
_GUARD = """guard AS (
    SELECT 1 FROM {lease_table}
    WHERE session_id = %(fence_session)s
      AND generation = %(fence_generation)s
      AND lease_epoch = %(fence_epoch)s
      AND lease_expires_at > %(fence_now)s
    FOR UPDATE
)"""
_INSERT_WRITES = f"""WITH {_GUARD},
ins AS (
    INSERT INTO checkpoint_writes
        (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    SELECT %(thread_id)s, %(checkpoint_ns)s, %(checkpoint_id)s, %(task_id)s, %(task_path)s,
           %(idx)s, %(channel)s, %(type)s, %(blob)s FROM guard
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
    RETURNING 1
)
SELECT count(*) FROM guard"""

_UPSERT_WRITES = f"""WITH {_GUARD},
ins AS (
    INSERT INTO checkpoint_writes
        (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    SELECT %(thread_id)s, %(checkpoint_ns)s, %(checkpoint_id)s, %(task_id)s, %(task_path)s,
           %(idx)s, %(channel)s, %(type)s, %(blob)s FROM guard
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
        channel = EXCLUDED.channel, type = EXCLUDED.type, blob = EXCLUDED.blob
    RETURNING 1
)
SELECT count(*) FROM guard"""


class LeaseFence:
    """The writer's claimed lease: identity of the guarded session row."""

    def __init__(self, session_id: str, generation: int, epoch: int, *, now= None):
        if type(generation) is not int or generation < 1:
            raise ValueError('positive lease generation required')
        if type(epoch) is not int or epoch < 1:
            raise ValueError('positive lease epoch required')
        self.session_id = session_id
        self.generation = generation
        self.epoch = epoch
        import time
        self._now = now or time.time

    def params(self) -> dict:
        return {'fence_session': self.session_id, 'fence_generation': self.generation,
                'fence_epoch': self.epoch, 'fence_now': self._now()}


class FencedPostgresSaver(PostgresSaver):
    """PostgresSaver whose every mutation is fenced against the DB lease row."""

    def __init__(self, conn, *, lease: LeaseFence, pipe=None, serde=None,
                 lease_table: str = LEASE_TABLE, on_checkpoint_committed=None):
        super().__init__(conn, pipe=pipe, serde=serde)
        if not isinstance(lease, LeaseFence):
            raise ValueError('explicit lease fence required')
        if not lease_table or "'" in lease_table:
            raise ValueError('trusted lease table name required')
        self._lease = lease
        self._lease_table = lease_table
        # Called with (checkpoint_id, super-step) after each committed put: the
        # staged provider-response context advances its deterministic super-step
        # counter here (checkpoint ids are random per commit; the step is not).
        self.on_checkpoint_committed = on_checkpoint_committed
        # Guarded writes use a dedicated connection: graph nodes run on worker
        # threads and the base saver drives its connection in pipeline mode,
        # so sharing it for fenced statements corrupts the pipeline.
        self._write_conn = psycopg.Connection.connect(conn.info.dsn, autocommit=False)
        self._write_lock = threading.Lock()

    def close(self):
        self._write_conn.close()

    def setup(self) -> None:
        # The checkpoint schema is migration-owned (delivery-api pre-records
        # the final langgraph migration index). The runtime database role must
        # never run DDL: when the schema is fully prepared there is nothing to
        # do, and when it is not, fail clearly instead of mutating the
        # database through library DDL. Tests and development bootstrap
        # explicitly through bootstrap_checkpoint_schema under a superuser.
        version = None
        try:
            row = self.conn.execute("SELECT max(v) AS v FROM checkpoint_migrations").fetchone()
        except Exception:
            row = None
        if row is not None:
            version = row.get('v') if isinstance(row, dict) else row[0]
        if version is None or version < len(self.MIGRATIONS) - 1:
            raise RuntimeError(
                'checkpoint schema is not prepared: apply the delivery-api '
                'checkpoint migration (20261022120000_session_checkpoint_tables); '
                'the runtime database role never runs DDL')

    def _fenced(self, sql: str, params: dict):
        merged = dict(params)
        merged.update(self._lease.params())
        row = self._write_conn.execute(sql, merged).fetchone()
        guard = (next(iter(row.values())) if isinstance(row, dict) else row[0]) if row is not None else 0
        if guard == 0:
            self._write_conn.rollback()
            raise StopRequested('session lease lost (stop or takeover)')
        return row

    def _guard_lock(self):
        row = self._write_conn.execute(
            f"SELECT 1 FROM {self._lease_table} WHERE session_id = %(fence_session)s"
            " AND generation = %(fence_generation)s AND lease_epoch = %(fence_epoch)s"
            " AND lease_expires_at > %(fence_now)s FOR UPDATE",
            self._lease.params()).fetchone()
        guard = (next(iter(row.values())) if isinstance(row, dict) else row[0]) if row is not None else 0
        if guard == 0:
            self._write_conn.rollback()
            raise StopRequested('session lease lost (stop or takeover)')

    def put(self, config, checkpoint, metadata, new_versions):
        # Fenced put, mirroring the base saver byte-for-byte: primitive channel
        # values inline in the checkpoints row, everything else in
        # checkpoint_blobs (DO NOTHING - immutable), checkpoints upsert (DO
        # UPDATE). The lease row lock is held for the whole transaction, so a
        # stop/takeover bump serializes against every row of the put.
        from langgraph.checkpoint.serde.types import _DeltaSnapshot
        from langgraph.checkpoint.base import get_serializable_checkpoint_metadata
        from psycopg.types.json import Jsonb
        configurable = config['configurable'].copy()
        thread_id = configurable.pop('thread_id')
        checkpoint_ns = configurable.pop('checkpoint_ns')
        checkpoint_id = configurable.pop('checkpoint_id', None)
        copy = checkpoint.copy()
        copy['channel_values'] = copy['channel_values'].copy()
        blob_values = {}
        for k, v in checkpoint['channel_values'].items():
            if isinstance(v, _DeltaSnapshot):
                blob_values[k] = copy['channel_values'].pop(k)
                copy['channel_values'][k] = True
            elif v is None or isinstance(v, (str, int, float, bool)):
                pass
            else:
                blob_values[k] = copy['channel_values'].pop(k)
        blob_versions = {k: v for k, v in new_versions.items() if k in blob_values}
        with self._write_lock:
            try:
                self._guard_lock()
                with self._write_conn.cursor() as cur:
                    if blob_versions:
                        cur.executemany(
                            "INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)"
                            " VALUES (%s,%s,%s,%s,%s,%s)"
                            " ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING",
                            [(thread_id, checkpoint_ns, k, str(blob_versions[k]),
                              *self.serde.dumps_typed(blob_values[k])) for k in blob_versions])
                    cur.execute(
                        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id,"
                        " parent_checkpoint_id, checkpoint, metadata) VALUES (%s,%s,%s,%s,%s,%s)"
                        " ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET"
                        " checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata",
                        (thread_id, checkpoint_ns, checkpoint['id'], checkpoint_id,
                         Jsonb(copy), Jsonb(get_serializable_checkpoint_metadata(config, metadata))))
            except BaseException:
                self._write_conn.rollback()
                raise
            else:
                self._write_conn.commit()
        if self.on_checkpoint_committed is not None:
            self.on_checkpoint_committed(checkpoint['id'], (metadata or {}).get('step', -1))
        return {'configurable': {'thread_id': thread_id, 'checkpoint_ns': checkpoint_ns,
                                 'checkpoint_id': checkpoint['id']}}

    def put_writes(self, config, writes, task_id, task_path=''):
        # Same contract as the base saver, each write its own guarded insert so
        # a fence loss mid-batch raises before any later write lands.
        # Reserved channels overwrite (library retry semantics); regular task
        # writes are first-writer-wins so a stale batch never replaces a
        # legitimate recorded write. Per-row guarded inserts: correctness over
        # batching at interview-session throughput.
        reserved = all(channel in WRITES_IDX_MAP for channel, _ in writes)
        sql = (_UPSERT_WRITES if reserved else _INSERT_WRITES).format(lease_table=self._lease_table)
        with self._write_lock:
            try:
                for idx, (channel, value) in enumerate(writes):
                    type_, blob = self.serde.dumps_typed(value)
                    self._fenced(sql, {
                        'thread_id': config['configurable']['thread_id'],
                        'checkpoint_ns': config['configurable']['checkpoint_ns'],
                        'checkpoint_id': config['configurable']['checkpoint_id'],
                        'task_id': task_id, 'task_path': task_path,
                        'idx': WRITES_IDX_MAP.get(channel, idx),
                        'channel': channel, 'type': type_, 'blob': blob})
            except BaseException:
                self._write_conn.rollback()
                raise
            else:
                self._write_conn.commit()

def bootstrap_checkpoint_schema(conn) -> None:
    """Explicit one-off schema bootstrap for tests and development superusers.

    Production databases are prepared by the delivery-api migration; this runs
    the pinned library DDL directly and is never called from runtime code.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    PostgresSaver(conn).setup()
