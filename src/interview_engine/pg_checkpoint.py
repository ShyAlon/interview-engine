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
from functools import wraps

def _answer_cache_locked(method):
    @wraps(method)
    def locked(self,*args,**kwargs):
        lock=getattr(self,'_write_lock',None)
        if lock is None:return method(self,*args,**kwargs)
        with lock:return method(self,*args,**kwargs)
    return locked


from contextlib import contextmanager

import psycopg
from langgraph.checkpoint.base import WRITES_IDX_MAP
from langgraph.checkpoint.postgres import PostgresSaver
import logging

log = logging.getLogger("interview_engine.pg_checkpoint")

from .stop_fence import StopRequested

from contextvars import ContextVar
from copy import deepcopy
_answer_checkpoint_memory=ContextVar('answer_checkpoint_memory',default=False)

@contextmanager
def answer_checkpoint_memory():
    token=_answer_checkpoint_memory.set(True)
    try:yield
    finally:_answer_checkpoint_memory.reset(token)

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
                 lease_table: str = LEASE_TABLE, claims: tuple | None = None):
        super().__init__(conn, pipe=pipe, serde=serde)
        if not isinstance(lease, LeaseFence):
            raise ValueError('explicit lease fence required')
        if not lease_table or "'" in lease_table:
            raise ValueError('trusted lease table name required')
        self._lease = lease
        self._lease_table = lease_table
        # Per-transaction tenant claims (pg_claims): carried by every
        # connection the saver drives. None only in mirrored-DDL superuser
        # tests; the production wiring always passes them.
        self._claims = claims if claims is not None else getattr(conn, 'claims', None)
        # Guarded writes borrow one claimed connection+transaction per batch
        # from the process pool (_guarded_conn): no per-session pinned
        # connection at all. self.conn may BE the pool - langgraph's base saver
        # checks out per operation (ConnectionPool is supported natively);
        # pipeline mode is unavailable on a pool, which costs a few round
        # trips per checkpoint commit and changes no semantics.
        self._write_lock = threading.RLock()
        self._answer_enabled=_answer_checkpoint_memory.get()

    @_answer_cache_locked
    def get_tuple(self,config):
        if not (_answer_checkpoint_memory.get() or getattr(self,'_answer_enabled',False)):return super().get_tuple(config)
        if not hasattr(self,'_answer_memory'):self._answer_memory={};self._answer_latest={}
        cfg=config.get('configurable',{})
        from langgraph.checkpoint.base import get_checkpoint_id
        key=(cfg.get('thread_id'),cfg.get('checkpoint_ns',''),get_checkpoint_id(config) or None)
        if key[2] is None and key[:2] in self._answer_latest:key=(*key[:2],self._answer_latest[key[:2]])
        if key not in self._answer_memory:
            value=super().get_tuple(config)
            # Cache only a real checkpoint, not a claim that no checkpoint can
            # ever appear. Explicit pinned reads remain pinned.
            if value is not None:
                actual=(*key[:2],value.checkpoint['id'])
                self._answer_memory[actual]=deepcopy(value)
                if key[2] is None:self._answer_latest[key[:2]]=actual[2]
            return value
        return deepcopy(self._answer_memory[key])

    @_answer_cache_locked
    def remember_answer_checkpoint(self,config,checkpoint,metadata):
        if not (_answer_checkpoint_memory.get() or getattr(self,'_answer_enabled',False)):return
        from langgraph.checkpoint.base import CheckpointTuple,get_serializable_checkpoint_metadata
        if not hasattr(self,'_answer_memory'):self._answer_memory={};self._answer_latest={}
        cfg=config['configurable'];identity=(cfg['thread_id'],cfg.get('checkpoint_ns',''))
        produced={'configurable':{**cfg,'checkpoint_id':checkpoint['id']}}
        parent=deepcopy(config) if cfg.get('checkpoint_id') else None
        self._answer_memory[(*identity,checkpoint['id'])]=CheckpointTuple(produced,deepcopy(checkpoint),
            get_serializable_checkpoint_metadata(config,metadata),parent,[])
        self._answer_latest[identity]=checkpoint['id']

    @contextmanager
    def _guarded_conn(self):
        # One claimed transaction for a guarded write batch: claims ride the
        # transaction stream; the connection returns to the pool at the end.
        from .pg_claims import claimed_transaction
        from langgraph.checkpoint.postgres import _internal
        with _internal.get_connection(self.conn) as conn:
            try:
                with claimed_transaction(conn,self._claims):
                    yield conn
            finally:
                from psycopg.pq import TransactionStatus
                if conn.info.transaction_status != TransactionStatus.IDLE:
                    conn.rollback()

    @contextmanager
    def _cursor(self, *, pipeline: bool = False):
        # When claims are bound, every saver operation runs in one explicit
        # claimed transaction: on the autocommit base connection an explicit
        # block is the only way transaction-local claims (set_config
        # is_local=true) span the operation's statements. Startup-option GUCs
        # are not used: poolers drop them (pg_claims). Pipeline behavior is
        # preserved inside the claimed transaction.
        if self._claims is None:
            with super()._cursor(pipeline=pipeline) as cur:
                yield cur
            return
        from .pg_claims import claimed_transaction
        from langgraph.checkpoint.postgres import _internal
        with self.lock, _internal.get_connection(self.conn) as conn:
            with claimed_transaction(conn,self._claims):
                # No pipeline on a pooled connection (langgraph forbids it);
                # the read paths here are single round trips anyway.
                with conn.cursor(binary=True, row_factory=psycopg.rows.dict_row) as cur:
                    yield cur

    def setup(self) -> None:
        # The checkpoint schema is migration-owned (delivery-api pre-records
        # the final langgraph migration index). The runtime database role must
        # never run DDL: when the schema is fully prepared there is nothing to
        # do, and when it is not, fail clearly instead of mutating the
        # database through library DDL. Tests and development bootstrap
        # explicitly through bootstrap_checkpoint_schema under a superuser.
        version = None
        try:
            # Routed through _cursor so the claimed transaction carries the
            # tenant boundary for this read too.
            with self._cursor() as cur:
                row = cur.execute("SELECT max(v) AS v FROM checkpoint_migrations").fetchone()
        except Exception:
            row = None
        if row is not None:
            version = row.get('v') if isinstance(row, dict) else row[0]
        if version is None or version < len(self.MIGRATIONS) - 1:
            raise RuntimeError(
                'checkpoint schema is not prepared: apply the delivery-api '
                'checkpoint migration (20261022120000_session_checkpoint_tables); '
                'the runtime database role never runs DDL')

    def _fence_diagnose(self, conn, what: str) -> None:
        """Best-effort read of the current lease row after a guard rejected a
        checkpoint write: logs which predicate failed (generation / epoch /
        expiry) so the rejection is diagnosable from logs alone. Read-only; a
        failed diagnostic never masks the original rejection."""
        try:
            row = conn.execute(
                f"SELECT generation, lease_epoch, lease_expires_at FROM {self._lease_table}"
                " WHERE session_id = %(fence_session)s",
                {'fence_session': self._lease.session_id}).fetchone()
            if row is None:
                log.warning("lease.fence.rejected what=%s session=%s reason=session-row-missing expected gen=%s epoch=%s",
                            what, self._lease.session_id, self._lease.generation, self._lease.epoch)
                return
            get = (lambda k: row.get(k) if isinstance(row, dict) else row[{'generation': 0, 'lease_epoch': 1, 'lease_expires_at': 2}[k]])
            now = self._lease._now()
            reasons = []
            if get('generation') != self._lease.generation:
                reasons.append("generation %s != %s" % (get('generation'), self._lease.generation))
            if get('lease_epoch') != self._lease.epoch:
                reasons.append("lease_epoch %s != %s" % (get('lease_epoch'), self._lease.epoch))
            if not (get('lease_expires_at') or 0) > now:
                reasons.append("lease expired %.1fs ago (ttl lapsed)" % (now - (get('lease_expires_at') or 0)))
            log.warning("lease.fence.rejected what=%s session=%s reason=%s expected gen=%s epoch=%s",
                        what, self._lease.session_id, "; ".join(reasons) or "no predicate mismatch (race)",
                        self._lease.generation, self._lease.epoch)
        except Exception as diag_exc:
            log.warning("lease.fence.rejected what=%s session=%s diagnostic-read-failed %s: %s",
                        what, self._lease.session_id, type(diag_exc).__name__, str(diag_exc)[:200])

    def _fenced(self, conn, sql: str, params: dict):
        merged = dict(params)
        merged.update(self._lease.params())
        row = conn.execute(sql, merged).fetchone()
        guard = (next(iter(row.values())) if isinstance(row, dict) else row[0]) if row is not None else 0
        if guard == 0:
            self._fence_diagnose(conn, 'checkpoint write')
            raise StopRequested('session lease lost (stop or takeover)')
        return row

    def _guard_lock(self, conn):
        row = conn.execute(
            f"SELECT 1 FROM {self._lease_table} WHERE session_id = %(fence_session)s"
            " AND generation = %(fence_generation)s AND lease_epoch = %(fence_epoch)s"
            " AND lease_expires_at > %(fence_now)s FOR UPDATE",
            self._lease.params()).fetchone()
        guard = (next(iter(row.values())) if isinstance(row, dict) else row[0]) if row is not None else 0
        if guard == 0:
            self._fence_diagnose(conn, 'checkpoint guard lock')
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
        cached_checkpoint=deepcopy(checkpoint)
        def known_blob(key):
            version=checkpoint.get('channel_versions',{}).get(key)
            for identity,entry in getattr(self,'_answer_memory',{}).items():
                prior=entry.checkpoint
                if (identity[:2]==(thread_id,checkpoint_ns) and str(prior.get('channel_versions',{}).get(key))==str(version)
                    and key in prior.get('channel_values',{})):
                    cached_checkpoint['channel_values'][key]=deepcopy(prior['channel_values'][key]);return True
            return False
        cache_safe=all(known_blob(k) for k in blob_values if k not in blob_versions)
        with self._write_lock, self._guarded_conn() as conn:
            try:
                self._guard_lock(conn)
                with conn.cursor() as cur:
                    if blob_versions:
                        cur.executemany(
                            "INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)"
                            " VALUES (%s,%s,%s,%s,%s,%s)"
                            " ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING",
                            [(thread_id, checkpoint_ns, k, str(blob_versions[k]),
                              *self.serde.dumps_typed(blob_values[k])) for k in blob_versions])
                        if cur.rowcount!=len(blob_versions):
                            cache_safe=cache_safe and all(known_blob(k) for k in blob_versions)
                    cur.execute(
                        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id,"
                        " parent_checkpoint_id, checkpoint, metadata) VALUES (%s,%s,%s,%s,%s,%s)"
                        " ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET"
                        " checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata",
                        (thread_id, checkpoint_ns, checkpoint['id'], checkpoint_id,
                         Jsonb(copy), Jsonb(get_serializable_checkpoint_metadata(config, metadata))))
            except BaseException:
                raise
        if cache_safe:self.remember_answer_checkpoint(config,cached_checkpoint,metadata)
        elif hasattr(self,'_answer_memory'):
            # Immutable blob collisions or omitted old blob versions must be
            # reconstructed from durable storage, never from proposed bytes.
            identity=(thread_id,checkpoint_ns)
            self._answer_memory.pop((*identity,checkpoint['id']),None)
            self._answer_latest.pop(identity,None)
        return {'configurable': {'thread_id': thread_id, 'checkpoint_ns': checkpoint_ns,
                                 'checkpoint_id': checkpoint['id']}}

    @_answer_cache_locked
    def put_writes(self, config, writes, task_id, task_path=''):
        # Same contract as the base saver, each write its own guarded insert so
        # a fence loss mid-batch raises before any later write lands.
        # Reserved channels overwrite (library retry semantics); regular task
        # writes are first-writer-wins so a stale batch never replaces a
        # legitimate recorded write. Per-row guarded inserts: correctness over
        # batching at interview-session throughput.
        if hasattr(self,'_answer_memory'):
            cfg=config['configurable']
            self._answer_memory.pop((cfg['thread_id'],cfg.get('checkpoint_ns',''),cfg.get('checkpoint_id')),None)
        reserved = all(channel in WRITES_IDX_MAP for channel, _ in writes)
        sql = (_UPSERT_WRITES if reserved else _INSERT_WRITES).format(lease_table=self._lease_table)
        with self._write_lock, self._guarded_conn() as conn:
            for idx, (channel, value) in enumerate(writes):
                type_, blob = self.serde.dumps_typed(value)
                self._fenced(conn, sql, {
                    'thread_id': config['configurable']['thread_id'],
                    'checkpoint_ns': config['configurable']['checkpoint_ns'],
                    'checkpoint_id': config['configurable']['checkpoint_id'],
                    'task_id': task_id, 'task_path': task_path,
                    'idx': WRITES_IDX_MAP.get(channel, idx),
                    'channel': channel, 'type': type_, 'blob': blob})

def bootstrap_checkpoint_schema(conn) -> None:
    """Explicit one-off schema bootstrap for tests and development superusers.

    Production databases are prepared by the delivery-api migration; this runs
    the pinned library DDL directly and is never called from runtime code.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    PostgresSaver(conn).setup()
