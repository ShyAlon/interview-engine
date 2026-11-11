"""Postgres session registry: placement leases and fenced control-state saves.

The file-backed registry (http-session.json + StopFence) orders writers on one
host only. This store serves the same invariants against the shared registry
table so a session can be claimed, served, stopped and restored on any host.

Schema ownership: the engine_sessions table and engine_session_journal are
created ONLY by the delivery-api migration
20261022090000_session_placement_leases.sql. This module never creates or
alters tables - there is one authoritative schema and one physical registry,
so the publish transaction (session row + journal page + checkpoint pointer)
commits on a single boundary.

Rules:
- Claim-before-serve: one atomic conditional UPDATE wins the lease. The lease
  identity is an EXECUTION INSTANCE (unique per serving process), never a bare
  hostname: two workers on one host cannot take each other's live lease.
  Winning an expired lease bumps lease_epoch. A re-claim by the same live
  instance is idempotent - it extends the expiry without bumping the epoch, so
  in-flight fenced writes of that instance stay valid.
- Fencing: control-state saves and publication-pointer updates commit only
  while the writer's (owner, generation, epoch) still matches and the lease is
  unexpired. A stop or a competing takeover serializes on the same row lock;
  there is no epsilon window between check and write. A failed fence raises
  StopRequested; nothing is silently dropped.
- Every mutation is bound to the owner id. Cross-owner claims, saves and stops
  match no row: a foreign tenant gets the same KeyError/claim-loss as a
  missing session. The HTTP boundary still performs the real authorization;
  this is defense in depth, not the check itself.
- Stop bumps generation and expires the lease WITHOUT holding the execution
  lease, and blocks every later fenced mutation.
- Restore reads only the staged publication pointer and control state, never
  saver-latest: a crashed writer can leave unpublished checkpoint rows, and
  they stay invisible until a fenced save publishes them.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from contextvars import ContextVar
from contextlib import contextmanager

_answer_publication=ContextVar('answer_publication',default=None)

@contextmanager
def answer_publication(operation):
    token=_answer_publication.set(operation)
    try:yield
    finally:_answer_publication.reset(token)


import psycopg

from .stop_fence import StopRequested

log = logging.getLogger("interview_engine.pg_session_store")


class PublicationConflict(Exception):
    """A replay or delayed write conflicts with a committed publication."""


@dataclass(frozen=True)
class SessionClaim:
    """A won lease: the fence identity for every mutation of this session."""
    session_id: str
    owner_id: str
    generation: int
    epoch: int
    instance: str


class PgSessionStore:
    """Registry operations against the migration-owned engine_sessions table."""

    def __init__(self, pool, *, claims: tuple[str, str] | None = None, close_pool: bool = False):
        # pool: the process connection pool (pg_pool). claims = (owner_id,
        # session_id) applied per transaction on every checkout (pg_claims);
        # None only in mirrored-DDL superuser tests. close_pool: True only when
        # this store built the pool (standalone factory); an injected shared
        # pool is closed by its owner.
        self.pool = pool
        self.claims = claims if claims is not None else getattr(pool, 'claims', None)
        self._close_pool = close_pool

    def _borrow(self):
        # One borrowed connection per operation; never returned mid-transaction.
        from .pg_pool import borrow_claimed
        return borrow_claimed(self.pool, self.claims)

    def heartbeat_many(self, claims: list, ttl: float) -> list:
        """One tick for every held lease on ONE borrowed connection.

        Returns the claims whose beat matched no row (lease lost: expired,
        stopped or taken over). The lease expiry checked by every fenced write
        is the authority; this only extends live leases and reports loss.
        """
        lost = []
        with self._borrow() as conn:
            for claim in claims:
                now = time.time()
                row = conn.execute(
                    "UPDATE engine_sessions SET lease_expires_at = %(expiry)s, updated_at = %(now)s"
                    " WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid AND generation = %(gen)s"
                    "   AND lease_epoch = %(epoch)s AND placement_instance = %(instance)s"
                    "   AND lease_expires_at > %(now)s"
                    " RETURNING 1",
                    {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
                     "epoch": claim.epoch, "instance": claim.instance,
                     "expiry": now + ttl, "now": now}).fetchone()
                if row is None:
                    lost.append(claim)
            conn.commit()
        return lost

    def close(self) -> None:
        if self._close_pool:
            self.pool.close()

    def create(self, session_id: str, task_id: str, owner_id: str) -> None:
        with self._borrow() as conn:
            now = time.time()
            try:
                row = conn.execute(
                    "INSERT INTO engine_sessions(session_id,owner_id,task_id,generation,created_at,lease_epoch,lease_expires_at)"
                    " VALUES (%s,%s,%s,1,%s,0,0)"
                    " ON CONFLICT (session_id) DO NOTHING"
                    " RETURNING session_id",
                    (session_id, owner_id, task_id, now)).fetchone()
                if row is None:
                    # The delivery app pre-binds its session row at Start
                    # (begin_interview_start -> set_engine_session). Adopt exactly
                    # that row; any other existing row is the real conflict.
                    existing = conn.execute(
                        "SELECT owner_id::text AS owner_id, task_id FROM engine_sessions WHERE session_id = %s",
                        (session_id,)).fetchone()
                    if existing is None or existing["owner_id"] != owner_id or existing["task_id"] != task_id:
                        conn.rollback()
                        raise psycopg.errors.UniqueViolation("engine_sessions session_id conflict")
                conn.commit()
            except Exception as exc:
                # Never leave the shared serving connection in an aborted
                # transaction: one failed op must not poison every later claim.
                conn.rollback()
                from .pg_diagnostics import tag_stage
                tag_stage(exc, "session insert", conn)
                raise

    def claim(self, session_id: str, owner_id: str, instance: str, ttl: float) -> SessionClaim | None:
        with self._borrow() as conn:
            """Win the lease atomically, or return None while another instance holds it.

            An expired lease is taken with an epoch bump. A re-claim by the same
            live instance extends the expiry without a bump, so its own in-flight
            fenced writes are not invalidated by reentrant requests.
            """
            now = time.time()
            try:
                row = conn.execute(
                    "UPDATE engine_sessions"
                    " SET lease_epoch = CASE WHEN lease_expires_at <= %(now)s THEN lease_epoch + 1 ELSE lease_epoch END,"
                    "     placement_instance = %(instance)s, lease_expires_at = %(expiry)s, updated_at = %(now)s"
                    " WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid"
                    "   AND (lease_expires_at <= %(now)s OR placement_instance = %(instance)s)"
                    "   AND COALESCE(control->>'stopped','false') != 'true'"
                    " RETURNING generation, lease_epoch",
                    {"sid": session_id, "owner": owner_id, "instance": instance,
                     "expiry": now + ttl, "now": now}).fetchone()
                conn.commit()
            except Exception as exc:
                # Never leave the shared serving connection aborted.
                conn.rollback()
                from .pg_diagnostics import tag_stage
                tag_stage(exc, "claim setup", conn)
                raise
            if row is None:
                return None
            return SessionClaim(session_id, owner_id, row["generation"], row["lease_epoch"], instance)

    def claim_generator_expected(self, binding, instance: str, ttl: float):
        """Transfer an unleased session to its generator without restoring a graph."""
        with self._borrow() as conn, conn.transaction():
            row=conn.execute("SELECT task_id,generation,lease_epoch,lease_expires_at,control FROM engine_sessions WHERE session_id=%s AND owner_id=%s::uuid FOR UPDATE",
                (binding.session_id,str(binding.owner_id))).fetchone()
            now=time.time()
            if (row is None or row['task_id']!=binding.task_id or row['generation']!=binding.generation
                or row['lease_epoch']!=binding.lease_epoch or row['lease_expires_at']>now or (row['control'] or {}).get('stopped')):
                raise PermissionError('generator execution binding changed, stopped or still leased')
            epoch=row['lease_epoch']+1
            conn.execute("UPDATE engine_sessions SET lease_epoch=%s,placement_instance=%s,lease_expires_at=%s,updated_at=%s WHERE session_id=%s AND owner_id=%s::uuid",
                (epoch,instance,now+ttl,now,binding.session_id,str(binding.owner_id)))
            return SessionClaim(binding.session_id,str(binding.owner_id),binding.generation,epoch,instance)

    def claim_start_expected(self, binding, instance: str, ttl: float):
        """Claim only the parent's empty durable bootstrap row, never overwrite."""
        with self._borrow() as conn, conn.transaction():
            row=conn.execute("SELECT task_id,generation,lease_epoch,lease_expires_at,control FROM engine_sessions WHERE session_id=%s AND owner_id=%s::uuid FOR UPDATE",
                (binding.session_id,str(binding.owner_id))).fetchone()
            now=time.time()
            if (row is None or row['task_id']!=binding.task_id or row['generation']!=binding.generation
                or row['lease_epoch']!=binding.lease_epoch or row['lease_expires_at']>now or row['control']):
                raise PermissionError('child bootstrap changed, published or still leased')
            epoch=row['lease_epoch']+1
            conn.execute("UPDATE engine_sessions SET lease_epoch=%s,placement_instance=%s,lease_expires_at=%s,updated_at=%s WHERE session_id=%s AND owner_id=%s::uuid",
                (epoch,instance,now+ttl,now,binding.session_id,str(binding.owner_id)))
            return SessionClaim(binding.session_id,str(binding.owner_id),binding.generation,epoch,instance)

    def claim_published_expected(self, binding, operation, instance: str, ttl: float):
        """Compare immutable published evidence and acquire a NEW lease atomically.

        An envelope epoch names the pre-claim epoch. A successful new child
        takes epoch+1. No parent live-lease reuse or saver-latest fallback.
        """
        from .child_step import published_digest
        from requi_child_contract import InterviewAnswer,InterviewRoute
        if _answer_publication.get() is not None and isinstance(operation,InterviewAnswer):
            return self._claim_answer_expected(binding,operation,instance,ttl)
        with self._borrow() as conn, conn.transaction():
            row=conn.execute(
                "SELECT task_id,generation,lease_epoch,lease_expires_at,control,"
                "published_thread_id,published_checkpoint_ns,published_checkpoint_id "
                "FROM engine_sessions WHERE session_id=%s AND owner_id=%s::uuid FOR UPDATE",
                (binding.session_id,str(binding.owner_id))).fetchone()
            now=time.time()
            if (row is None or row['task_id']!=binding.task_id or row['generation']!=binding.generation
                or row['lease_epoch']!=binding.lease_epoch or row['lease_expires_at']>now):
                raise PermissionError('published child lease binding changed or still held')
            control=row.get('control') or {}
            if control.get('stopped') or control.get('sequence')!=operation.expected_sequence:
                raise PermissionError('published child control stopped or sequence changed')
            if published_digest(row)!=operation.checkpoint_sha256:
                raise PermissionError('published child checkpoint evidence changed')
            if isinstance(operation,(InterviewAnswer,InterviewRoute)):
                pending=control.get('open_input') or {}
                if str(pending.get('input_id'))!=operation.wait_id:
                    raise PermissionError('published child wait changed')
                context=pending.get('context') or {}
                revision=context.get('evidence_revision') or 'engine:'+binding.session_id+':'+str(control['sequence'])
                if isinstance(operation,InterviewAnswer) and revision!=operation.review_revision:
                    raise PermissionError('published child review changed')
            epoch=row['lease_epoch']+1
            conn.execute(
                "UPDATE engine_sessions SET lease_epoch=%s,placement_instance=%s,lease_expires_at=%s,updated_at=%s "
                "WHERE session_id=%s AND owner_id=%s::uuid",
                (epoch,instance,now+ttl,now,binding.session_id,str(binding.owner_id)))
            claim=SessionClaim(binding.session_id,str(binding.owner_id),binding.generation,epoch,instance)
            return claim,dict(row)

    def _claim_answer_expected(self,binding,operation,instance,ttl):
        """Cold answer claim and restore row in one conditional statement.

        Input/sequence and generation/epoch replace repeated exact-byte digest
        checks. Schema readiness rides this startup read, not another trip.
        """
        from .pg_checkpoint import FencedPostgresSaver
        now=time.time()
        binding_projection=""
        if self.claims is not None:
            binding_projection=", (SELECT jsonb_agg(jsonb_build_object('payload',encode(payload,'hex'),'sha256',sha256,'ordinal',ordinal,'total',total) ORDER BY ordinal) FROM public.requi_document_parts(%(sid)s,%(gen)s,'evidence/schema-binding.json')) AS schema_binding_parts"
        with self._borrow() as conn,conn.transaction():
            row=conn.execute("""WITH won AS (UPDATE engine_sessions SET lease_epoch=lease_epoch+1,
                placement_instance=%(instance)s,lease_expires_at=%(expiry)s,updated_at=%(now)s
                WHERE session_id=%(sid)s AND owner_id=%(owner)s::uuid AND task_id=%(task)s
                  AND generation=%(gen)s AND lease_epoch=%(epoch)s AND lease_expires_at<=%(now)s
                  AND NOT coalesce((control->>'stopped')::boolean,false)
                  AND (control->>'sequence')::bigint=%(sequence)s
                  AND control->'open_input'->>'input_id'=%(wait)s
                  AND coalesce(nullif(control->'open_input'->'context'->>'evidence_revision',''),
                    'engine:'||session_id||':'||(control->>'sequence'))=%(revision)s
                RETURNING task_id,generation,lease_epoch,lease_expires_at,control,
                    published_thread_id,published_checkpoint_ns,published_checkpoint_id,
                    (SELECT max(v) FROM checkpoint_migrations) AS checkpoint_schema_version)
                SELECT won.*,cp.*"""+binding_projection+""" FROM won LEFT JOIN LATERAL ("""+FencedPostgresSaver.SELECT_SQL+"""
                    WHERE thread_id=won.published_thread_id AND checkpoint_ns=won.published_checkpoint_ns
                      AND checkpoint_id=won.published_checkpoint_id) cp ON true""",
                {'sid':binding.session_id,'owner':str(binding.owner_id),'task':binding.task_id,
                 'gen':binding.generation,'epoch':binding.lease_epoch,'now':now,'expiry':now+ttl,
                 'instance':instance,'sequence':operation.expected_sequence,'wait':operation.wait_id,
                 'revision':operation.review_revision}).fetchone()
            if row is None:raise PermissionError('answer claim input/generation/lease changed or still held')
            version=row['checkpoint_schema_version']
            if version is None or version<len(FencedPostgresSaver.MIGRATIONS)-1:
                raise RuntimeError('checkpoint schema is not prepared: apply the delivery-api checkpoint migration')
            self._checkpoint_schema_version=version
            self._answer_schema_binding=(binding.session_id,str(binding.owner_id),self.claims,binding.generation,row['lease_epoch'],instance,row.get('schema_binding_parts'))
            # Keep only the exact published pointer returned by the same atomic
            # claim. A v3 legacy pending-send migration still uses the saver path.
            self._answer_checkpoint_row=dict(row) if row.get('checkpoint') and row['checkpoint'].get('v',0)>=4 else None
            claim=SessionClaim(binding.session_id,str(binding.owner_id),binding.generation,row['lease_epoch'],instance)
            return claim,dict(row)

    def release(self, claim: SessionClaim) -> bool:
        with self._borrow() as conn:
            """Voluntarily end a held lease at a committed boundary so another
            instance can claim immediately. Identity, generation and epoch must all
            match, so a stale owner can never release a lease it already lost."""
            now = time.time()
            try:
                row = conn.execute(
                    "UPDATE engine_sessions SET lease_expires_at = %(now)s, updated_at = %(now)s"
                    " WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid AND generation = %(gen)s"
                    "   AND lease_epoch = %(epoch)s AND placement_instance = %(instance)s"
                    " RETURNING 1",
                    {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
                     "epoch": claim.epoch, "instance": claim.instance, "now": now}).fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return row is not None

    def heartbeat(self, claim: SessionClaim, ttl: float) -> bool:
        with self._borrow() as conn:
            """Extend a held lease (never bumps the epoch). False means it was lost."""
            now = time.time()
            try:
                row = conn.execute(
                    "UPDATE engine_sessions SET lease_expires_at = %(expiry)s, updated_at = %(now)s"
                    " WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid AND generation = %(gen)s"
                    "   AND lease_epoch = %(epoch)s AND placement_instance = %(instance)s"
                    "   AND lease_expires_at > %(now)s"
                    " RETURNING 1",
                    {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
                     "epoch": claim.epoch, "instance": claim.instance,
                     "expiry": now + ttl, "now": now}).fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return row is not None

    def _fence_diagnose(self, claim: SessionClaim, what: str) -> None:
        with self._borrow() as conn:
            """Best-effort read of the current lease row after a fence rejected a
            write: logs which predicate failed (owner / generation / epoch /
            expiry) so a swallowed StopRequested is diagnosable from logs alone.
            Read-only; a failed diagnostic never masks the original rejection."""
            try:
                row = conn.execute(
                    "SELECT generation, lease_epoch, lease_expires_at, placement_instance"
                    " FROM engine_sessions WHERE session_id = %(sid)s",
                    {"sid": claim.session_id}).fetchone()
                if row is None:
                    log.warning("lease.fence.rejected what=%s session=%s reason=session-row-missing expected gen=%s epoch=%s",
                                what, claim.session_id, claim.generation, claim.epoch)
                    return
                now = time.time()
                reasons = []
                if row["generation"] != claim.generation:
                    reasons.append("generation %s != %s" % (row["generation"], claim.generation))
                if row["lease_epoch"] != claim.epoch:
                    reasons.append("lease_epoch %s != %s" % (row["lease_epoch"], claim.epoch))
                if not (row["lease_expires_at"] or 0) > now:
                    reasons.append("lease expired %.1fs ago (ttl lapsed)" % (now - (row["lease_expires_at"] or 0)))
                log.warning("lease.fence.rejected what=%s session=%s reason=%s expected gen=%s epoch=%s instance=%s",
                            what, claim.session_id, "; ".join(reasons) or "no predicate mismatch (owner or race)",
                            claim.generation, claim.epoch, row["placement_instance"])
            except Exception as diag_exc:
                log.warning("lease.fence.rejected what=%s session=%s diagnostic-read-failed %s: %s",
                            what, claim.session_id, type(diag_exc).__name__, str(diag_exc)[:200])

    def fenced_save(self, claim: SessionClaim, control: dict, *,
                    published_thread_id: str | None = None,
                    published_checkpoint_ns: str = "",
                    published_checkpoint_id: str | None = None) -> None:
        with self._borrow() as conn:
            """Commit control state and the staged publication pointer under the fence."""
            try:
                row = conn.execute(
                    "UPDATE engine_sessions"
                    " SET control = %(control)s::jsonb, published_thread_id = %(tid)s,"
                    "     published_checkpoint_ns = %(ns)s, published_checkpoint_id = %(cid)s,"
                    "     updated_at = %(now)s"
                    " WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid AND generation = %(gen)s"
                    "   AND lease_epoch = %(epoch)s AND lease_expires_at > %(now)s"
                    " RETURNING 1",
                    {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
                     "epoch": claim.epoch, "control": json.dumps(control), "tid": published_thread_id,
                     "ns": published_checkpoint_ns, "cid": published_checkpoint_id,
                     "now": time.time()}).fetchone()
            except Exception as exc:
                conn.rollback()
                from .pg_diagnostics import tag_stage
                tag_stage(exc, "checkpoint save", conn)
                raise
            if row is None:
                conn.rollback()
                self._fence_diagnose(claim, "save")
                raise StopRequested("session lease fence rejected the save")
            conn.commit()

    def stop(self, session_id: str, owner_id: str, control: dict | None = None) -> int:
        with self._borrow() as conn:
            """Owner-authorized atomic Stop: stopped control AND generation bump in
            one transaction. No execution lease, no wait on any worker: a live
            foreign lease, a blocked advance or a failed earlier publication cannot
            prevent it. The row lock serializes the bump against any in-flight
            worker publish - whichever commits second wins the row, and after this
            transaction every fenced write from the old generation fails.

            The authorization decision belongs to the HTTP boundary; binding the
            bump to the owner id keeps a foreign tenant from stopping (or even
            detecting) this session through the store. When the caller serves the
            session locally it passes the quiesced control (open input cleared,
            newer sequence); otherwise the staged control is kept and only marked.
            """
            row = conn.execute(
                "UPDATE engine_sessions SET generation = CASE WHEN control->>'stopped'='true' THEN generation ELSE generation + 1 END, lease_expires_at = 0,"
                " control = CASE WHEN control->>'stopped'='true' THEN control ELSE COALESCE(%(control)s::jsonb, control, '{}'::jsonb) || %(stopped_marker)s::jsonb END,"
                " updated_at = %(now)s WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid"
                " RETURNING generation",
                {"sid": session_id, "owner": owner_id,
                 "control": json.dumps(control) if control is not None else None,
                 "stopped_marker": '{"stopped": true}',
                 "now": time.time()}).fetchone()
            conn.commit()
            if row is None:
                raise KeyError(session_id)
            return row["generation"]

    def acknowledge_stop_quiescent(self,claim:SessionClaim,stopped_generation:int)->bool:
        """Only the actual stopped writer can prove its exit, never a takeover.

        The registry calls this after its last writer exits. Expiry/restart is
        not proof. Owner, stopped generation, old lease epoch and instance all
        match in one update. A foreign/stale registry cannot acknowledge it.
        """
        if stopped_generation != claim.generation+1:return False
        with self._borrow() as conn:
            row=conn.execute(
                "UPDATE engine_sessions SET control=control || '{\"stop_quiescent\":true}'::jsonb, updated_at=%s "
                "WHERE session_id=%s AND owner_id=%s::uuid AND generation=%s "
                "AND lease_epoch=%s AND placement_instance=%s AND control->>'stopped'='true' RETURNING generation",
                (time.time(),claim.session_id,claim.owner_id,stopped_generation,claim.epoch,claim.instance)).fetchone()
            conn.commit()
            return row is not None

    def load_published(self, session_id: str, owner_id: str) -> dict | None:
        with self._borrow() as conn:
            """Restore from the staged publication only - never saver-latest rows."""
            row = conn.execute(
                "SELECT task_id, generation, control, published_thread_id,"
                "       published_checkpoint_ns, published_checkpoint_id"
                " FROM engine_sessions WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid",
                {"sid": session_id, "owner": owner_id}).fetchone()
            if row is None:
                return None
            return dict(row)

    def staged_log(self, session_id: str, generation: int, *, execution_point: str,
                   node: str, stage: str, item: str = '') -> list[dict]:
        """The ordered staged rows for one logical call sequence: a (stage,item)
        sequence issued by one node under one execution point (the durable
        checkpoint the execution resumed from) in one generation."""
        with self._borrow() as conn, conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ordinal,attempt_id,request_sha256,response,usage,provider,model FROM staged_provider_responses "
                    "WHERE session_id=%s AND generation=%s AND execution_point=%s AND node=%s AND stage=%s AND item=%s ORDER BY ordinal",
                    (session_id, generation, execution_point, node, stage, item))
                return cur.fetchall()

    def record_dispatch(self, claim: SessionClaim, *, execution_point: str, node: str,
                        stage: str, item: str, ordinal: int,
                        request_sha256: str, provider: str, model: str) -> None:
        """Open a logical call's staged row BEFORE the ledger attempt is reserved:
        any crash from here on leaves visible evidence instead of an invisible rerun."""
        import time as _time
        with self._borrow() as conn, conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO staged_provider_responses(owner_id,session_id,generation,execution_point,node,stage,item,ordinal,request_sha256,provider,model,dispatched_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (claim.owner_id, claim.session_id, claim.generation, execution_point, node,
                     stage, item, ordinal, request_sha256, provider, model, _time.time()))

    def bind_attempt(self, claim: SessionClaim, *, execution_point: str, node: str,
                     stage: str, item: str, ordinal: int, attempt_id: str) -> None:
        """Bind the ledger attempt to the staged row (once)."""
        with self._borrow() as conn, conn.transaction():
            with conn.cursor() as cur:
                row = cur.execute(
                    "UPDATE staged_provider_responses SET attempt_id=%s WHERE session_id=%s AND generation=%s AND execution_point=%s AND node=%s AND stage=%s AND item=%s AND ordinal=%s AND attempt_id IS NULL RETURNING ordinal",
                    (attempt_id, claim.session_id, claim.generation, execution_point, node, stage, item, ordinal)).fetchone()
        if row is None:
            raise PublicationConflict('staged response attempt binding raced or repeated')

    def complete_staged(self, claim: SessionClaim, *, execution_point: str, node: str,
                        stage: str, item: str, ordinal: int,
                        response: dict, usage: dict | None) -> None:
        """Stage the raw provider result + measured usage (once) BEFORE settle/handoff."""
        import json as _json, time as _time
        with self._borrow() as conn, conn.transaction():
            with conn.cursor() as cur:
                row = cur.execute(
                    "UPDATE staged_provider_responses SET response=%s,usage=%s,completed_at=%s "
                    "WHERE session_id=%s AND generation=%s AND execution_point=%s AND node=%s AND stage=%s AND item=%s AND ordinal=%s AND response IS NULL RETURNING ordinal",
                    (_json.dumps(response), _json.dumps(usage) if usage is not None else None, _time.time(),
                     claim.session_id, claim.generation, execution_point, node, stage, item, ordinal)).fetchone()
        if row is None:
            raise PublicationConflict('staged response already completed or missing')

    def append_journal(self, claim: SessionClaim, cursor: int, page: dict) -> None:
        with self._borrow() as conn:
            """Append one journal page under the fence.

            Pages are immutable evidence keyed by (session, cursor): a duplicate
            cursor is a no-op, never an overwrite. Part of the same staged
            publication discipline as control state - a stop or takeover that
            landed rejects the append with StopRequested.
            """
            try:
                guard=conn.execute("SELECT session_id FROM engine_sessions WHERE session_id=%s AND owner_id=%s::uuid"
                    " AND generation=%s AND lease_epoch=%s AND lease_expires_at > extract(epoch from clock_timestamp()) FOR UPDATE",
                    (claim.session_id,claim.owner_id,claim.generation,claim.epoch)).fetchone()
                if guard is None:
                    raise StopRequested('session lease fence rejected the journal append')
                self._require_journal_mode(conn,claim.session_id,'buffered')
                if page.get('journal_mode','buffered') != 'buffered':
                    raise PublicationConflict('append_journal requires buffered pages')
                row = conn.execute(
                    "INSERT INTO engine_session_journal(session_id,cursor,page,created_at)"
                    " SELECT %(sid)s, %(cursor)s, %(page)s::jsonb, %(now)s FROM engine_sessions"
                    " WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid AND generation = %(gen)s"
                    "   AND lease_epoch = %(epoch)s AND lease_expires_at > %(now)s"
                    " FOR UPDATE"
                    " ON CONFLICT (session_id, cursor) DO NOTHING"
                    " RETURNING 1",
                    {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
                     "epoch": claim.epoch, "cursor": cursor, "page": json.dumps(page),
                     "now": time.time()}).fetchone()
                # A duplicate cursor returns no row without the guard failing; detect
                # fence failure separately so a stale writer never passes silently.
                if row is None:
                    guard = conn.execute(
                        "SELECT 1 FROM engine_sessions WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid"
                        " AND generation = %(gen)s AND lease_epoch = %(epoch)s AND lease_expires_at > %(now)s",
                        {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
                         "epoch": claim.epoch, "now": time.time()}).fetchone()
                    if guard is not None:
                        conn.commit()
                        return  # page already staged at this cursor; immutable, kept
                    conn.rollback()
                    self._fence_diagnose(claim, "journal append")
                    raise StopRequested("session lease fence rejected the journal append")
                conn.commit()
            except Exception as exc:
                conn.rollback()
                from .pg_diagnostics import tag_stage
                tag_stage(exc, "journal append", conn)
                raise

    @staticmethod
    def _require_journal_mode(conn, session_id, mode):
        # Caller holds the session row lock. Legacy rows are buffered.
        row=conn.execute("SELECT page FROM engine_session_journal WHERE session_id=%s ORDER BY cursor LIMIT 1",
                         (session_id,)).fetchone()
        if row is not None and row['page'].get('journal_mode','buffered') != mode:
            raise PublicationConflict('cannot mix observational and buffered progress for one session')

    def append_observational_progress(self, claim: SessionClaim, page: dict) -> int:
        """Allocate + append live progress in one short fenced transaction.

        No control/checkpoint writes. A crashed step may leave observational
        signals, which never claim to be committed step evidence.
        """
        from vmodel_contracts.progress_signal import ProgressSignal
        signal=ProgressSignal.model_validate({k:v for k,v in page.items() if k != "journal_mode"})  # validate before checkout
        if signal.session_id != claim.session_id or page.get('journal_mode') != 'observational':
            raise ValueError('observational progress scope/mode mismatch')
        params={"sid":claim.session_id,"owner":claim.owner_id,"gen":claim.generation,
                "epoch":claim.epoch}
        with self._borrow() as conn:
            try:
                row=conn.execute("SELECT task_id FROM engine_sessions WHERE session_id=%(sid)s"
                    " AND owner_id=%(owner)s::uuid AND generation=%(gen)s AND lease_epoch=%(epoch)s"
                    " AND lease_expires_at > extract(epoch from clock_timestamp()) FOR UPDATE",params).fetchone()
                if row is None:
                    raise StopRequested('session lease fence rejected observational progress')
                if row['task_id'] != signal.task_id:
                    raise ValueError('observational progress task mismatch')
                self._require_journal_mode(conn,claim.session_id,'observational')
                # The row lock serializes all cursor writers; expiry is checked
                # again after waiting for the lock, at insertion time.
                payload=dict(page)
                inserted=conn.execute("INSERT INTO engine_session_journal(session_id,cursor,page,created_at)"
                    " SELECT %(sid)s,COALESCE(MAX(cursor),0)+1,"
                    " jsonb_set(%(page)s::jsonb,'{cursor}',to_jsonb(COALESCE(MAX(cursor),0)+1)),"
                    " extract(epoch from clock_timestamp()) FROM engine_session_journal WHERE session_id=%(sid)s"
                    " HAVING EXISTS(SELECT 1 FROM engine_sessions WHERE session_id=%(sid)s"
                    " AND lease_expires_at > extract(epoch from clock_timestamp())) RETURNING cursor",
                    {**params,'page':json.dumps(payload)}).fetchone()
                if inserted is None:
                    raise StopRequested('session lease expired while allocating progress cursor')
                conn.commit()
                return inserted['cursor']
            except BaseException:
                conn.rollback()
                raise

    def progress_page(self, session_id: str, owner_id: str, after: int, limit: int):
        """Publication row + committed journal pages in ONE borrow.

        The woken progress drain is the per-turn hot read: separate
        load_published + read_journal each paid claims setup and a round trip
        (2026-11-09 turn evidence: ~2.8s per fetch at remote-pooler latency).
        Same rows and owner scoping, half the fixed cost.
        """
        with self._borrow() as conn:
            published = conn.execute(
                "SELECT task_id, generation, control, published_thread_id,"
                "       published_checkpoint_ns, published_checkpoint_id"
                " FROM engine_sessions WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid",
                {"sid": session_id, "owner": owner_id}).fetchone()
            rows = conn.execute(
                "SELECT j.cursor, j.page FROM engine_session_journal j"
                " JOIN engine_sessions s ON s.session_id = j.session_id"
                " WHERE j.session_id = %(sid)s AND s.owner_id = %(owner)s::uuid AND j.cursor > %(after)s"
                " ORDER BY j.cursor LIMIT %(limit)s",
                {"sid": session_id, "owner": owner_id, "after": after, "limit": limit}).fetchall()
            return (dict(published) if published is not None else None,
                    [{"cursor": r["cursor"], "page": r["page"]} for r in rows])

    def read_journal(self, session_id: str, owner_id: str, after: int, limit: int) -> list[dict]:
        with self._borrow() as conn:
            """Committed journal pages for progress reads; cross-owner reads see nothing."""
            rows = conn.execute(
                "SELECT j.cursor, j.page FROM engine_session_journal j"
                " JOIN engine_sessions s ON s.session_id = j.session_id"
                " WHERE j.session_id = %(sid)s AND s.owner_id = %(owner)s::uuid AND j.cursor > %(after)s"
                " ORDER BY j.cursor LIMIT %(limit)s",
                {"sid": session_id, "owner": owner_id, "after": after, "limit": limit}).fetchall()
            return [{"cursor": r["cursor"], "page": r["page"]} for r in rows]

    def publish_step(self, claim: SessionClaim, control: dict, *,
                     journal_cursor: int | None = None,
                     journal_page: dict | None = None,
                     journal_pages: list[tuple[int, dict]] | None = None,
                     published_thread_id: str | None = None,
                     published_checkpoint_ns: str = "",
                     published_checkpoint_id: str | None = None,
                     fault_hook=None) -> None:
        operation=_answer_publication.get()
        if operation is not None and operation.operation.kind=='interview.answer' and not journal_pages and journal_page is None and fault_hook is None:
            return self._publish_answer(claim,operation,control,published_thread_id,published_checkpoint_ns,published_checkpoint_id)
        with self._borrow() as conn:
            """One transaction per published step: journal page, control state and
            the checkpoint pointer commit together under the lease fence.

            Ordering within one lease (epoch/generation fences do not order valid
            writes): the control payload's integer "sequence" is the step's
            revision. The lease row is locked FOR UPDATE first, so the revision
            check below is a real compare-and-swap:
            - a NEWER sequence commits;
            - the SAME sequence commits only as an exact replay (control, pointer
              and journal page all identical to what is committed) and is a no-op;
            - anything older, or a same-cursor journal page with different bytes,
              is a conflicting replay: rollback and PublicationConflict.
            Readers only ever see committed steps: a crash or failure leaves NONE
            of the three behind (fault_hook exists only to inject a mid-transaction
            failure in tests).
            """
            if "sequence" not in control or not isinstance(control["sequence"], int):
                raise ValueError('control payload must carry an integer "sequence" revision')
            params = {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
                      "epoch": claim.epoch, "now": time.time()}
            try:
                row = conn.execute(
                    "SELECT control, published_thread_id, published_checkpoint_ns, published_checkpoint_id"
                    " FROM engine_sessions WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid"
                    " AND generation = %(gen)s AND lease_epoch = %(epoch)s AND lease_expires_at > %(now)s"
                    " FOR UPDATE",
                    params).fetchone()
            except Exception as exc:
                conn.rollback()
                from .pg_diagnostics import tag_stage
                tag_stage(exc, "publication guard", conn)
                raise
            if row is None:
                conn.rollback()
                self._fence_diagnose(claim, "publication")
                raise StopRequested("session lease fence rejected the publication")
            prior = row["control"] or {}
            prior_seq = prior.get("sequence", 0)
            new_seq = control["sequence"]
            if new_seq <= prior_seq:
                exact = (prior == control
                         and row["published_thread_id"] == published_thread_id
                         and row["published_checkpoint_ns"] == published_checkpoint_ns
                         and row["published_checkpoint_id"] == published_checkpoint_id)
                if exact:
                    replay_pages = list(journal_pages or [])
                    if journal_page is not None:
                        replay_pages.append((journal_cursor, journal_page))
                    for cursor, page in replay_pages:
                        existing = conn.execute(
                            "SELECT page FROM engine_session_journal WHERE session_id = %(sid)s AND cursor = %(cursor)s",
                            {**params, "cursor": cursor}).fetchone()
                        exact = existing is not None and existing["page"] == page
                        if not exact:
                            break
                if exact:
                    conn.commit()   # exact replay of a committed step: no-op
                    return
                conn.rollback()
                raise PublicationConflict(
                    f"stale or conflicting publication for {claim.session_id}: "
                    f"sequence {new_seq} <= committed {prior_seq}")
            # All-or-nothing: any failure after the guard lock rolls the whole
            # step back - journal pages, control and pointer move together or not
            # at all, and the connection is never left in an aborted transaction.
            try:
                pages = list(journal_pages or [])
                if journal_page is not None:
                    if journal_cursor is None:
                        raise ValueError("journal page requires its cursor")
                    pages.append((journal_cursor, journal_page))
                if pages:
                    self._require_journal_mode(conn,claim.session_id,'buffered')
                    if any(page.get('journal_mode','buffered') != 'buffered' for _,page in pages):
                        raise PublicationConflict('step publication requires buffered journal pages')
                for cursor, page in pages:
                    inserted = conn.execute(
                        "INSERT INTO engine_session_journal(session_id,cursor,page,created_at)"
                        " VALUES (%(sid)s,%(cursor)s,%(page)s::jsonb,%(now)s)"
                        " ON CONFLICT (session_id, cursor) DO NOTHING RETURNING 1",
                        {**params, "cursor": cursor, "page": json.dumps(page)}).fetchone()
                    if inserted is None:
                        existing = conn.execute(
                            "SELECT page FROM engine_session_journal WHERE session_id = %(sid)s AND cursor = %(cursor)s",
                            {**params, "cursor": cursor}).fetchone()
                        if existing is None or existing["page"] != page:
                            raise PublicationConflict(
                                f"journal cursor {cursor} for {claim.session_id} already holds a different page")
                        raise PublicationConflict(
                            f"journal cursor {cursor} for {claim.session_id} belongs to an earlier step")
                if fault_hook is not None:
                    fault_hook()   # test-only: inject a failure between journal insert and pointer commit
                conn.execute(
                    "UPDATE engine_sessions"
                    " SET control = %(control)s::jsonb, published_thread_id = %(tid)s,"
                    "     published_checkpoint_ns = %(ns)s, published_checkpoint_id = %(cid)s,"
                    "     updated_at = %(now)s"
                    " WHERE session_id = %(sid)s",
                    {**params, "control": json.dumps(control), "tid": published_thread_id,
                     "ns": published_checkpoint_ns, "cid": published_checkpoint_id})
            except BaseException as exc:
                conn.rollback()
                from .pg_diagnostics import tag_stage
                tag_stage(exc, "step publication", conn)
                raise
            conn.commit()

    def _publish_answer(self,claim,operation,control,thread,namespace,checkpoint):
        """One conditional answer publication; no intermediate exact-byte replay."""
        body=operation.operation;binding=operation.binding
        if (claim.session_id!=binding.session_id or claim.owner_id!=str(binding.owner_id)
            or claim.generation!=binding.generation or type(control.get('sequence')) is not int
            or control['sequence']<=body.expected_sequence):
            raise PublicationConflict('answer publication identity or sequence changed')
        payload={**control,'_answer_operation_id':operation.operation_id}
        params={'sid':claim.session_id,'owner':claim.owner_id,'task':binding.task_id,
            'gen':claim.generation,'epoch':claim.epoch,'now':time.time(),'operation':operation.operation_id,
            'expected':body.expected_sequence,'sequence':control['sequence'],'wait':body.wait_id,
            'control':json.dumps(payload),'thread':thread,'ns':namespace,'checkpoint':checkpoint}
        from .pg_checkpoint import _answer_checkpoint_publication
        batch=_answer_checkpoint_publication.get()
        if batch is not None:
            for saver in batch.savers:
                lease=saver._lease
                if (lease.session_id,lease.generation,lease.epoch)!=(claim.session_id,claim.generation,claim.epoch):
                    raise StopRequested('staged checkpoint lease differs from final publication')
        with self._borrow() as conn:
            with conn.transaction():
                if batch is not None and not batch.committed:batch.flush(conn)
                row=conn.execute("""UPDATE engine_sessions SET
                    control=CASE WHEN control->>'_answer_operation_id'=%(operation)s THEN control ELSE %(control)s::jsonb END,
                    published_thread_id=CASE WHEN control->>'_answer_operation_id'=%(operation)s THEN published_thread_id ELSE %(thread)s END,
                    published_checkpoint_ns=CASE WHEN control->>'_answer_operation_id'=%(operation)s THEN published_checkpoint_ns ELSE %(ns)s END,
                    published_checkpoint_id=CASE WHEN control->>'_answer_operation_id'=%(operation)s THEN published_checkpoint_id ELSE %(checkpoint)s END,
                    updated_at=%(now)s
                    WHERE session_id=%(sid)s AND owner_id=%(owner)s::uuid AND task_id=%(task)s
                      AND generation=%(gen)s AND lease_epoch=%(epoch)s AND lease_expires_at>%(now)s
                      AND NOT coalesce((control->>'stopped')::boolean,false)
                      AND (((control->>'sequence')::bigint=%(expected)s AND control->'open_input'->>'input_id'=%(wait)s)
                        OR (control->>'_answer_operation_id'=%(operation)s AND (control->>'sequence')::bigint=%(sequence)s))
                    RETURNING 1""",params).fetchone()
                if row is None:raise StopRequested('answer publication identity/lease fence rejected')
            if batch is not None:batch.committed=True

    def journal_head(self, session_id: str, owner_id: str) -> int:
        with self._borrow() as conn:
            """Highest committed journal cursor (0 when empty) for cursor allocation."""
            row = conn.execute(
                "SELECT COALESCE(MAX(j.cursor),0) AS head FROM engine_session_journal j"
                " JOIN engine_sessions s ON s.session_id = j.session_id"
                " WHERE j.session_id = %(sid)s AND s.owner_id = %(owner)s::uuid",
                {"sid": session_id, "owner": owner_id}).fetchone()
            return row["head"] if isinstance(row, dict) else row[0]
