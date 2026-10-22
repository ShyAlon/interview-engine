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
import time
from dataclasses import dataclass

import psycopg

from .stop_fence import StopRequested


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

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def create(self, session_id: str, task_id: str, owner_id: str) -> None:
        now = time.time()
        self.conn.execute(
            "INSERT INTO engine_sessions(session_id,owner_id,task_id,generation,created_at,lease_epoch,lease_expires_at)"
            " VALUES (%s,%s,%s,1,%s,0,0)",
            (session_id, owner_id, task_id, now))
        self.conn.commit()

    def claim(self, session_id: str, owner_id: str, instance: str, ttl: float) -> SessionClaim | None:
        """Win the lease atomically, or return None while another instance holds it.

        An expired lease is taken with an epoch bump. A re-claim by the same
        live instance extends the expiry without a bump, so its own in-flight
        fenced writes are not invalidated by reentrant requests.
        """
        now = time.time()
        row = self.conn.execute(
            "UPDATE engine_sessions"
            " SET lease_epoch = CASE WHEN lease_expires_at <= %(now)s THEN lease_epoch + 1 ELSE lease_epoch END,"
            "     placement_instance = %(instance)s, lease_expires_at = %(expiry)s, updated_at = %(now)s"
            " WHERE session_id = %(sid)s AND owner_id = %(owner)s"
            "   AND (lease_expires_at <= %(now)s OR placement_instance = %(instance)s)"
            " RETURNING generation, lease_epoch",
            {"sid": session_id, "owner": owner_id, "instance": instance,
             "expiry": now + ttl, "now": now}).fetchone()
        self.conn.commit()
        if row is None:
            return None
        return SessionClaim(session_id, owner_id, row["generation"], row["lease_epoch"], instance)

    def heartbeat(self, claim: SessionClaim, ttl: float) -> bool:
        """Extend a held lease (never bumps the epoch). False means it was lost."""
        now = time.time()
        row = self.conn.execute(
            "UPDATE engine_sessions SET lease_expires_at = %(expiry)s, updated_at = %(now)s"
            " WHERE session_id = %(sid)s AND owner_id = %(owner)s AND generation = %(gen)s"
            "   AND lease_epoch = %(epoch)s AND placement_instance = %(instance)s"
            "   AND lease_expires_at > %(now)s"
            " RETURNING 1",
            {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
             "epoch": claim.epoch, "instance": claim.instance,
             "expiry": now + ttl, "now": now}).fetchone()
        self.conn.commit()
        return row is not None

    def fenced_save(self, claim: SessionClaim, control: dict, *,
                    published_thread_id: str | None = None,
                    published_checkpoint_ns: str = "",
                    published_checkpoint_id: str | None = None) -> None:
        """Commit control state and the staged publication pointer under the fence."""
        row = self.conn.execute(
            "UPDATE engine_sessions"
            " SET control = %(control)s::jsonb, published_thread_id = %(tid)s,"
            "     published_checkpoint_ns = %(ns)s, published_checkpoint_id = %(cid)s,"
            "     updated_at = %(now)s"
            " WHERE session_id = %(sid)s AND owner_id = %(owner)s AND generation = %(gen)s"
            "   AND lease_epoch = %(epoch)s AND lease_expires_at > %(now)s"
            " RETURNING 1",
            {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
             "epoch": claim.epoch, "control": json.dumps(control), "tid": published_thread_id,
             "ns": published_checkpoint_ns, "cid": published_checkpoint_id,
             "now": time.time()}).fetchone()
        if row is None:
            self.conn.rollback()
            raise StopRequested("session lease fence rejected the save")
        self.conn.commit()

    def stop(self, session_id: str, owner_id: str) -> int:
        """Owner-bound Stop: bump generation and expire the lease. No lease held.

        The authorization decision belongs to the HTTP boundary; binding the
        bump to the owner id keeps a foreign tenant from stopping (or even
        detecting) this session through the store.
        """
        row = self.conn.execute(
            "UPDATE engine_sessions SET generation = generation + 1, lease_expires_at = 0,"
            " updated_at = %(now)s WHERE session_id = %(sid)s AND owner_id = %(owner)s"
            " RETURNING generation",
            {"sid": session_id, "owner": owner_id, "now": time.time()}).fetchone()
        self.conn.commit()
        if row is None:
            raise KeyError(session_id)
        return row["generation"]

    def load_published(self, session_id: str, owner_id: str) -> dict | None:
        """Restore from the staged publication only - never saver-latest rows."""
        row = self.conn.execute(
            "SELECT task_id, generation, control, published_thread_id,"
            "       published_checkpoint_ns, published_checkpoint_id"
            " FROM engine_sessions WHERE session_id = %(sid)s AND owner_id = %(owner)s",
            {"sid": session_id, "owner": owner_id}).fetchone()
        if row is None:
            return None
        return dict(row)
