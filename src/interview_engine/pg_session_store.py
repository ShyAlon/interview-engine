"""Postgres session registry: placement leases and fenced control-state saves.

The file-backed registry (http-session.json + StopFence) orders writers on one
host only. This store moves the same invariants into the database so a session
can be claimed, served, stopped and restored on any host:

- Claim-before-serve: a host serves a session only after one atomic
  conditional UPDATE wins its lease (expired or already its own). Each won
  claim bumps lease_epoch; the returned claim carries (generation, epoch) and
  is the fence identity for every later mutation, including checkpoint writes
  through pg_checkpoint.FencedPostgresSaver.
- Fencing: control-state saves and publication-pointer updates commit only
  while the writer's (generation, epoch) still matches and the lease is
  unexpired. A stop or a competing takeover serializes on the same row lock,
  so there is no epsilon window between check and write. A failed fence
  raises StopRequested; nothing is silently dropped.
- Stop is owner-authorized: it bumps generation and expires the lease WITHOUT
  holding the execution lease, and blocks every later fenced mutation.
- Restore reads only the staged publication pointer and control state, never
  saver-latest: a crashed writer can leave unpublished checkpoint rows, and
  they stay invisible until a fenced save publishes them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import psycopg

from .stop_fence import StopRequested

REGISTRY_DDL = """CREATE TABLE IF NOT EXISTS engine_sessions(
 session_id TEXT PRIMARY KEY,
 task_id TEXT NOT NULL,
 generation BIGINT NOT NULL,
 lease_epoch BIGINT NOT NULL,
 placement_host TEXT,
 lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0,
 control JSONB,
 published_thread_id TEXT,
 published_checkpoint_ns TEXT NOT NULL DEFAULT '',
 published_checkpoint_id TEXT,
 created_at DOUBLE PRECISION NOT NULL,
 updated_at DOUBLE PRECISION NOT NULL)"""


@dataclass(frozen=True)
class SessionClaim:
    """A won lease: the fence identity for every mutation of this session."""
    session_id: str
    generation: int
    epoch: int
    host: str


class PgSessionStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def setup(self) -> None:
        self.conn.execute(REGISTRY_DDL)
        self.conn.commit()

    def create(self, session_id: str, task_id: str) -> None:
        now = time.time()
        self.conn.execute(
            "INSERT INTO engine_sessions(session_id,task_id,generation,lease_epoch,lease_expires_at,created_at,updated_at)"
            " VALUES (%s,%s,1,0,0,%s,%s)",
            (session_id, task_id, now, now))
        self.conn.commit()

    def claim(self, session_id: str, host: str, ttl: float) -> SessionClaim | None:
        """Win the lease atomically, or return None when another live lease holds it."""
        now = time.time()
        row = self.conn.execute(
            "UPDATE engine_sessions"
            " SET lease_epoch = lease_epoch + 1, placement_host = %(host)s,"
            "     lease_expires_at = %(expiry)s, updated_at = %(now)s"
            " WHERE session_id = %(sid)s"
            "   AND (lease_expires_at <= %(now)s OR placement_host = %(host)s)"
            " RETURNING lease_epoch, generation",
            {"sid": session_id, "host": host, "expiry": now + ttl, "now": now}).fetchone()
        self.conn.commit()
        if row is None:
            return None
        return SessionClaim(session_id, row["generation"], row["lease_epoch"], host)

    def heartbeat(self, claim: SessionClaim, ttl: float) -> bool:
        """Extend a held lease. False means the lease was lost (stop or takeover)."""
        now = time.time()
        row = self.conn.execute(
            "UPDATE engine_sessions SET lease_expires_at = %(expiry)s, updated_at = %(now)s"
            " WHERE session_id = %(sid)s AND generation = %(gen)s"
            "   AND lease_epoch = %(epoch)s AND placement_host = %(host)s"
            "   AND lease_expires_at > %(now)s"
            " RETURNING 1",
            {"sid": claim.session_id, "gen": claim.generation, "epoch": claim.epoch,
             "host": claim.host, "expiry": now + ttl, "now": now}).fetchone()
        self.conn.commit()
        return row is not None

    def fenced_save(self, claim: SessionClaim, control: dict, *,
                    published_thread_id: str | None = None,
                    published_checkpoint_ns: str = "",
                    published_checkpoint_id: str | None = None) -> None:
        """Commit control state and the staged publication pointer under the fence."""
        import json as _json
        row = self.conn.execute(
            "UPDATE engine_sessions"
            " SET control = %(control)s::jsonb, published_thread_id = %(tid)s,"
            "     published_checkpoint_ns = %(ns)s, published_checkpoint_id = %(cid)s,"
            "     updated_at = %(now)s"
            " WHERE session_id = %(sid)s AND generation = %(gen)s"
            "   AND lease_epoch = %(epoch)s AND lease_expires_at > %(now)s"
            " RETURNING 1",
            {"sid": claim.session_id, "gen": claim.generation, "epoch": claim.epoch,
             "control": _json.dumps(control), "tid": published_thread_id,
             "ns": published_checkpoint_ns, "cid": published_checkpoint_id,
             "now": time.time()}).fetchone()
        if row is None:
            self.conn.rollback()
            raise StopRequested("session lease fence rejected the save")
        self.conn.commit()

    def stop(self, session_id: str) -> int:
        """Owner-authorized Stop: bump generation and expire the lease. No lease held."""
        row = self.conn.execute(
            "UPDATE engine_sessions SET generation = generation + 1, lease_expires_at = 0,"
            " updated_at = %(now)s WHERE session_id = %(sid)s RETURNING generation",
            {"sid": session_id, "now": time.time()}).fetchone()
        self.conn.commit()
        if row is None:
            raise KeyError(session_id)
        return row["generation"]

    def load_published(self, session_id: str) -> dict | None:
        """Restore from the staged publication only - never saver-latest rows."""
        row = self.conn.execute(
            "SELECT task_id, generation, control, published_thread_id,"
            "       published_checkpoint_ns, published_checkpoint_id"
            " FROM engine_sessions WHERE session_id = %s",
            (session_id,)).fetchone()
        if row is None:
            return None
        return dict(row)
