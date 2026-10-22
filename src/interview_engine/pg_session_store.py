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

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def spawn(self) -> "PgSessionStore":
        # A second store on its own connection: a psycopg connection is not
        # thread-safe, so a background heartbeat must never share the serving
        # thread's connection.
        import psycopg.rows
        return PgSessionStore(psycopg.connect(self.conn.info.dsn, autocommit=False,
                                              row_factory=psycopg.rows.dict_row))

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
            " WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid"
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
            " WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid AND generation = %(gen)s"
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
            " WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid AND generation = %(gen)s"
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

    def stop(self, session_id: str, owner_id: str, control: dict | None = None) -> int:
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
        row = self.conn.execute(
            "UPDATE engine_sessions SET generation = generation + 1, lease_expires_at = 0,"
            " control = COALESCE(%(control)s::jsonb, control, '{}'::jsonb) || %(stopped_marker)s::jsonb,"
            " updated_at = %(now)s WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid"
            " RETURNING generation",
            {"sid": session_id, "owner": owner_id,
             "control": json.dumps(control) if control is not None else None,
             "stopped_marker": '{"stopped": true}',
             "now": time.time()}).fetchone()
        self.conn.commit()
        if row is None:
            raise KeyError(session_id)
        return row["generation"]

    def load_published(self, session_id: str, owner_id: str) -> dict | None:
        """Restore from the staged publication only - never saver-latest rows."""
        row = self.conn.execute(
            "SELECT task_id, generation, control, published_thread_id,"
            "       published_checkpoint_ns, published_checkpoint_id"
            " FROM engine_sessions WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid",
            {"sid": session_id, "owner": owner_id}).fetchone()
        if row is None:
            return None
        return dict(row)

    def stage_response(self, claim: SessionClaim, *, stage: str, item: str | None,
                       request_sha256: str, payload: dict, usage: dict | None,
                       provider: str, model: str, attempt: int = 1) -> None:
        """Durably stage one usable provider response before interpretation (PR B).
        Same-key replay is a no-op read; a same-key DIFFERENT payload conflicts loudly."""
        import json as _json, time as _time
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO staged_provider_responses(owner_id,session_id,stage,item,request_sha256,response,usage,provider,model,attempt,created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (session_id,stage,request_sha256) DO NOTHING",
                (claim.owner_id, claim.session_id, stage, item, request_sha256, _json.dumps(payload),
                 _json.dumps(usage) if usage is not None else None, provider, model, attempt, _time.time()))
            cur.execute(
                "SELECT response FROM staged_provider_responses WHERE session_id=%s AND stage=%s AND request_sha256=%s",
                (claim.session_id, stage, request_sha256))
            row = cur.fetchone()
        if row is None or row["response"] != payload:
            raise PublicationConflict('staged response conflict: same request hash, different payload')

    def load_staged_response(self, session_id: str, owner_id: str, *, stage: str,
                             request_sha256: str) -> dict | None:
        """The staged response for this exact request, or None."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT response,usage,provider,model,attempt FROM staged_provider_responses "
                "WHERE session_id=%s AND owner_id=%s AND stage=%s AND request_sha256=%s",
                (session_id, owner_id, stage, request_sha256))
            return cur.fetchone()

    def append_journal(self, claim: SessionClaim, cursor: int, page: dict) -> None:
        """Append one journal page under the fence.

        Pages are immutable evidence keyed by (session, cursor): a duplicate
        cursor is a no-op, never an overwrite. Part of the same staged
        publication discipline as control state - a stop or takeover that
        landed rejects the append with StopRequested.
        """
        row = self.conn.execute(
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
            guard = self.conn.execute(
                "SELECT 1 FROM engine_sessions WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid"
                " AND generation = %(gen)s AND lease_epoch = %(epoch)s AND lease_expires_at > %(now)s",
                {"sid": claim.session_id, "owner": claim.owner_id, "gen": claim.generation,
                 "epoch": claim.epoch, "now": time.time()}).fetchone()
            if guard is not None:
                self.conn.commit()
                return  # page already staged at this cursor; immutable, kept
            self.conn.rollback()
            raise StopRequested("session lease fence rejected the journal append")
        self.conn.commit()

    def read_journal(self, session_id: str, owner_id: str, after: int, limit: int) -> list[dict]:
        """Committed journal pages for progress reads; cross-owner reads see nothing."""
        rows = self.conn.execute(
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
        row = self.conn.execute(
            "SELECT control, published_thread_id, published_checkpoint_ns, published_checkpoint_id"
            " FROM engine_sessions WHERE session_id = %(sid)s AND owner_id = %(owner)s::uuid"
            " AND generation = %(gen)s AND lease_epoch = %(epoch)s AND lease_expires_at > %(now)s"
            " FOR UPDATE",
            params).fetchone()
        if row is None:
            self.conn.rollback()
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
                    existing = self.conn.execute(
                        "SELECT page FROM engine_session_journal WHERE session_id = %(sid)s AND cursor = %(cursor)s",
                        {**params, "cursor": cursor}).fetchone()
                    exact = existing is not None and existing["page"] == page
                    if not exact:
                        break
            if exact:
                self.conn.commit()   # exact replay of a committed step: no-op
                return
            self.conn.rollback()
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
            for cursor, page in pages:
                inserted = self.conn.execute(
                    "INSERT INTO engine_session_journal(session_id,cursor,page,created_at)"
                    " VALUES (%(sid)s,%(cursor)s,%(page)s::jsonb,%(now)s)"
                    " ON CONFLICT (session_id, cursor) DO NOTHING RETURNING 1",
                    {**params, "cursor": cursor, "page": json.dumps(page)}).fetchone()
                if inserted is None:
                    existing = self.conn.execute(
                        "SELECT page FROM engine_session_journal WHERE session_id = %(sid)s AND cursor = %(cursor)s",
                        {**params, "cursor": cursor}).fetchone()
                    if existing is None or existing["page"] != page:
                        raise PublicationConflict(
                            f"journal cursor {cursor} for {claim.session_id} already holds a different page")
                    raise PublicationConflict(
                        f"journal cursor {cursor} for {claim.session_id} belongs to an earlier step")
            if fault_hook is not None:
                fault_hook()   # test-only: inject a failure between journal insert and pointer commit
            self.conn.execute(
                "UPDATE engine_sessions"
                " SET control = %(control)s::jsonb, published_thread_id = %(tid)s,"
                "     published_checkpoint_ns = %(ns)s, published_checkpoint_id = %(cid)s,"
                "     updated_at = %(now)s"
                " WHERE session_id = %(sid)s",
                {**params, "control": json.dumps(control), "tid": published_thread_id,
                 "ns": published_checkpoint_ns, "cid": published_checkpoint_id})
        except BaseException:
            self.conn.rollback()
            raise
        self.conn.commit()

    def journal_head(self, session_id: str, owner_id: str) -> int:
        """Highest committed journal cursor (0 when empty) for cursor allocation."""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(j.cursor),0) AS head FROM engine_session_journal j"
            " JOIN engine_sessions s ON s.session_id = j.session_id"
            " WHERE j.session_id = %(sid)s AND s.owner_id = %(owner)s::uuid",
            {"sid": session_id, "owner": owner_id}).fetchone()
        return row["head"] if isinstance(row, dict) else row[0]
