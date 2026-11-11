"""Per-transaction tenant claims for every engine database connection.

The Supabase pooler drops libpq startup-option GUCs (the conninfo `options`
payload): the 2026-10-25 live probe against the preview DSN showed
read_only_requested=on but read_only_verified=off, and startup claims visible
in neither of two sampled transactions. Claims carried in the startup packet
never reach the backend there, and every FORCE-RLS write then fails 42501.

Claims therefore ride the transaction stream itself: the first statement of
every transaction sets them transaction-locally (set_config is_local=true) -
the same pattern delivery-api's postgres_scope uses against the same database.
They reset at transaction end and are re-applied on the next transaction.
There is no startup-packet or session-level fallback.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack

import psycopg
import psycopg.rows
from psycopg.pq import TransactionStatus

CLAIM_SQL = ("SELECT set_config('request.jwt.claim.sub',%s,true),"
             " set_config('request.jwt.claim.session_id',%s,true)")


def apply_claims(conn, claims):
    """Enter the restricted runtime role and tenant in the same transaction."""
    with conn.pipeline():
        _enqueue_claims(conn,claims)


def _enqueue_claims(conn,claims):
    conn.execute('SET LOCAL ROLE requi_backend')
    conn.execute(CLAIM_SQL,claims)


class ClaimedConnection:
    """A psycopg connection that applies its tenant claims per transaction.

    execute()/cursor() apply the claims as the first statement when a new
    implicit transaction is beginning. transaction() applies them right after
    BEGIN - the only way transaction-local state spans several statements on
    an autocommit connection. Every other attribute proxies through.
    """

    def __init__(self, conn: psycopg.Connection, owner_id: str, session_id: str):
        if not owner_id or not session_id:
            raise ValueError('tenant claims require a real owner and session')
        self._conn = conn
        self.claims = (owner_id, session_id)

    def _apply_if_idle(self) -> None:
        if not self._conn.autocommit and self._conn.info.transaction_status == TransactionStatus.IDLE:
            apply_claims(self._conn, self.claims)

    def execute(self, *args, **kwargs):
        if not self._conn.autocommit and self._conn.info.transaction_status == TransactionStatus.IDLE:
            with self._conn.pipeline():
                _enqueue_claims(self._conn,self.claims)
                cursor=self._conn.execute(*args, **kwargs)
            return cursor
        return self._conn.execute(*args, **kwargs)

    def cursor(self, *args, **kwargs):
        self._apply_if_idle()
        return self._conn.cursor(*args, **kwargs)

    @contextmanager
    def transaction(self, *args, **kwargs):
        with claimed_transaction(self._conn,self.claims,*args,**kwargs):
            yield self

    def __getattr__(self, name):
        return getattr(self._conn, name)


@contextmanager
def claimed_transaction(conn,claims,*args,**kwargs):
    """One setup batch without an implicit second BEGIN before claims.

    A public Transaction starts/ends the block; temporary autocommit prevents
    execute() from emitting another BEGIN while the queued BEGIN is still idle.
    Nested savepoints keep the existing connection mode and transaction.
    """
    restore=conn.autocommit
    idle=conn.info.transaction_status == TransactionStatus.IDLE
    if idle:conn.autocommit=True
    try:
        with ExitStack() as stack:
            with conn.pipeline():
                stack.enter_context(psycopg.Transaction(conn,*args,**kwargs))
                if claims is not None:_enqueue_claims(conn,claims)
            yield conn
    finally:
        if idle and not conn.closed:
            if conn.info.transaction_status != TransactionStatus.IDLE:conn.rollback()
            conn.autocommit=restore


def connect_claimed(dsn: str, owner_id: str, session_id: str, **kwargs) -> ClaimedConnection:
    """Connect and bind (owner, session) claims for per-transaction application."""
    kwargs.setdefault('row_factory', psycopg.rows.dict_row)
    return ClaimedConnection(psycopg.connect(dsn, **kwargs), owner_id, session_id)
