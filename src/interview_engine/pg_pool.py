"""One database connection pool per process for every engine database user.

The 2026-10-26 EMAXCONNSESSION incident: the engine pinned a connection per
concurrent actor (serving, per-session heartbeat, application base, staging,
saver write) because psycopg connections are not thread-safe. Nothing in the
lease/fence/claims design needs a pinned socket: fences are data-level
(checked per statement) and tenant claims are applied per transaction
(pg_claims). All engine database access borrows from this pool and returns
the connection at operation end.

Budget: the Supabase pooler (session mode) allows 15 clients. Standalone the
engine caps at ENGINE_POOL_MAX; hosted in-process inside delivery-api the
DELIVERY side builds one shared pool for the whole process and injects it
(see delivery_api.db_pool) - the engine never imports delivery code; the
contract is psycopg_pool.ConnectionPool's public interface.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

import psycopg
import psycopg.rows
from psycopg.pq import TransactionStatus
from psycopg_pool import ConnectionPool

log = logging.getLogger("interview_engine.pg_pool")

# Standalone budget: delivery 6 + engine 4 <= pooler 15 (both repos pin this
# arithmetic in a connection-accounting test). Hosted in-process the delivery
# host injects its own shared pool instead, and THIS cap does not apply.
ENGINE_POOL_MAX = 4


def build_pool(dsn: str, *, max_size: int = ENGINE_POOL_MAX, name: str = "engine") -> ConnectionPool:
    """One pool; open eagerly so a bad DSN fails clear at startup, not on first use."""
    pool = ConnectionPool(dsn, min_size=0, max_size=max_size, name=name, open=False,
                          kwargs={"row_factory": psycopg.rows.dict_row, "autocommit": False})
    pool.open()
    return pool


@contextmanager
def borrow(pool: ConnectionPool):
    """Check out one raw connection for one operation.

    A borrowed connection never goes back mid-transaction: callers commit or
    roll back explicitly, and this guard rolls back anything they missed.
    """
    with pool.connection() as conn:
        try:
            yield conn
        finally:
            if conn.info.transaction_status != TransactionStatus.IDLE:
                conn.rollback()


@contextmanager
def borrow_claimed(pool: ConnectionPool, claims: tuple[str, str] | None):
    """Check out one connection with tenant claims applied per transaction.

    Claims ride the transaction stream (pg_claims), never the pool: a pooled
    connection serves any session/tenant, claimed at checkout.
    """
    from .pg_claims import ClaimedConnection
    with pool.connection() as raw:
        conn = ClaimedConnection(raw, *claims) if claims is not None else raw
        try:
            yield conn
        finally:
            if raw.info.transaction_status != TransactionStatus.IDLE:
                raw.rollback()
