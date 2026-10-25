"""Operator-only failure diagnostics for session-start 503 responses.

The block is sanitized by construction: the error class and object type are
kept, identifier values are redacted, and claim fields are PRESENCE booleans -
never claim values, never secrets. When the failed connection cannot answer
(aborted transaction or closed connection) the fields record UNKNOWN instead
of a false boolean. The block travels on the 503 response header so the
protocol error contract is untouched, and only the delivery service reads it
into an operator-only channel.
"""
from __future__ import annotations

import re

UNKNOWN = "UNKNOWN"
_MAX_MESSAGE = 300
_QUOTED_IDENTIFIER = re.compile(r'"[^"\r\n]{0,200}"')
_BARE_UUID = re.compile(r'(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b')
_WHITESPACE = re.compile(r'\s+')

_STATE_SQL = (
    "SELECT current_user AS role,"
    " current_setting('request.jwt.claim.sub', true) AS sub,"
    " current_setting('request.jwt.claim.session_id', true) AS sid"
)


def tag_stage(exc: BaseException, stage: str, conn=None) -> BaseException:
    """Attach the failing statement stage without changing the error type.

    The failing connection travels on the error so the HTTP boundary can ask
    it for role and claim presence; when it cannot answer, the block records
    UNKNOWN.
    """
    try:
        exc.engine_stage = stage
        if conn is not None:
            exc.engine_conn = conn
    except Exception:
        pass
    return exc


def _chain(exc: BaseException, depth: int = 6):
    """The error and its explicit causes, bounded. Wrappers such as the
    persistence startup error keep the tagged database error as their cause.
    """
    current = exc
    seen = set()
    while current is not None and depth and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        depth -= 1


def _first(exc: BaseException, name: str):
    for link in _chain(exc):
        value = getattr(link, name, None)
        if value is not None:
            return value
    return None


def stage_of(exc: BaseException) -> str | None:
    stage = _first(exc, "engine_stage")
    return stage if isinstance(stage, str) and stage else None


def sanitize_message(text: object) -> str:
    """Keep the error class and object type; redact identifier values.

    Quoted identifiers (relation, role and column names) and bare UUIDs are
    tenant-shape data and are replaced by fixed placeholders. Bounded length,
    single line, ASCII-safe output.
    """
    if not text:
        return ""
    flat = _WHITESPACE.sub(" ", str(text)).strip()
    flat = _QUOTED_IDENTIFIER.sub('"[identifier]"', flat)
    flat = _BARE_UUID.sub("[uuid]", flat)
    return flat[:_MAX_MESSAGE]


def _connection_state(conn) -> dict:
    """Role and claim PRESENCE from the failing connection.

    An aborted transaction or a closed/broken connection cannot answer; every
    field then records UNKNOWN rather than guessing. Claim presence prefers
    what the connection carries in-process (a claimed connection knows the
    claims it applies to every transaction); only a plain connection is asked
    for its startup settings, because transaction-local claims do not survive
    the rollback that follows a failure and reading them there would record a
    false absence. session_active_for_sub is only queried when both claims
    are visible as settings, and stays UNKNOWN otherwise.
    """
    state = {"role": UNKNOWN, "sub_present": UNKNOWN, "session_id_present": UNKNOWN,
             "session_active_for_sub": UNKNOWN}
    carried = getattr(conn, "claims", None)
    try:
        row = conn.execute(_STATE_SQL).fetchone()
        if hasattr(row, "keys"):
            role, sub, sid = row["role"], row["sub"], row["sid"]
        else:
            role, sub, sid = row[0], row[1], row[2]
    except Exception:
        role = sub = sid = None
        if carried is None:
            return state
    if role:
        state["role"] = str(role)
    if carried is not None:
        try:
            state["sub_present"] = bool(carried[0])
            state["session_id_present"] = bool(carried[1])
        except Exception:
            state["sub_present"] = state["session_id_present"] = UNKNOWN
        return state
    state["sub_present"] = bool(sub)
    state["session_id_present"] = bool(sid)
    if state["sub_present"] and state["session_id_present"]:
        try:
            active = conn.execute("SELECT public.requi_session_active() AS active").fetchone()
            flag = active["active"] if hasattr(active, "keys") else active[0]
            state["session_active_for_sub"] = bool(flag)
        except Exception:
            state["session_active_for_sub"] = UNKNOWN
    return state


def build_operator_block(exc: BaseException) -> dict:
    """The sanitized operator block for one failed start attempt.

    sqlstate is PostgreSQL's own five-character class code when the failure
    came from the database. stage names which statement failed when the
    persistence layer tagged it. The message is redacted by sanitize_message.
    Connection fields are UNKNOWN unless the failing connection itself
    answered them.
    """
    sqlstate = _first(exc, "sqlstate")
    block = {
        "error_class": type(exc).__name__,
        "sqlstate": sqlstate if isinstance(sqlstate, str) and len(sqlstate) == 5 else None,
        "stage": stage_of(exc),
        "message": sanitize_message(str(exc)),
        "role": UNKNOWN,
        "sub_present": UNKNOWN,
        "session_id_present": UNKNOWN,
        "session_active_for_sub": UNKNOWN,
    }
    conn = _first(exc, "engine_conn")
    if conn is not None:
        block.update(_connection_state(conn))
    return block
