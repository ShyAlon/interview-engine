"""Per-operation database statement counting.

Redundant-query cost is visible as one number per operation, never as
per-query lines: the counter rides a ContextVar, the proxies tick it, and the
owning operation summarizes (e.g. advance.turn.end ... queries=N). No
statement text, parameters or results are ever recorded.
"""
from contextvars import ContextVar
from contextlib import contextmanager

_current = ContextVar('engine_query_counter', default=None)


class QueryCounter:
    __slots__ = ("count",)

    def __init__(self):
        self.count = 0


@contextmanager
def measure():
    """One operation's statement budget. Nested measures are independent."""
    counter = QueryCounter()
    token = _current.set(counter)
    try:
        yield counter
    finally:
        _current.reset(token)


def _tick():
    counter = _current.get()
    if counter is not None:
        counter.count += 1


class CountingCursor:
    """Cursor passthrough that ticks the operation counter per statement."""
    __slots__ = ("_cur",)

    def __init__(self, cur):
        object.__setattr__(self, "_cur", cur)

    def execute(self, *args, **kwargs):
        _tick()
        return self._cur.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        _tick()
        return self._cur.executemany(*args, **kwargs)

    def __enter__(self):
        self._cur.__enter__()
        return self

    def __exit__(self, *args):
        return self._cur.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._cur, name)


class CountingConnection:
    """Connection passthrough: execute ticks, cursor() returns CountingCursor."""
    __slots__ = ("_conn",)

    def __init__(self, conn):
        object.__setattr__(self, "_conn", conn)

    def execute(self, *args, **kwargs):
        _tick()
        return self._conn.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        _tick()
        return self._conn.executemany(*args, **kwargs)

    def cursor(self, *args, **kwargs):
        return CountingCursor(self._conn.cursor(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._conn, name)
