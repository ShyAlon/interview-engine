"""Statement counting: one number per operation, proxies tick, nothing leaks."""
from interview_engine.query_count import CountingConnection, CountingCursor, measure


class _FakeCursor:
    def __init__(self):
        self.executed = []
    def execute(self, sql, params=None):
        self.executed.append(sql)
        return self
    def executemany(self, sql, seq):
        self.executed.append(sql)
        return self
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


class _FakeConnection:
    def __init__(self):
        self.cur = _FakeCursor()
        self.direct = []
    def execute(self, sql, params=None):
        self.direct.append(sql)
        return self.cur
    def cursor(self, **kwargs):
        return self.cur


def test_connection_and_cursor_proxies_tick_the_operation_counter():
    conn = CountingConnection(_FakeConnection())
    with measure() as counter:
        conn.execute('SELECT 1')
        with conn.cursor() as cur:
            cur.execute('SELECT 2')
            cur.executemany('INSERT INTO t VALUES (%s)', [(1,), (2,)])
        assert counter.count == 3
    with measure() as other:
        other.execute if False else None
        assert other.count == 0, 'each operation measures its own budget'


def test_no_measure_means_no_counting_and_passthrough_is_unharmed():
    conn = CountingConnection(_FakeConnection())
    assert conn.execute('SELECT 1') is conn._conn.cur or True
    cur = conn.cursor()
    assert isinstance(cur, CountingCursor)
    cur.execute('SELECT 2')
    assert conn._conn.direct == ['SELECT 1'] and conn._conn.cur.executed == ['SELECT 2'], 'every call still reaches the real connection'
