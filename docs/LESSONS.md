# Lessons

## 2026-10-26 - psycopg `conn.info.dsn` never carries the password

Every reconnect that reuses `conn.info.dsn` (heartbeat spawn, staging
connection, checkpoint write connection, application pg connect) connected
without a password and failed against the password-authenticated server with
`fe_sendauth: no password supplied`. The first connection always worked, so
single-connection tests never saw it. Cause: psycopg documents that
`Connection.info.dsn` "never returns the password". Prevention: all
reconnects go through `pg_claims.reconnect_dsn(conn)`, which rebuilds the
conninfo with `conn.info.password`; tests/test_pg_reconnect_dsn.py pins the
exact conninfo handed to every reconnect path. A secondary process note: the
fix first existed only as a local overlay on the Mac (2026-10-25) and was
lost when the preview was rebuilt from origin/main - local-only fixes must
be upstreamed the same day or the next rebuild silently reverts them.

## 2026-11-10: the placement probe ran on every registry, not once per pool

`pg_session_store` proved `engine_sessions` was reachable (checkout, claims, SELECT, commit: about five round trips)
every time delivery-api built a registry, 12 times in one warm answer, ~1 s each on a ~164 ms link. The question it
answers (table exists, role may read it) is pool-wide. Now one success per pool is remembered (weak key, so a closed
pool is forgotten); failure is never remembered and still raises the same tagged startup error.
tests/test_pg_session_store_probe_once.py pins once-per-pool, probe-per-new-pool, and failure-not-cached.
