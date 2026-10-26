# 2026-10-26 - Reads must not claim, heartbeat or construct (pool exhaustion)

Incident: a user viewing a parked task pinned ~4 Supabase pooler connections
and the session lease forever: the registry's only load path claimed the
lease, started a heartbeat and constructed the full application even for a
read-only snapshot. With the delivery pool (10) and two engine sessions the
15-client pooler budget ran out and a new session start died at
`FencedPostgresSaver.__init__` with FATAL EMAXCONNSESSION.

Lessons:
- Reads are reads. `SessionRegistry.get(serve=False)` restores the staged
  publication in memory only: no claim, no heartbeat, no graph, no publish
  write. The claimless restore is never cached, so a later serve claims first.
- Preserve admission contracts when you skip work: the snapshot route's 422
  for legacy unpinned sessions used to fall out of graph construction. The
  read path re-checks that condition explicitly (`_read_restore`).
- Idle connections are a budget line. The saver's guarded-write connection
  now opens lazily on the first fenced write; per-session peak is 4 conns
  (serving, heartbeat, application base, write) and
  `ENGINE_PEAK_SESSION_CONNS` + the delivery budget are asserted by tests
  against the pooler's 15.
