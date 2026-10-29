# Live PG progress must not pretend a step committed

BufferedJournal intentionally publishes only at step commit. It cannot satisfy
heartbeats during a blocked step. Add explicit constructor-only observational
mode, leaving buffered default, deployment wiring and local SQLite unchanged.
Live rows carry journal_mode=observational in JSONB. The public ProgressSignal
schema is unchanged: transport strips this internal marker; clients must treat
all progress as observational, never graph/control/checkpoint evidence.

Allocation, insertion and lease guard are one transaction under the session row
lock. Owner/generation/epoch and wall-clock expiry fence writes. Durable first-row
mode checks prevent mixing live and buffered streams, in both directions,
including across process restarts. No migrations or pool additions. Each live
signal uses one checkout and three data statements; claimed transactions add
one claims statement (not counted in mirrored-superuser benchmark).

Fail-first new tests: 3 failed in 2.22s, missing class. Final selection of new
cases and original journal/store/application/recovery suites: 51 passed in
50.23s (final measured run in PR). No old assertion changed
or removed. Tests cover concurrent allocation, paging, restart, blocked read,
invalid data, stop, stale generation/epoch, expiry/takeover, scope and mode mixing.
Delivery tests apply actual migrations and FORCE RLS, check tenant and revoked
real auth-session access, and preserve all blocked-advance assertions.

Local disposable PG benchmark, 200 serial signals: 200 checkouts, 600 data
statements, mean append .829ms, p95 1.042ms, max 2.610ms. Guard roundtrip including
lock wait mean .183ms/p95 .232ms; post-guard-to-commit-return lock-hold upper bound
mean .558ms/p95 .724ms/max 1.604ms. This is NOT a production pooler/network or
contended-load latency guarantee. Active long-step heartbeats mean one transaction
per signal, unlike buffered per-step publication. No recommendation to activate
without production-shaped load measurement. Hosted CI/full suite unverified.
