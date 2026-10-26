# 2026-10-26: One connection pool per process

## What broke
The 2026-10-26 EMAXCONNSESSION incident: the engine pinned one connection per
concurrent actor (registry serving, per-session heartbeat thread, application
base, staging, saver write) because psycopg connections are not thread-safe.
Against the 15-client Supabase pooler, a few live sessions exhausted the budget.

## Why the pins were never needed
- Fencing is data-level: every guarded write re-checks (session, generation,
  epoch, expiry) in the same statement. The write-time check is the authority,
  not the connection.
- Tenant claims ride the transaction stream (pg_claims set_config as the first
  statement of each transaction), so a pooled connection serves any session.
- langgraph's PostgresSaver accepts a psycopg ConnectionPool natively and
  checks out per operation.

## The redesign
- One psycopg_pool per process (pg_pool.build_pool); every store/saver method
  borrows per operation and the borrow guard rolls back anything left open.
- One shared heartbeat loop per registry beats every held lease in a single
  heartbeat_many call (60s beat, TTL 150s). Lost claims mark sessions stopped;
  transient tick failures log and the loop continues - it must never give up,
  because it beats every session in the process.
- The budget becomes ONE number per process: delivery 6 + engine 4 <= 15,
  pinned in test_connection_budget_fits_pooler.

## Costs accepted (measured, not guessed)
- Pipeline mode is unavailable on a pool. Measured on local pgserver (50 puts
  each, median): pinned+pipeline 0.68ms vs pool checkout 0.74ms per checkpoint
  put (+0.06ms, +9%). Against a network pooler the delta is roughly one extra
  RTT per put - negligible against model-call latency.
- Unclean process death: lease reclaim waits out the TTL (<=2.5min) instead of
  the old faster beat detecting it sooner.
- Claims must be applied at every checkout (borrow_claimed) - discipline now
  enforced by construction, since no code path holds a raw conn.

## Harness lesson
Tests that pinned implementation details (exact rollback counts, per-session
heartbeat threads, lazy write conn) all needed rewrites; tests that pinned
contracts (fence rejections, log lines, claims visibility) survived with only
construction changes. Pin contracts, not wiring.
