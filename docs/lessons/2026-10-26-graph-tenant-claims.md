# 2026-10-26 — Graph connections claimed the interview session; RLS read every tenant table as empty

## Incident
Task 8c273f92abf8 ("Meditator 7.2", session 02054ca8-1369-44d6-b03c-db9ea47787fa,
2026-10-26 14:44-14:46Z): start accepted, lease claimed, heartbeat ran, session
published seq=1, two DeepSeek calls settled - then the first checkpoint write
failed: `lease.fence.rejected what=checkpoint guard lock
reason=session-row-missing expected gen=1 epoch=1` -> StopRequested -> swallowed
-> start returned RUNNING seq=1 open_input=null. The interview parked with no
progress events. Same shape as 7.1 earlier that day: the pg checkpoint path has
never completed a real advance.

## Root cause (proven from code + DDL)
Two different "session" identities meet at the graph's Postgres connections:
- Tenant claims (pg_claims.CLAIM_SQL `request.jwt.claim.session_id`) must name a
  live AUTH session: delivery's forced RLS policy `tenant_isolation`
  (20261007150000) requires `requi_session_active()`, which checks auth.sessions.
  tenant_worker builds one registry per (owner, auth session), so every store,
  heartbeat and staging connection carries the auth session id - and worked.
- The lease claim (SessionClaim) is the fence identity; its session_id is the
  INTERVIEW session.
application.py built the graph's pg_conn and the saver's _write_conn with
claims=(pg_claim.owner_id, pg_claim.session_id) - the interview session. On
those connections requi_session_active() is false, so FORCE-RLS tenant tables
read EMPTY. The first claims-dependent checkpoint statement (guard lock) found
no engine_sessions row; the diagnose read found none either; saver.setup() had
survived only because checkpoint_migrations' policy is USING(true). Checkpoint
reads were equally empty, so any retried advance would restart from scratch.

## Why tests missed it
pg-mode tests run against pgserver as superuser (RLS bypassed) with claimless
stores, so the wrong claims pair was never observable. The new test gives the
store claims=('owner', auth_session) and asserts the saver's connections carry
exactly that pair, not the lease claim's interview session.

## Fix
application.py: the graph's tenant claims come from `pg_store.claims`
(the auth-session pair) via `_graph_tenant_claims`; claimless stores
(mirrored-DDL superuser tests) keep the legacy lease-claim pair.

## Lesson
When one table row is visible to some connections and "missing" to others,
compare what each connection CLAIMS before suspecting deletion, snapshots, or
stale code. Anywhere two identifiers share the word "session", name them
(auth_session vs interview_session) at every seam - a generic `session_id`
parameter silently accepts the wrong one.
