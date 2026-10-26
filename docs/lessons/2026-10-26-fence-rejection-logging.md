# 2026-10-26 — Fence rejections must log why they tripped

## Incident
Task 9dd349f2ef25 ("Meditator 7.1", session 8386d290-c6cb-4060-bb55-0ec6a8b39ebe,
2026-10-26 ~07:54:46Z): two qwen2.5-coder:14b calls settled, then start() logged
`stop.swallowed reason=stop-requested` and returned success RUNNING with seq=1 and
open_input=null. The session parked forever: the first advance never committed a
journal entry, so delivery showed "Waiting for stage progress" indefinitely.

## Root cause hypothesis (unproven at fix time)
lease_ttl=30s; the ~10s heartbeat thread likely starved or hung under Ollama host
load, the lease lapsed, and a fenced write (most likely pg_checkpoint inside the
graph, since publish_step never runs inside _advance) raised StopRequested, which
start()'s outer catch swallowed as "reason=stop-requested" with no exception detail.

## The logging gap
Six fence raise sites (pg_session_store save/journal/publication, pg_checkpoint
_fenced/_guard_lock, stop_fence check/stopped-usage-sink) rejected with no record of
which predicate failed or what generation/lease state was observed. The two
stop.swallowed sites logged only the reason string, not the real exception. One log
line could not distinguish a lost lease from a takeover from a quiescence timeout.

## Fix (no behavior change)
- stop.swallowed warnings now log `cause=<type>: <str(exc)[:300]>`.
- stop_fence logs `stop.fence.tripped` at the quiescence, check (with
  requested/expected/seen generation), and stopped-usage-sink sites.
- pg_session_store and pg_checkpoint gained a best-effort `_fence_diagnose()` that
  SELECTs generation/lease_epoch/lease_expires_at/placement_instance before raising
  and logs which predicate failed. Diagnosis never masks the rejection: any error in
  the probe itself is logged and the original raise proceeds.

## Lesson
Every fail-closed rejection needs to answer "why" at the moment it trips, with the
observed state that made it trip. A swallowed exception is a deleted stack trace:
log the real cause at the swallow site, because the outer layer only sees the reason
string. Hermetic tests can assert the log contract without a database by faking the
store connection.

## Addendum: the heartbeat was invisible too
The lease heartbeat thread logged only per-beat failures. The two decisive paths
were silent: `alive=False` (lease lost - stop or takeover) marked the session
stopped with no line at all, and the 5-failure give-up returned quietly. The
thread now logs its start (session, interval, ttl), the lost-lease transition
(WARNING), and the give-up (WARNING). In the in-process deployment these lines
reach the delivery log buffer and the mirrored engine.log.
