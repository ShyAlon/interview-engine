# PG-only application preparation

Fail-first application construction test proves missing PG store/lease refuses
before model routing, directory creation or any provider work. Application no
longer constructs EvidenceLake, file checkpointer or ProgressJournal. Existing
fenced PG saver, database evidence and buffered/observational journal preserved.

Converted root-checkpoint namespace and call-end journal tests to disposable PG
rather than dropping assertions. Existing custom-factory provider guard test now
supplies its actual registry PG store/lease before probing the unchanged dispatch
refusal. Assertions for zero provider calls and refusal text remain.

Focused application/live progress/namespace/call-end/PG-required gate: 27 passed,
no skips/deselections. No test execution deleted, no security assertion weakened.
This is preparation, NOT mergeable yet: HTTP startup and TenantRegistryRouter
still expose explicit file mode. Full collection/test gate and remaining runtime
entry retirement required before PR. No deploy, live DB changes/reset, spend,
auth change or flake-tracing removal. Hosted CI unverified.

Service startup now refuses explicit retired file mode. Real delivery schema
startup suite: 20 passed, no skips. Tenant router no longer accepts
allow_file_sessions or constructs a file registry. Cache/capacity/isolation
structure tests use an explicit fake PG port; real disposable PG HTTP test
checks foreign-owner read/stop refusal, own-owner Stop and authentication.
PG session IDs are globally unique, unlike the old per-owner same-ID file
namespace; that old duplicate-ID capability and stop-marker file assertions
are intentionally retired, not weakened into a false parity claim. Symlink
workspace and service-token rejection tests retained. One file-mode startup
execution retired with the capability; missing-DSN refusal remains.
Router/startup/application gate: 32 passed, no skips. Still preparation: direct
SessionRegistry file fallbacks and old test fixtures remain to convert before PR.

Direct registry file save/load/stop/progress and standalone program factory now
refuse or use PG only. Removed progress_journal implementation; all three
progress tests use real observational PG, preserving cursor/concurrency/auth
and sequence assertions. HTTP fixtures run PG only; redundant memory/file
executions retired, test functions retained. File marker quiescence was NOT
retired: real-PG replacement tracks registry writers and deadline-provider
threads through actual exit. Stop CAS acknowledgement binds exact old owner,
instance, epoch and stopped generation; stopped readers never take a new lease.
Restart without exit proof stays STOP_REQUESTED. Existing file Stop test
functions are ported, not deleted; final STOPPED assertions retained. Captured
HTTP responses can precede writer-finally acknowledgement, so tests separately
check durable STOPPED after the request returns. Stop increments sequence once;
the PG late-result test accounts for acceptance plus Stop, not a late commit.

Fail-first caught two real PG gaps: a blocked factory still called its late
program after Stop; and a timed-out provider thread falsely allowed STOPPED.
Both regressions are covered, with stale/foreign CAS and restart cases.
Registry/Stop/fence/diagnostics gate 49 passed; Stop/progress/registry 30 passed;
legacy route/resume + binding operator fixtures 46 passed. HTTP/config gate
previously 63 passed plus one progress client-shutdown test error, fixed by not
closing the app while its start is blocked and rerun in the 30-pass gate.
Broad suite survey timed out after 54 percent, not a full-suite result.
Cross-repo ordering exposed a test sys.path leak loading delivery's similarly
named test module. Use monkeypatch.syspath_prepend so teardown restores the path;
real cross-repo gateway + subsequent engine crash test now 4 passed.

Still preparation, not ready: full gated test batches, dormant binding migration
and labeling SQLite retirement, and explicit operator recovery / Debug evidence
for a process crash without writer-exit proof remain to finish. No live writes.
