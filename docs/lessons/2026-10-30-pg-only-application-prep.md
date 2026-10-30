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

Full-suite split survey across 98 files: third0 304 pass + one stale expected
Stop status; third1 327 pass; third2 297 pass + one old progress import.
These were corrected to proven PG STOPPED and real fenced PG journal respectively.
Focused correction gate recorded separately. No full-suite green claim yet.

Final gate after fixes and operator-path retirement: all 99 test files split
into three non-overlapping batches, DELIVERY_TEST_DELIVERY_SOURCE enabled:
300 + 312 + 313 = 925 passed, no skips/deselections. Collection verified separately.
Hosted CI remains unverified. No runtime/script SQLite import remains and its
package dependency is removed. The source technology catalog is unaffected.

Explicit final retirement accounting:
- Delete tests/test_binding_migration.py's nine legacy operator executions:
  dry-run/apply/backup/repeat; read-only snapshot plus stamp application; schema
  and identity conflict; scan/services-stopped gate; interrupted backup; missing
  checkpoint/unsupported stamp/symlink; concurrent changed source; interrupted
  final audit; true binding drift. The 0.19-to-0.20 stamp-only file migration
  itself is retired, not ported. New immutable binding drift/migration-refusal
  test preserves the retained no-silent-rewrite behavior, not migration support.
- Remove the SQLite checkpoint input and bulk discovery helper from labeling
  enrichment. One latest-local-checkpoint reader execution is retired; explicit
  exported transcript enrichment remains and is tested, with hashes unguessed.
- HTTP memory/file parametrized repetitions retired; all shared test functions
  still execute on PG. Explicit file-start cases become refusal tests. The one
  tenant file-router success execution is retired. Missing-store Stop route
  test becomes constructor refusal, since unsupported registry never starts.
- Graph crash/resume test renamed and uses real fenced PG, not MemorySaver.

Crash recovery: process death before acknowledgement intentionally leaves
STOP_REQUESTED. Existing Debug engine.process_status_name shows this; a direct
projection test proves both statuses remain visible. Offline operator recovery
requires the exact stopped owner/session/generation/old epoch/old process ID,
with dry-run default and explicit --apply --writers-terminated attestation.
Operator must terminate the actual process and accepted provider threads first;
a timeout, lease expiry or new registry is never proof. Script changes only
quiescence JSON under a row lock, no new schema/role/public route/resume. It is
not an automatic process-death detector and was exercised only on disposable PG.
The agent did not run this script on live state. README gives the recovery path.
