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
