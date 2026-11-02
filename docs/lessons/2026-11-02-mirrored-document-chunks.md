# Application test schema must track document functions

The production adapter switched to chunked document functions in engine85, but the shared application test schema still created only the pre-chunk table. Full CI had27failures before new progress changes. The same wiring test failed on clean main and the progress branch with UndefinedFunction(requi_document_parts).

Mirror the migration-owned chunk columns, table, reset trigger, append and parts functions in the shared disposable schema. Keep this fixture schema-only: do not imitate tenant authorization, remove runtime checks, or stub document reads. Full migrated FORCE-RLS integration remains in delivery-api.

Verification:72tests covering application wiring, checkpoint namespace, live progress, call summaries, crash recovery, resume and write budgets pass. The selected wiring test fails on clean main before this fixture update.
