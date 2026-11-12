# Diagnostic-copy cleanup after confirmed save

The existing answer flush already removed its local backup after the document
adapter returned from its fenced transaction. That did not make old files safe
to delete by age, and its unconditional unlink could remove a newer replacement
at the same operation path. Save failure and cleanup failure also shared the
same warning even when the notes were committed.

The cleanup rule is now exact:

1. Capture the operations and serialized recovery snapshot before saving.
2. Let the existing document adapter commit under its normal owner/auth/session,
   generation, lease and Stop fence. No new publication transaction is wrapped
   around the adapter and no local capacity lock spans database/provider work.
3. After it returns, acquire the same owner lock used for atomic spool writes.
   Read a bounded regular, no-follow local file and compare exact snapshot bytes.
   Remove only a matching file whose inode has not changed during the read.
4. Keep changed, unsafe, malformed, oversized, failed and unknown copies. A
   cleanup error reports confirmed notes plus deferred local cleanup, not a
   failed database save. Errors never replace the original answer/Stop error.

This introduces no TTL, age eviction, background sweep, receipt journal, database
schema change, import or replay. A crash between commit and cleanup can still
leave an unconfirmed local copy. It stays under the existing per-owner capacity
limits. File age or task completion is not confirmation. Capacity exhaustion
still stops new backup copies, not answer work. Old crash temporaries remain.

Verification so far: the replaced-copy and cleanup-warning assertions failed
before the fix. Sixteen focused diagnostic tests and a 34-test answer/transaction
boundary group pass. Engine fast gate: 780 passed, three optional skips. Its
stale runtime-role double now models the existing pipeline boundary and retains
the exact role-before-claims assertion. Actual migrated-PG proof: four integration cases pass, including a new test
that reads committed bytes from a separate connection before cleanup and proves
Stop keeps unsaved evidence without changing saved bytes. Delivery installed
child/diagnostic export/Stop/durable-HTTP group: ten passed.

All 24 engine heavy-classified files were exercised in bounded groups: 59
passed with one cross-repo test-module-name collision in the first group, 53
passed in the second, 103 in the third. The affected wiring module passes all
15 cases in its own process. The collision comes from a delivery migration
fixture adding its tests directory to the module path; no product code changed
to hide it. This is not a claim of one uninterrupted full-suite pass. Hosted
CI is not claimed green. No paid execution or live DB used.
