# 2026-11-07 - PR checks on open/reopen/ready only, not every push

`synchronize` entered test.yml on 2026-11-03 (commit c3ee040) and made every
push to an open PR re-run the whole suite. The owner's standing rule since
2026-10-10 is automatic PR checks on opened, reopened and ready_for_review
only; pushes are covered by the local gate (./scripts/test-local.sh) until the
planned fast (<=5 minute) PR gate lands. delivery-api made the same change in
its own PR today. If a flow ever needs CI on updated code, trigger it
explicitly (workflow_dispatch or a comment-driven run) instead of re-adding
synchronize for every PR.
