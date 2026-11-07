# 2026-11-07 - Gate every PR on the fast core; full suite weekly and on demand

Mirrors delivery-api #694/#695. The whole suite (103 files, including 23 that
boot a throwaway Postgres per test) ran on every pull_request event. The
every-PR gate is now the deterministic fast core: scripts/ci/classify_tests.py
selects the 80 files that touch neither managed_pg_server/managed_pool nor a
browser - 728 tests, measured 46s locally. The full suite keeps its depth on
workflow_dispatch, weekly (Sunday 02:53 UTC, staggered after delivery-api's
02:41 slot), and ready_for_review.

Every fast-gate file already ran in the old PR job, so the gate is a strict
subset of previous PR coverage. Misclassification degrades safely: a PG test
in the fast set just boots its throwaway server (slower, still correct); a
fast file classified heavy loses per-PR coverage until the weekly run.
